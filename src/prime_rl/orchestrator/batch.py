import copy

import torch

from prime_rl.orchestrator.types import TensorTrainingExample, TrainingExample
from prime_rl.trainer.rl.data import MicroBatch


def prepare_sample(
    training_example: TrainingExample,
    seq_len: int,
) -> TensorTrainingExample:
    """
    Prepare a problem for sequence packing training.
    Tokenize and prepare tensors.
    """

    # Prepare prompt tokens
    prompt_token_ids = torch.tensor(training_example["prompt_ids"]).long()
    prompt_token_mask = torch.tensor(training_example["prompt_mask"]).long()

    # Prepare completion tokens
    completion_token_ids = torch.tensor(training_example["completion_ids"]).long()
    completion_token_mask = torch.tensor(training_example["completion_mask"]).long()

    # Prepare input_ids, loss_mask, position_ids, inference_logprobs, and advantages
    input_ids = torch.cat([prompt_token_ids, completion_token_ids]).long()
    loss_mask = torch.cat([prompt_token_mask, completion_token_mask]).bool()
    inference_logprobs = torch.cat(
        [torch.zeros(len(prompt_token_ids)), torch.tensor(training_example["completion_logprobs"])]
    ).float()
    advantages = torch.tensor(training_example["advantage"]).repeat(len(input_ids)).float()
    position_ids = torch.arange(len(input_ids)).long()

    # Prepare MoE routing overrides if present.
    moe_routing_overrides = None
    prompt_expert_indices = training_example.get("prompt_expert_indices")
    prompt_expert_probs = training_example.get("prompt_expert_probs")
    completion_expert_indices = training_example.get("completion_expert_indices")
    completion_expert_probs = training_example.get("completion_expert_probs")
    if (
        prompt_expert_indices is not None
        and prompt_expert_probs is not None
        and completion_expert_indices is not None
        and completion_expert_probs is not None
    ):
        expert_indices_full = prompt_expert_indices + completion_expert_indices
        expert_probs_full = prompt_expert_probs + completion_expert_probs
        if len(expert_indices_full) == len(input_ids):
            num_layers = len(expert_indices_full[0])
            moe_routing_overrides = {}
            for layer_idx in range(num_layers):
                indices_layer = [tok[layer_idx] for tok in expert_indices_full]
                probs_layer = [tok[layer_idx] for tok in expert_probs_full]
                moe_routing_overrides[layer_idx] = {
                    "indices": torch.tensor(indices_layer, dtype=torch.int64),
                    "probs": torch.tensor(probs_layer, dtype=torch.float32),
                }

    if len(input_ids) > seq_len:
        input_ids = input_ids[:seq_len]
        loss_mask = loss_mask[:seq_len]
        inference_logprobs = inference_logprobs[:seq_len]
        position_ids = position_ids[:seq_len]
        advantages = advantages[:seq_len]
        if moe_routing_overrides is not None:
            for layer_idx, override in moe_routing_overrides.items():
                moe_routing_overrides[layer_idx] = {
                    "indices": override["indices"][:seq_len],
                    "probs": override["probs"][:seq_len],
                }

    assert len(input_ids) == len(advantages) == len(loss_mask) == len(position_ids) == len(inference_logprobs), (
        f"input_ids: {len(input_ids)}, advantages: {len(advantages)}, loss_mask: {len(loss_mask)}, position_ids: {len(position_ids)}, inference_logprobs: {len(inference_logprobs)}"
    )
    return TensorTrainingExample(
        input_ids=input_ids,
        advantages=advantages,
        loss_mask=loss_mask,
        position_ids=position_ids,
        inference_logprobs=inference_logprobs,
        moe_routing_overrides=moe_routing_overrides,
    )


def prepare_micro_batch(samples: list[MicroBatch], temperature: float):
    micro_batch = {}

    for key in ["input_ids", "advantages", "loss_mask", "inference_logprobs", "position_ids"]:
        micro_batch[key] = torch.stack([sample[key] for sample in samples], dim=0)

    # Stack MoE routing overrides if present for all samples.
    overrides = [sample.get("moe_routing_overrides") for sample in samples]
    if all(o is not None for o in overrides):
        combined: dict[int, dict[str, torch.Tensor]] = {}
        for layer_idx in overrides[0].keys():  # type: ignore
            combined[layer_idx] = {
                "indices": torch.stack([o[layer_idx]["indices"] for o in overrides], dim=0),  # type: ignore
                "probs": torch.stack([o[layer_idx]["probs"] for o in overrides], dim=0),  # type: ignore
            }
        micro_batch["moe_routing_overrides"] = combined
    else:
        micro_batch["moe_routing_overrides"] = None

    micro_batch["temperature"] = temperature

    return micro_batch


def packed_samples_into_micro_bs(
    samples: list[TensorTrainingExample], max_seq_len: int
) -> list[list[TensorTrainingExample]]:
    """
    Pack samples into micro_batch efficiently.
    We follow the First Fit Decreasing algorithm to pack the samples into bins and minimize potential padding while never truncating.
    """
    sorted_samples = sorted(samples, key=lambda x: len(x["input_ids"]), reverse=True)

    ## we create bins
    micro_batches = []

    for sample in sorted_samples:
        # Try to find a bin that can fit this sequence
        bin_found = False
        for bin_idx, bin_content in enumerate(micro_batches):
            # Calculate current bin length
            bin_len = sum(len(s["input_ids"]) for s in bin_content)
            # Check if sequence fits in this bin
            if bin_len + len(sample["input_ids"]) <= max_seq_len:
                micro_batches[bin_idx].append(sample)
                bin_found = True
                break

        # If no suitable bin found, create a new bin
        if not bin_found:
            micro_batches.append([sample])

    return micro_batches


def prepare_micro_batch_packing(
    samples: list[TensorTrainingExample], max_seq_len: int, temperature: float
) -> MicroBatch:
    """
    Prepare a micro batch for packing mode. take multi sample and return a batch of shape [1, micro_bs * max_seq_len].
    Would additionally pad the batch to the max sequence length.
    """
    micro_batch = {}
    assert sum([len(sample["input_ids"]) for sample in samples]) <= max_seq_len, (
        "Total tokens of samples is greater than max sequence length"
    )

    for key in ["input_ids", "advantages", "loss_mask", "position_ids", "inference_logprobs"]:
        micro_batch[key] = torch.cat([sample[key] for sample in samples], dim=0).unsqueeze(0)

    # Concatenate MoE routing overrides if present.
    overrides_list = [sample.get("moe_routing_overrides") for sample in samples]
    if all(o is not None for o in overrides_list):
        combined: dict[int, dict[str, torch.Tensor]] = {}
        for layer_idx in overrides_list[0].keys():  # type: ignore
            combined[layer_idx] = {
                "indices": torch.cat([o[layer_idx]["indices"] for o in overrides_list], dim=0).unsqueeze(0),  # type: ignore
                "probs": torch.cat([o[layer_idx]["probs"] for o in overrides_list], dim=0).unsqueeze(0),  # type: ignore
            }
        micro_batch["moe_routing_overrides"] = combined
    else:
        micro_batch["moe_routing_overrides"] = None

    micro_batch["temperature"] = temperature

    return micro_batch


def prepare_batch(
    rollouts: list[TrainingExample],
    temperature: float,
    seq_len: int,
    num_train_workers: int,
) -> list[list[MicroBatch]]:
    """
    Prepare a batch of problems for each GPU. Each batch is a list of micro batches.
    Each micro batch is shape [1, seq_len], the namber of sample is not fixed per micro batch.
    """
    rollouts = copy.deepcopy(rollouts)
    max_seq_len = seq_len

    all_samples = [prepare_sample(rollout, max_seq_len) for rollout in rollouts]

    micro_batches_list = packed_samples_into_micro_bs(all_samples, max_seq_len)
    micro_batches = [
        prepare_micro_batch_packing(micro_batch, max_seq_len, temperature) for micro_batch in micro_batches_list
    ]

    num_padding_batch = -len(micro_batches) % num_train_workers

    # because of fsdp we need to make sure that each data ran has the same number of micro batches otherwise training will hang.
    # We create fake micro batches to fill the gap with real data but zero advantages, they would not contribute to the loss.
    if num_train_workers > 1 and num_padding_batch > 0:
        padded_batch = copy.deepcopy(micro_batches[0])
        padded_batch["advantages"] = torch.zeros_like(padded_batch["advantages"])
        padded_batch["loss_mask"] = torch.zeros_like(padded_batch["loss_mask"], dtype=torch.bool)
        micro_batches.extend([padded_batch for _ in range(num_padding_batch)])

    assert len(micro_batches) % num_train_workers == 0, (
        "Number of micro batches is not divisible by number of data ranks"
    )

    per_gpu_micro_batches = len(micro_batches) // num_train_workers
    batches_per_gpu = []
    for _ in range(num_train_workers):
        batches = []
        for _ in range(per_gpu_micro_batches):
            batches.append(micro_batches.pop(0))
        batches_per_gpu.append(batches)

    return batches_per_gpu
