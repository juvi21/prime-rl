from copy import deepcopy

import verifiers as vf

from prime_rl.orchestrator.types import TrainingExample
from prime_rl.utils.logger import get_logger


def interleave_rollout(state: vf.State) -> list[TrainingExample]:
    """
    Convert vf.State to a *single* trainable rollout by interleaving the trajectory.

    For multi-turn conversations, this function uses prompt_logprobs from the final turn
    to ensure correct importance ratio computation. This fixes the re-tokenization issue
    where past assistant tokens may have different IDs when re-tokenized.

    The key insight: The final turn's prompt_ids and prompt_logprobs are aligned because
    they come from the same vLLM tokenization run. Using these for past turns ensures
    the importance ratio is computed correctly.
    """
    logger = get_logger()
    trajectory = state["trajectory"]

    # Single-turn case: use original behavior (no re-tokenization issue)
    if len(trajectory) == 1:
        first_step = trajectory[0]
        return [
            TrainingExample(
                prompt_ids=deepcopy(first_step["tokens"]["prompt_ids"]),
                prompt_mask=deepcopy(first_step["tokens"]["prompt_mask"]),
                completion_ids=deepcopy(first_step["tokens"]["completion_ids"]),
                completion_mask=deepcopy(first_step["tokens"]["completion_mask"]),
                completion_logprobs=deepcopy(first_step["tokens"]["completion_logprobs"]),
                advantage=None,
            )
        ]

    # Multi-turn case: use final turn's tokenization as source of truth
    first_step = trajectory[0]
    final_step = trajectory[-1]
    final_tokens = final_step["tokens"]
    assert final_tokens is not None

    # Get the final turn's data (aligned tokenization)
    final_prompt_ids = final_tokens["prompt_ids"]
    final_prompt_logprobs = final_tokens.get("prompt_logprobs")
    final_completion_ids = final_tokens["completion_ids"]
    final_completion_logprobs = final_tokens["completion_logprobs"]

    # First turn's prompt length (this is the "original" prompt before any completions)
    first_prompt_len = len(first_step["tokens"]["prompt_ids"])

    # Fall back to legacy behavior if prompt_logprobs not available
    if final_prompt_logprobs is None:
        logger.warning(
            f"prompt_logprobs not available for example {state.get('example_id', 'unknown')}. "
            f"Using legacy interleave logic."
        )
        return _interleave_rollout_legacy(state)

    # Build the completion_ids from final turn's prompt (minus first prompt) + final completion
    # This uses the re-tokenized sequence which is aligned with prompt_logprobs
    completion_ids = deepcopy(final_prompt_ids[first_prompt_len:]) + deepcopy(final_completion_ids)

    # Build completion_logprobs: use prompt_logprobs for past tokens, completion_logprobs for final turn
    completion_logprobs = deepcopy(final_prompt_logprobs[first_prompt_len:]) + deepcopy(final_completion_logprobs)

    # Build completion_mask: need to identify which tokens are from model completions vs user/system
    completion_mask = _build_completion_mask_from_trajectory(
        trajectory, first_prompt_len, final_prompt_ids, final_completion_ids
    )

    # Validate lengths match
    if len(completion_ids) != len(completion_logprobs):
        logger.warning(
            f"Length mismatch in example {state.get('example_id', 'unknown')}: "
            f"completion_ids={len(completion_ids)}, completion_logprobs={len(completion_logprobs)}"
        )
        # Truncate to minimum length to avoid errors
        min_len = min(len(completion_ids), len(completion_logprobs))
        completion_ids = completion_ids[:min_len]
        completion_logprobs = completion_logprobs[:min_len]
        completion_mask = completion_mask[:min_len]

    interleaved_rollout = TrainingExample(
        prompt_ids=deepcopy(final_prompt_ids[:first_prompt_len]),
        prompt_mask=[0] * first_prompt_len,
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        completion_logprobs=completion_logprobs,
        advantage=None,
    )
    return [interleaved_rollout]


def _interleave_rollout_legacy(state: vf.State) -> list[TrainingExample]:
    """
    Legacy interleave logic for backwards compatibility.

    Uses original token IDs and logprobs from each step. This is susceptible to
    re-tokenization issues in multi-turn scenarios where BPE may produce different
    token IDs for the same text when context changes.
    """
    trajectory = state["trajectory"]
    first_step = trajectory[0]
    first_prompt_len = len(first_step["tokens"]["prompt_ids"])

    completion_ids: list[int] = []
    completion_logprobs: list[float] = []
    completion_mask: list[int] = []

    for step_idx, step in enumerate(trajectory):
        tokens = step["tokens"]
        assert tokens is not None

        if step_idx == 0:
            # First step: add completion directly
            completion_ids.extend(deepcopy(tokens["completion_ids"]))
            completion_logprobs.extend(deepcopy(tokens["completion_logprobs"]))
            completion_mask.extend([1] * len(tokens["completion_ids"]))
        else:
            # Later steps: add new prompt tokens (user/tool), then completion
            prev_tokens = trajectory[step_idx - 1]["tokens"]
            prev_total_len = len(prev_tokens["prompt_ids"]) + len(prev_tokens["completion_ids"])
            new_prompt_len = len(tokens["prompt_ids"]) - prev_total_len

            if new_prompt_len > 0:
                new_prompt_ids = tokens["prompt_ids"][prev_total_len:]
                completion_ids.extend(deepcopy(new_prompt_ids))
                completion_logprobs.extend([0.0] * new_prompt_len)
                completion_mask.extend([0] * new_prompt_len)

            completion_ids.extend(deepcopy(tokens["completion_ids"]))
            completion_logprobs.extend(deepcopy(tokens["completion_logprobs"]))
            completion_mask.extend([1] * len(tokens["completion_ids"]))

    interleaved_rollout = TrainingExample(
        prompt_ids=deepcopy(first_step["tokens"]["prompt_ids"]),
        prompt_mask=[0] * first_prompt_len,
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        completion_logprobs=completion_logprobs,
        advantage=None,
    )
    return [interleaved_rollout]


def _build_completion_mask_from_trajectory(
    trajectory: list[vf.TrajectoryStep],
    first_prompt_len: int,
    final_prompt_ids: list[int],
    final_completion_ids: list[int],
) -> list[int]:
    """
    Build a mask indicating which tokens should contribute to the loss.

    Only tokens from actual model completions get mask=1.
    User/system/tool messages and prompts get mask=0.

    This works by tracking the cumulative length of prompt+completion for each step,
    and marking the completion portions.
    """
    total_completion_len = len(final_prompt_ids) - first_prompt_len + len(final_completion_ids)
    mask = [0] * total_completion_len

    # Track position in the completion sequence
    # Start after the first prompt
    current_pos = 0

    for step_idx, step in enumerate(trajectory):
        tokens = step["tokens"]
        assert tokens is not None

        if step_idx == 0:
            # First step: mark its completion as trainable
            completion_len = len(tokens["completion_ids"])
            for i in range(completion_len):
                if current_pos + i < len(mask):
                    mask[current_pos + i] = 1
            current_pos += completion_len
        else:
            # Subsequent steps: first add the "new prompt" portion (mask=0), then completion (mask=1)
            prev_total_len = len(trajectory[step_idx - 1]["tokens"]["prompt_ids"]) + len(
                trajectory[step_idx - 1]["tokens"]["completion_ids"]
            )
            new_prompt_len = len(tokens["prompt_ids"]) - prev_total_len
            current_pos += new_prompt_len  # Skip new prompt tokens (already 0)

            # Mark completion tokens as trainable
            completion_len = len(tokens["completion_ids"])
            for i in range(completion_len):
                if current_pos + i < len(mask):
                    mask[current_pos + i] = 1
            current_pos += completion_len

    return mask


def branch_rollout(state: vf.State) -> list[TrainingExample]:
    """Convert vf.State to *multiple* trainable rollouts using branching trajectories strategy."""
    rollouts = []
    for step in state["trajectory"]:
        assert "tokens" in step
        tokens = step["tokens"]
        rollout = TrainingExample(
            prompt_ids=deepcopy(tokens["prompt_ids"]),
            prompt_mask=deepcopy(tokens["prompt_mask"]),
            completion_ids=deepcopy(tokens["completion_ids"]),
            completion_mask=deepcopy(tokens["completion_mask"]),
            completion_logprobs=deepcopy(tokens["completion_logprobs"]),
            advantage=None,
        )
        rollouts.append(rollout)
    return rollouts
