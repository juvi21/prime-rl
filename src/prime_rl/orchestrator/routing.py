from __future__ import annotations

from typing import Any


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def attach_routing_to_state(state: Any) -> None:
    """Attach MoE routing metadata from response logprobs into vf.State tokens.

    Expects vLLM to populate `experts` on prompt_logprobs and output logprobs
    entries. This function is a no-op if routing is not present.
    """
    for step in state.get("trajectory", []):
        tokens = step.get("tokens")
        response = step.get("response")
        if tokens is None or response is None:
            continue

        prompt_ids = tokens.get("prompt_ids") or []
        completion_ids = tokens.get("completion_ids") or []

        # Prompt experts from top-level prompt_logprobs list.
        prompt_logprobs = _get_attr(response, "prompt_logprobs")
        prompt_expert_indices = []
        prompt_expert_probs = []
        if prompt_logprobs is not None:
            try:
                for pos, tok_id in enumerate(prompt_ids):
                    lp_dict = prompt_logprobs[pos]
                    if lp_dict is None:
                        raise ValueError("missing prompt logprobs")
                    lp_obj = lp_dict.get(tok_id) if isinstance(lp_dict, dict) else None
                    experts = _get_attr(lp_obj, "experts")
                    if experts is None:
                        raise ValueError("missing experts")
                    prompt_expert_indices.append([layer["ids"] for layer in experts])
                    prompt_expert_probs.append([layer["probs"] for layer in experts])
            except Exception:
                prompt_expert_indices = None
                prompt_expert_probs = None
        else:
            prompt_expert_indices = None
            prompt_expert_probs = None

        # Completion experts from choice.logprobs.content.
        choice = _get_attr(response, "choices", [None])[0]
        choice_logprobs = _get_attr(choice, "logprobs")
        content = _get_attr(choice_logprobs, "content")
        completion_expert_indices = []
        completion_expert_probs = []
        if content is not None:
            try:
                for item in content[: len(completion_ids)]:
                    experts = _get_attr(item, "experts")
                    if experts is None:
                        raise ValueError("missing experts")
                    completion_expert_indices.append([layer["ids"] for layer in experts])
                    completion_expert_probs.append([layer["probs"] for layer in experts])
            except Exception:
                completion_expert_indices = None
                completion_expert_probs = None
        else:
            completion_expert_indices = None
            completion_expert_probs = None

        tokens["prompt_expert_indices"] = prompt_expert_indices
        tokens["prompt_expert_probs"] = prompt_expert_probs
        tokens["completion_expert_indices"] = completion_expert_indices
        tokens["completion_expert_probs"] = completion_expert_probs

