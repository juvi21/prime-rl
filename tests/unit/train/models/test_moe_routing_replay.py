import pytest
import torch

from prime_rl.trainer.models.layers.moe import MoE, MoEArgs

pytestmark = [pytest.mark.gpu]


def test_moe_routing_replay_matches_normal() -> None:
    moe_args = MoEArgs(
        num_experts=8,
        num_shared_experts=0,
        score_func="softmax",
        route_norm=True,
        route_scale=1.0,
        top_k=2,
        use_grouped_mm=False,
        load_balance_coeff=None,
    )
    with torch.device("cuda"):
        moe = MoE(moe_args, dim=32, hidden_dim=64).cuda()
        x = torch.randn(2, 5, 32, device="cuda")

    # Capture current routing decisions.
    top_scores, indices, _ = moe.router(x.view(-1, 32), moe.expert_bias)
    override = {
        "indices": indices.view(2, 5, moe_args.top_k),
        "probs": top_scores.view(2, 5, moe_args.top_k),
    }

    out_normal = moe(x)
    out_replay = moe(x, routing_override=override)

    assert torch.allclose(out_normal, out_replay, atol=1e-5), (
        f"Max diff: {(out_normal - out_replay).abs().max()}"
    )


def test_moe_router_logprobs_track_current_policy() -> None:
    moe_args = MoEArgs(
        num_experts=4,
        num_shared_experts=0,
        score_func="softmax",
        route_norm=False,
        route_scale=1.0,
        top_k=1,
        use_grouped_mm=False,
        load_balance_coeff=None,
    )
    with torch.device("cuda"):
        moe = MoE(moe_args, dim=16, hidden_dim=32).cuda()
        x = torch.randn(1, 6, 16, device="cuda")

    top_scores, indices, _ = moe.router(x.view(-1, 16), moe.expert_bias)
    override = {
        "indices": indices.view(1, 6, moe_args.top_k),
        "probs": top_scores.view(1, 6, moe_args.top_k),
    }

    _, router_lp1 = moe(x, routing_override=override, return_router_logprobs=True)

    with torch.no_grad():
        moe.router.gate.weight.add_(0.1 * torch.randn_like(moe.router.gate.weight))

    _, router_lp2 = moe(x, routing_override=override, return_router_logprobs=True)

    assert not torch.allclose(router_lp1, router_lp2), "Router logprobs should change with router weights."

