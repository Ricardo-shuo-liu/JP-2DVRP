"""Stage1 TSP route model used as a frozen reward evaluator in Stage2."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from utils.data import make_env
from utils.config.stage2 import Config


SUPPORTED_ROUTE_MODELS = ("AttentionModel", "POMO", "MDAM", "PointerNetwork")


def _policy_kwargs(model_name: str, config: Config) -> dict:
    if model_name == "PointerNetwork":
        return {
            "embed_dim": config.route_embed_dim,
            "hidden_dim": config.route_embed_dim,
        }
    return {
        "embed_dim": config.route_embed_dim,
        "num_encoder_layers": config.route_num_encoder_layers,
        "num_heads": config.route_num_heads,
        "normalization": config.route_normalization,
    }


def build_fresh_route_model(config: Config):
    """Build a fresh top-level RL4CO model matching the requested architecture."""
    from rl4co import models

    model_cls = getattr(models, config.route_model, None)
    if model_cls is None or config.route_model not in SUPPORTED_ROUTE_MODELS:
        raise ValueError(
            f"不支持 route 模型 {config.route_model!r}；"
            f"可用: {', '.join(SUPPORTED_ROUTE_MODELS)}"
        )

    env = make_env(
        "TSPEnv",
        num_loc=max(config.num_loc // max(config.num_vehicles, 1), 5),
        check_solution=False,
    )
    model = model_cls(
        env,
        policy_kwargs=_policy_kwargs(config.route_model, config),
        baseline="shared" if config.route_model == "POMO" else "rollout",
        batch_size=8,
        train_data_size=64,
        val_data_size=16,
        test_data_size=16,
        optimizer_kwargs={"lr": config.lr},
    )
    return model, env


def _load_state_into_model(model, checkpoint: str) -> None:
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = raw.get("state_dict", raw)
    if not isinstance(state, dict) or not state:
        raise ValueError(f"{checkpoint} 中未找到可加载的 state_dict")

    model_keys = set(model.state_dict())
    ckpt_policy_keys = {
        key for key in state if key.startswith("policy.") and key in model_keys
    }
    missing_policy = {key for key in model_keys if key.startswith("policy.")} - set(state)
    if not ckpt_policy_keys:
        raise ValueError(
            f"{checkpoint} 中没有与模型匹配的 policy.* 参数；请检查 --route-model 是否与 Stage1 一致"
        )
    if missing_policy:
        missing = sorted(missing_policy)[:5]
        raise ValueError(
            f"checkpoint 与 route 模型结构不匹配，缺失 policy 参数: {missing}"
        )
    model.load_state_dict(state, strict=False)


class RouteEvaluator:
    """Evaluate TSP routes of every cluster using a frozen Stage1 model."""

    def __init__(self, config: Config):
        self.config = config
        model, self.route_env = build_fresh_route_model(config)
        if config.route_checkpoint:
            _load_state_into_model(model, config.route_checkpoint)
        self.model = model
        self.policy = model.policy
        self.model.eval()
        for param in self.policy.parameters():
            param.requires_grad_(False)

    def _route_reward(self, cluster_td: TensorDict) -> torch.Tensor:
        """Return best reward for a single cluster subproblem."""
        name = self.config.route_model
        if name in ("AttentionModel", "POMO", "PointerNetwork"):
            out = self.policy(
                cluster_td,
                self.route_env,
                phase="test",
                decode_type="greedy",
            )
            reward = out["reward"]
            if reward.dim() > 1:
                reward = reward.max(-1).values
            return reward
        if name == "MDAM":
            out = self.policy(
                cluster_td,
                self.route_env,
                phase="test",
                decode_type="greedy",
            )
            reward = out["reward"]  # [B, num_paths]
            return reward.max(-1).values
        raise NotImplementedError(name)

    @torch.no_grad()
    def evaluate(
        self, td: TensorDict, assignments: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate all clusters and return (total cost, capacity overflow)."""
        device = td.device
        if next(self.policy.parameters()).device != device:
            self.policy = self.policy.to(device)
            self.model = self.model.to(device)
            self.route_env = self.route_env.to(device)
        assignments = assignments.to(device)
        batch_size, num_nodes = assignments.shape
        num_vehicles = self.config.num_vehicles
        if assignments.max() >= num_vehicles or assignments.min() < 0:
            raise ValueError("assignment 超出车辆编号范围")

        depot = td["locs"][:, :1, :]
        customer_locs = td["locs"][:, 1:, :]
        demand = td["demand"]
        if demand.dim() == 3:
            demand = demand.squeeze(-1)
        if demand.dim() != 2:
            demand = demand.view(batch_size, num_nodes)

        # Capacity overflow in normalized units
        capacity = td.get("vehicle_capacity", None)
        if capacity is None:
            capacity = torch.ones(batch_size, 1, device=device)
        capacity = capacity.reshape(batch_size, 1)
        cluster_demand = torch.zeros(
            batch_size, num_vehicles, device=device, dtype=demand.dtype
        )
        cluster_demand.scatter_add_(1, assignments, demand)
        overflow = torch.clamp(cluster_demand - capacity, min=0).sum(-1)

        costs = torch.zeros(batch_size, device=device)
        for b in range(batch_size):
            cost_b = 0.0
            for m in range(num_vehicles):
                member_idx = (assignments[b] == m).nonzero(as_tuple=False).squeeze(-1)
                if member_idx.numel() == 0:
                    continue
                member_locs = customer_locs[b][member_idx]
                cluster_locs = torch.cat((depot[b], member_locs), dim=0)
                cluster_td = TensorDict(
                    {"locs": cluster_locs.unsqueeze(0)}, batch_size=[1], device=device
                )
                cluster_td = self.route_env.reset(cluster_td)
                reward = self._route_reward(cluster_td)
                cost_b = cost_b - reward.item()
            costs[b] = cost_b
        return costs, overflow
