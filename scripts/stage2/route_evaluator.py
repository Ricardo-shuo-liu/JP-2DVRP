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

    num_vehicles = (
        config.num_vehicles
        or config.num_vehicles_max
        or config.num_vehicles_min
        or 1
    )
    env = make_env(
        "TSPEnv",
        num_loc=max(config.num_loc // max(num_vehicles, 1), 5),
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

    def _route_rewards_chunk(self, cluster_td: TensorDict) -> torch.Tensor:
        """Return one reward per padded cluster for one TSP batch."""
        num_clusters = cluster_td.batch_size[0]
        name = self.config.route_model
        if name == "POMO":
            num_starts = int(self.config.route_num_starts)
            if num_starts <= 0:
                num_starts = self.route_env.get_num_starts(cluster_td)
            self.policy.test_decode_type = "multistart_greedy"
            out = self.policy(
                cluster_td,
                self.route_env,
                phase="test",
                num_starts=num_starts,
            )
            reward = out["reward"]
            # POMO returns start-major rewards: [num_starts, num_clusters].
            return reward.reshape(num_starts, num_clusters).amax(dim=0)

        if name in ("AttentionModel", "PointerNetwork"):
            out = self.policy(
                cluster_td,
                self.route_env,
                phase="test",
                decode_type="greedy",
            )
            return out["reward"].reshape(num_clusters)

        if name == "MDAM":
            out = self.policy(
                cluster_td,
                self.route_env,
                phase="test",
                decode_type="greedy",
            )
            return out["reward"].max(-1).values
        raise NotImplementedError(name)

    def _route_rewards(self, cluster_td: TensorDict) -> torch.Tensor:
        """Optionally split the cluster batch to control TSP evaluation memory."""
        total = cluster_td.batch_size[0]
        chunk = int(self.config.route_eval_batch_size)
        if chunk <= 0 or chunk >= total:
            return self._route_rewards_chunk(cluster_td)
        rewards = []
        for start in range(0, total, chunk):
            sub_td = cluster_td[start : start + chunk]
            rewards.append(self._route_rewards_chunk(sub_td))
        return torch.cat(rewards, dim=0)

    def _cluster_costs(
        self, td: TensorDict, assignments: torch.Tensor, num_vehicles: int
    ) -> torch.Tensor:
        """Evaluate all clusters in one padded batch."""
        batch_size, num_nodes = assignments.shape
        device = td.device

        depot = td["locs"][:, :1, :]
        customer_locs = td["locs"][:, 1:, :]

        # Sort nodes by cluster id and compute each node's position inside its
        # cluster without Python loops over (batch, vehicle).
        order = torch.argsort(assignments, dim=1, stable=True)
        sorted_assignments = assignments.gather(1, order)
        is_new_cluster = torch.ones_like(sorted_assignments, dtype=torch.bool)
        is_new_cluster[:, 1:] = (
            sorted_assignments[:, 1:] != sorted_assignments[:, :-1]
        )
        positions = torch.arange(num_nodes, device=device).expand(
            batch_size, num_nodes
        )
        cluster_starts = torch.where(
            is_new_cluster,
            positions,
            torch.zeros_like(positions),
        )
        cluster_starts = torch.cummax(cluster_starts, dim=1).values
        rank_sorted = positions - cluster_starts
        rank = torch.empty_like(rank_sorted)
        rank.scatter_(1, order, rank_sorted)

        cluster_counts = torch.zeros(
            batch_size, num_vehicles, device=device, dtype=torch.long
        )
        cluster_counts.scatter_add_(
            1,
            assignments,
            torch.ones_like(assignments, dtype=torch.long),
        )
        max_members = max(
            int(cluster_counts.max().item()),
            int(self.config.route_pad_to),
        )
        cluster_len = max(2, max_members + 1)
        padded = depot[:, None, :, :].expand(
            batch_size, num_vehicles, cluster_len, 2
        ).clone()
        batch_index = torch.arange(batch_size, device=device)[:, None].expand(
            batch_size, num_nodes
        )
        padded[batch_index, assignments, 1 + rank] = customer_locs

        cluster_td = TensorDict(
            {"locs": padded.reshape(batch_size * num_vehicles, cluster_len, 2)},
            batch_size=[batch_size * num_vehicles],
            device=device,
        )
        cluster_td = self.route_env.reset(cluster_td)
        rewards = self._route_rewards(cluster_td)
        return -rewards.reshape(batch_size, num_vehicles)

    @torch.no_grad()
    def evaluate(
        self,
        td: TensorDict,
        assignments: torch.Tensor,
        vehicle_counts: torch.Tensor | int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return per-vehicle costs, overflow, and the valid-vehicle mask."""
        device = td.device
        if next(self.policy.parameters()).device != device:
            self.policy = self.policy.to(device)
            self.model = self.model.to(device)
            self.route_env = self.route_env.to(device)
        assignments = assignments.to(device)
        batch_size, num_nodes = assignments.shape
        if vehicle_counts is None:
            vehicle_counts = torch.full(
                (batch_size,),
                int(assignments.max().item()) + 1,
                dtype=torch.long,
                device=device,
            )
        elif not torch.is_tensor(vehicle_counts):
            vehicle_counts = torch.full(
                (batch_size,),
                int(vehicle_counts),
                dtype=torch.long,
                device=device,
            )
        else:
            vehicle_counts = vehicle_counts.reshape(-1).to(device).long()
            if vehicle_counts.numel() == 1:
                vehicle_counts = vehicle_counts.expand(batch_size)
            if vehicle_counts.numel() != batch_size:
                raise ValueError(
                    "vehicle_counts 数量必须等于 batch_size"
                )

        max_vehicles = int(vehicle_counts.max().item())
        valid_vehicle = (
            torch.arange(max_vehicles, device=device)[None, :]
            < vehicle_counts[:, None]
        )
        if assignments.max() >= max_vehicles or assignments.min() < 0:
            raise ValueError("assignment 超出车辆编号范围")
        if not valid_vehicle.gather(
            1, assignments.clamp_max(max_vehicles - 1)
        ).all():
            raise ValueError("assignment 含无效车辆编号")

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
            batch_size, max_vehicles, device=device, dtype=demand.dtype
        )
        cluster_demand.scatter_add_(1, assignments, demand)
        overflow = torch.clamp(cluster_demand - capacity, min=0)
        overflow = overflow * valid_vehicle
        costs = self._cluster_costs(td, assignments, max_vehicles)
        return costs, overflow, valid_vehicle
