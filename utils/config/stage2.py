from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..config import BaseConfig


@dataclass
class Config(BaseConfig):
    """Stage2：CVRP 分簇 + 冻结 Stage1 TSP 路由评价器。"""

    # Stage2 完整问题数据来自 RL4CO CVRPEnv
    env: str = "CVRPEnv"
    num_loc: int = 50
    capacity: Optional[float] = None  # None 表示使用 RL4CO 按节点数的默认容量
    num_vehicles: Optional[int] = 8
    num_vehicles_min: Optional[int] = None
    num_vehicles_max: Optional[int] = None
    vehicle_count_margin: int = 0
    objective: str = "minmax"  # sum / minmax
    penalty: float = 10.0
    capacity_mask_mode: str = "static"  # none / static
    baseline: str = "rollout"

    # 分簇网络结构（2D-Ptr + DPN PE）
    embed_dim: int = 128
    num_encoder_layers: int = 3
    num_heads: int = 8
    temperature: float = 1.0

    # Stage1 路由评价器
    route_model: str = "AttentionModel"
    route_checkpoint: Optional[str] = None
    route_embed_dim: int = 128
    route_num_encoder_layers: int = 3
    route_num_heads: int = 8
    route_normalization: str = "batch"
    route_pad_to: int = 0  # 0: pad to the largest cluster in the batch
    route_num_starts: int = 0  # 0: use the padded TSP length for POMO
    route_eval_batch_size: int = 0  # 0: evaluate all clusters in one TSP batch

    # Stage2 默认按小规模可跑通配置；大规模训练请自行调大
    batch_size: int = 32
    train_data_size: int = 2_048
    val_data_size: int = 256

    # 固定数据文件
    data_dir: str = "data"
    train_file: Optional[str] = None
    val_file: Optional[str] = None
    test_file: Optional[str] = None

    # 其他生成参数（KEY=VALUE，可重复），例如 min_demand/max_demand/loc_distribution
    generator_params: dict = field(default_factory=dict)
