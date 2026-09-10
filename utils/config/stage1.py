from dataclasses import dataclass, field
from typing import Optional

from ..config import BaseConfig


@dataclass
class Config(BaseConfig):
    """Stage1：单簇/单车 TSP 路由模型训练配置。"""

    env: str = "TSPEnv"
    model: str = "AttentionModel"
    num_loc: int = 50
    embed_dim: int = 128
    num_encoder_layers: int = 3
    num_heads: int = 8
    normalization: str = "batch"
    baseline: str = "auto"  # auto: POMO->shared, 其余->rollout
    weight_decay: float = 0.0
    lr_scheduler: str = "none"  # none / multi_step
    lr_milestones: list[int] = field(default_factory=lambda: [501])
    lr_gamma: float = 0.1
    logit_clipping: float = 10.0
    feedforward_hidden: int = 512
    num_starts: Optional[int] = None
    generator_params: dict = field(default_factory=dict)
    seed: int = 1234
    val_batch_size: Optional[int] = None
    test_batch_size: Optional[int] = None

    # 固定数据文件（相对 data_dir 的文件名；不设置时训练/验证数据随机生成）
    data_dir: str = "data"
    train_file: Optional[str] = None
    val_file: Optional[str] = None
    test_file: Optional[str] = None
