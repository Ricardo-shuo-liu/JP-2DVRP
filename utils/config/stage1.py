from dataclasses import dataclass
from typing import Optional

from ..config import BaseConfig


@dataclass
class Config(BaseConfig):
    """Stage1 选择的具体 RL4CO 环境与模型。"""

    env: str = "TSPEnv"
    models: str = "AttentionModel"
    num_loc: int = 50
    embed_dim: int = 128
    num_encoder_layers: int = 3
    num_heads: int = 8

    # 固定数据文件（相对 data_dir 的文件名；不设置时训练/验证数据随机生成）
    data_dir: str = "data"
    train_file: Optional[str] = None
    val_file: Optional[str] = None
    test_file: Optional[str] = None
