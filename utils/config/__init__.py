from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, TypeVar


ConfigType = TypeVar("ConfigType", bound="BaseConfig")


@dataclass
class BaseConfig:
    """各 stage 共用的训练参数。"""

    log_dir: str = "logs"
    accelerator: str = "auto"
    devices: int = 1
    precision: str = "16-mixed"
    logger: str = "csv"

    epochs: int = 3
    batch_size: int = 512
    train_data_size: int = 100_000
    val_data_size: int = 10_000
    lr: float = 1e-4

    @classmethod
    def from_dict(cls: type[ConfigType], data: dict[str, Any]) -> ConfigType:
        """从普通 dict 构造配置，只取 dataclass 中存在的字段。"""
        return cls(
            **{
                item.name: data[item.name]
                for item in fields(cls)
                if item.name in data
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
