from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Iterable, TypeVar


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


def parse_key_value(item: str) -> tuple[str, Any]:
    """Parse a ``KEY=VALUE`` CLI item into a config key and a Python value."""
    if "=" not in item:
        raise ValueError(f"参数 {item!r} 格式应为 KEY=VALUE")
    key, raw = item.split("=", 1)
    key = key.strip()
    raw = raw.strip()
    if raw in ("True", "true"):
        value = True
    elif raw in ("False", "false"):
        value = False
    elif raw in ("None", "null"):
        value = None
    else:
        try:
            value = int(raw)
        except ValueError:
            try:
                value = float(raw)
            except ValueError:
                value = raw
    return key, value


def parse_key_values(items: Iterable[str] | None) -> dict[str, Any]:
    """Parse repeated ``KEY=VALUE`` CLI items into a dictionary."""
    if not items:
        return {}
    return dict(parse_key_value(item) for item in items)
