from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RunConfig:
    values: dict[str, Any]

    @classmethod
    def from_namespace(cls, ns: Any) -> "RunConfig":
        return cls(values=dict(vars(ns)))

    def require(self, key: str) -> Any:
        if key not in self.values:
            raise KeyError(f"Missing config key: {key}")
        return self.values[key]
