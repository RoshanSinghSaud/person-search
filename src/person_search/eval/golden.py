"""Golden (characterization) data for the 4 test cases.

A golden file pins the frames a reference run selected, in sampled-frame space
(the index of ``output_annotated.mp4`` and of ``evaluation.json``)::

    {
      "case": "test_case_001",
      "source": "repo 2 run_reid, untouched, default config",
      "config": {...},
      "selected_samples": [ ... ],
      "tracks": {"<sample>": "<track id>", ...}
    }

``make golden`` re-runs ``pipeline.run`` and requires an exact match.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class GoldenCase:
    case: str
    selected_samples: list[int]
    source: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    tracks: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> GoldenCase:
        data = json.loads(Path(path).read_text())
        return cls(
            case=data["case"],
            selected_samples=sorted(int(x) for x in data["selected_samples"]),
            source=data.get("source", ""),
            config=data.get("config", {}),
            tracks={str(k): str(v) for k, v in data.get("tracks", {}).items()},
        )

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(
                {
                    "case": self.case,
                    "source": self.source,
                    "config": self.config,
                    "selected_samples": sorted(self.selected_samples),
                    "tracks": self.tracks,
                },
                indent=1,
            )
        )


@dataclass(frozen=True)
class GoldenDiff:
    missing: tuple[int, ...]
    extra: tuple[int, ...]

    @property
    def ok(self) -> bool:
        return not self.missing and not self.extra

    def __str__(self) -> str:
        if self.ok:
            return "identical"
        return (
            f"missing {len(self.missing)} {list(self.missing)[:15]}, "
            f"extra {len(self.extra)} {list(self.extra)[:15]}"
        )


def compare(expected: GoldenCase, selected_samples: list[int]) -> GoldenDiff:
    exp, got = set(expected.selected_samples), set(selected_samples)
    return GoldenDiff(tuple(sorted(exp - got)), tuple(sorted(got - exp)))
