"""Resolve the model-ready dataset directory used by command-line tools."""

from __future__ import annotations

import os
from pathlib import Path


def _is_dataset_root(path: Path) -> bool:
    return (path / "packed").is_dir() and (path / "smartroi_masks").is_dir()


def discover_dataset_root() -> Path | None:
    """Find a model-ready dataset in the standard release layout."""
    environment = os.environ.get("EGOSPEED_DATASET_ROOT")
    if environment:
        root = Path(environment).expanduser().resolve()
        if not _is_dataset_root(root):
            raise ValueError(
                "EGOSPEED_DATASET_ROOT does not contain packed/ and "
                f"smartroi_masks/: {root}"
            )
        return root

    repository_root = Path(__file__).resolve().parents[3]
    candidates = [
        repository_root.parent / "EgoSpeedDataset",
        Path.cwd() / "EgoSpeedDataset",
        Path.cwd().parent / "EgoSpeedDataset",
    ]
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate not in seen and _is_dataset_root(candidate):
            return candidate
        seen.add(candidate)
    return None


def resolve_dataset_roots(
    dataset_root: str | Path | None,
    data_root: str | Path | None,
    mask_root: str | Path | None,
) -> tuple[Path, Path]:
    """Return packed-data and SmartROI-mask roots from either CLI layout."""
    if dataset_root is not None:
        if data_root is not None or mask_root is not None:
            raise ValueError(
                "Use --dataset-root by itself, or pass both --data-root and --mask-root"
            )
        root = Path(dataset_root)
        return root / "packed", root / "smartroi_masks"

    if data_root is None and mask_root is None:
        root = discover_dataset_root()
        if root is None:
            raise ValueError(
                "Could not find EgoSpeedDataset. Place it next to the repository, "
                "set EGOSPEED_DATASET_ROOT, or pass --dataset-root."
            )
        return root / "packed", root / "smartroi_masks"
    if data_root is None or mask_root is None:
        raise ValueError("Pass both --data-root and --mask-root")
    return Path(data_root), Path(mask_root)
