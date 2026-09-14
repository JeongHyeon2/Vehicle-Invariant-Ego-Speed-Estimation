"""Resolve the model-ready dataset directory used by command-line tools."""

from __future__ import annotations

from pathlib import Path


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

    if data_root is None or mask_root is None:
        raise ValueError(
            "Pass --dataset-root, or pass both --data-root and --mask-root"
        )
    return Path(data_root), Path(mask_root)
