#!/usr/bin/env python3
"""Run ABIDE download + SINDy analysis from terminal.

This script mirrors the notebook workflow in `abide_sindy.ipynb`, but it is
terminal-friendly and adds retry logic for unstable internet connections.
"""

from __future__ import annotations

import argparse
import json
import os
import tarfile
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from nilearn import datasets

from sindy import SindyArgs, process_data_in_windows


def parse_n_subjects(raw_value: str) -> int | None:
    """Parse user-provided n_subjects.

    Use `all` to download all available subjects.
    """
    value = raw_value.strip().lower()
    if value == "all":
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("n_subjects must be a positive integer or 'all'.")
    return parsed


def fetch_abide_with_retries(
    data_dir: Path,
    n_subjects: int | None,
    derivatives: str,
    retries: int,
    retry_wait_sec: int,
):
    """Fetch ABIDE PCP with basic retry/backoff for unreliable connections."""
    last_exc: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            print(f"[download] Attempt {attempt}/{retries}...")
            abide = datasets.fetch_abide_pcp(
                data_dir=str(data_dir),
                n_subjects=n_subjects,
                derivatives=derivatives,
                verbose=1,
            )
            print("[download] Download completed successfully.")
            return abide
        except Exception as exc:  # noqa: BLE001 - downloader can raise varied errors
            last_exc = exc
            print(f"[download] Attempt {attempt} failed: {exc}")
            if attempt < retries:
                wait_s = retry_wait_sec * attempt
                print(f"[download] Retrying in {wait_s}s...")
                time.sleep(wait_s)

    raise RuntimeError(f"ABIDE download failed after {retries} attempts: {last_exc}")


def summarize_timeseries(ts_data: np.ndarray) -> None:
    print("--- Data Statistics ---")
    print(f"Mean: {np.mean(ts_data):.4f}")
    print(f"Std Dev: {np.std(ts_data):.4f}")
    print(f"Min: {np.min(ts_data):.4f}, Max: {np.max(ts_data):.4f}")


def save_roi_plot(ts_data: np.ndarray, output_path: Path) -> None:
    plt.figure(figsize=(12, 5))
    for i in range(min(10, ts_data.shape[1])):
        plt.plot(ts_data[:, i], label=f"ROI {i + 1}")
    plt.title("Sample ROI Time Series from ABIDE (First 10 ROIs)")
    plt.xlabel("Timepoints")
    plt.ylabel("Signal Amplitude")
    plt.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
    plt.tight_layout()
    plt.savefig(output_path, dpi=140)
    plt.close()
    print(f"[output] Saved ROI plot to: {output_path}")


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, frozenset):
        return sorted(list(value))
    if isinstance(value, set):
        return sorted(list(value))
    if isinstance(value, dict):
        return {str(key): _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def _safe_filename(text: str) -> str:
    cleaned = [ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text]
    return "".join(cleaned).strip("._") or "sample"


def _sample_label(abide: Any, index: int) -> str:
    for attr_name in ("subject_id", "subject_ids", "subjects", "site_id", "ids"):
        values = getattr(abide, attr_name, None)
        if values is None:
            continue
        try:
            value = values[index]
        except Exception:  # noqa: BLE001 - dataset objects vary across nilearn versions
            continue
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="ignore")
        return str(value)
    return f"sample_{index:04d}"


def run_sindy(ts_data: np.ndarray, window_len: int, overlap: float, max_rois: int) -> dict[str, Any]:
    data_raw = ts_data.T
    data_raw = data_raw[:max_rois, :]
    print(f"Data shape for SINDy [Nodes, Timepoints]: {data_raw.shape}")

    args = SindyArgs(
        win_len=window_len,
        stride=None,
        overlap=overlap,
        d_max=2,
        simpl_rho=1.0,
        scale=1.0,
        tau2_q=0.50,
        tau3_q=0.50,
    )

    stride = max(1, int(round(window_len * (1.0 - overlap))))
    print(f"Data will be split into overlapping windows of length {window_len} and stride {stride}.")
    print("Starting processing over windows...")
    results = process_data_in_windows(data_raw, args)
    print(f"Processed {len(results)} windows.")

    all_edges = [set(tuple(edge) for edge in res["edges"]) for res in results]
    all_tris = [set(tuple(tri) for tri in res["triangles"]) for res in results]

    print("\n" + "=" * 50)
    print("TEMPORAL CHANGES SUMMARY")
    print("=" * 50)
    for i in range(1, len(results)):
        dropped_edges = all_edges[i - 1] - all_edges[i]
        new_edges = all_edges[i] - all_edges[i - 1]

        dropped_tris = all_tris[i - 1] - all_tris[i]
        new_tris = all_tris[i] - all_tris[i - 1]

        print(f"\nTransition Window {i - 1} -> {i}:")
        print(f"  Edges Added   (+{len(new_edges)}): {list(new_edges)}")
        print(f"  Edges Dropped (-{len(dropped_edges)}): {list(dropped_edges)}")
        print(f"  Tris Added    (+{len(new_tris)}): {list(new_tris)}")
        print(f"  Tris Dropped  (-{len(dropped_tris)}): {list(dropped_tris)}")

    return {
        "window_len": window_len,
        "stride": stride,
        "num_windows": len(results),
        "windows": results,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download ABIDE PCP into the current directory (or chosen directory) "
            "and run the SINDy workflow from abide_sindy.ipynb."
        )
    )
    parser.add_argument(
        "--download-dir",
        type=str,
        default=".",
        help="Directory where ABIDE data will be downloaded (default: current directory).",
    )
    parser.add_argument(
        "--n-subjects",
        type=str,
        default="1",
        help="Number of subjects to fetch, or 'all' for entire dataset (default: 1).",
    )
    parser.add_argument(
        "--derivatives",
        type=str,
        default="rois_ho",
        help="ABIDE derivative to fetch (default: rois_ho).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=5,
        help="Number of download retry attempts (default: 5).",
    )
    parser.add_argument(
        "--retry-wait-sec",
        type=int,
        default=15,
        help="Base wait in seconds before retries (default: 15).",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download dataset and exit without SINDy processing.",
    )
    parser.add_argument(
        "--window-len",
        type=int,
        default=50,
        help="Fixed window length in timepoints (default: 50).",
    )
    parser.add_argument(
        "--window-overlap",
        type=float,
        default=0.5,
        help="Fractional overlap between consecutive windows, in [0, 1). Default: 0.5.",
    )
    parser.add_argument(
        "--max-rois",
        type=int,
        default=20,
        help="Max ROIs to use for SINDy (default: 20).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    cli = parser.parse_args()

    download_dir = Path(cli.download_dir).resolve()
    download_dir.mkdir(parents=True, exist_ok=True)

    # Ensure nilearn uses this directory for cache/data files.
    os.environ["NILEARN_DATA"] = str(download_dir)

    n_subjects = parse_n_subjects(cli.n_subjects)

    print(f"[config] Download directory: {download_dir}")
    print(f"[config] n_subjects: {'all' if n_subjects is None else n_subjects}")
    print(f"[config] derivatives: {cli.derivatives}")

    if cli.window_len <= 0:
        raise ValueError("--window-len must be a positive integer.")
    if not (0.0 <= cli.window_overlap < 1.0):
        raise ValueError("--window-overlap must be in the range [0, 1).")

    abide = fetch_abide_with_retries(
        data_dir=download_dir,
        n_subjects=n_subjects,
        derivatives=cli.derivatives,
        retries=cli.retries,
        retry_wait_sec=cli.retry_wait_sec,
    )

    if cli.download_only:
        print("[done] Download completed (download-only mode).")
        return

    if not getattr(abide, "rois_ho", None):
        raise RuntimeError(
            "No ROI time series found in downloaded ABIDE data. "
            "Try a different derivative or number of subjects."
        )

    processed_dir = download_dir / "abide_sindy_processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    sample_payloads: list[dict[str, Any]] = []
    for sample_index, ts_data in enumerate(abide.rois_ho):
        sample_label = _sample_label(abide, sample_index)
        print("\n" + "=" * 80)
        print(f"[sample] {sample_index + 1}/{len(abide.rois_ho)}: {sample_label}")
        print(f"Time series shape (Timepoints, ROIs): {ts_data.shape}")

        summarize_timeseries(ts_data)
        if sample_index == 0:
            save_roi_plot(ts_data, download_dir / "abide_roi_preview.png")

        try:
            sample_result = run_sindy(
                ts_data,
                window_len=cli.window_len,
                overlap=cli.window_overlap,
                max_rois=cli.max_rois,
            )
        except ValueError as exc:
            print(f"[skip] {sample_label}: {exc}")
            continue

        sample_payload = {
            "sample_index": sample_index,
            "sample_id": sample_label,
            "timepoints": int(ts_data.shape[0]),
            "rois": int(ts_data.shape[1]),
            **sample_result,
        }
        sample_payloads.append(sample_payload)

        sample_path = processed_dir / f"{sample_index:04d}_{_safe_filename(sample_label)}.json"
        with sample_path.open("w", encoding="utf-8") as fp:
            json.dump(_to_jsonable(sample_payload), fp, indent=2)
        print(f"[output] Saved sample result to: {sample_path}")

    manifest = {
        "download_dir": str(download_dir),
        "n_subjects": "all" if n_subjects is None else n_subjects,
        "derivatives": cli.derivatives,
        "window_len": cli.window_len,
        "window_overlap": cli.window_overlap,
        "max_rois": cli.max_rois,
        "num_samples_processed": len(sample_payloads),
        "samples": [
            {
                "sample_index": payload["sample_index"],
                "sample_id": payload["sample_id"],
                "num_windows": payload["num_windows"],
                "window_len": payload["window_len"],
                "stride": payload["stride"],
            }
            for payload in sample_payloads
        ],
    }

    manifest_path = processed_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fp:
        json.dump(_to_jsonable(manifest), fp, indent=2)
    print(f"[output] Saved manifest to: {manifest_path}")

    archive_path = download_dir / "abide_sindy_processed.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(processed_dir, arcname="abide_sindy_processed")
    print(f"[output] Saved tar archive to: {archive_path}")

    print("[done] ABIDE + SINDy pipeline completed.")


if __name__ == "__main__":
    main()
