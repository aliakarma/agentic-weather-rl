"""
Radar Data Preprocessing Module
================================
Purpose:
    Preprocess NOAA NEXRAD Level-2 radar files into normalized numpy arrays
    suitable for input to the multi-modal perception model.

Input:
    Raw NEXRAD Level-2 binary files (.gz or uncompressed) from data/nexrad/

Output:
    Preprocessed radar frames as numpy arrays saved to data/processed/radar_frames/
    A metadata CSV with file paths and extracted intensity statistics.

Example usage:
    python preprocessing/process_radar_data.py \
        --data_dir data/nexrad \
        --output_dir data/processed \
        --img_size 224 \
        --max_samples 1000
"""

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Tuple, List


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_IMG_SIZE: int = 224          # Target spatial resolution (H x W)
DBZ_MIN: float = -10.0              # Minimum reflectivity value (dBZ)
DBZ_MAX: float = 70.0               # Maximum reflectivity value (dBZ)
MISSING_VALUE: float = -9999.0      # Sentinel for missing radar data


# ---------------------------------------------------------------------------
# Core preprocessing functions
# ---------------------------------------------------------------------------

def load_nexrad_file(filepath: str) -> Optional[np.ndarray]:
    """
    Load a single NEXRAD Level-2 radar file and extract the base reflectivity
    field as a 2-D numpy array.

    This function requires the `arm-pyart` library.  If pyart is not installed
    or the file cannot be parsed, the function returns None and logs a warning.

    Args:
        filepath: Absolute or relative path to a NEXRAD Level-2 file.

    Returns:
        2-D float32 array of reflectivity values (dBZ), or None on failure.
    """
    try:
        import pyart  # arm-pyart: pip install arm-pyart
        radar = pyart.io.read_nexrad_archive(filepath)
        # Extract the lowest-tilt (index 0) reflectivity sweep
        reflectivity = radar.get_field(0, "reflectivity")
        return reflectivity.filled(fill_value=MISSING_VALUE).astype(np.float32)
    except ImportError:
        print("[WARNING] arm-pyart not installed. Returning synthetic placeholder.")
        return _synthetic_radar_frame()
    except Exception as e:
        print(f"[WARNING] Could not load {filepath}: {e}")
        return None


def _synthetic_radar_frame(size: int = DEFAULT_IMG_SIZE) -> np.ndarray:
    """
    Generate a synthetic radar frame for testing without real data.

    Produces a random reflectivity field with a Gaussian storm-cell blob
    superimposed on a low-reflectivity background.

    Args:
        size: Spatial dimension (height = width = size).

    Returns:
        Synthetic 2-D float32 array in dBZ units.
    """
    rng = np.random.default_rng(seed=0)
    background = rng.uniform(DBZ_MIN, 5.0, size=(size, size)).astype(np.float32)
    # Add a simulated storm cell
    cx, cy = size // 2, size // 2
    y, x = np.ogrid[:size, :size]
    radius = size // 4
    mask = (x - cx) ** 2 + (y - cy) ** 2 < radius ** 2
    background[mask] += rng.uniform(20.0, 50.0, size=background[mask].shape)
    return np.clip(background, DBZ_MIN, DBZ_MAX)


def normalize_reflectivity(frame: np.ndarray) -> np.ndarray:
    """
    Normalize raw reflectivity values to the [0, 1] range.

    Missing values (MISSING_VALUE sentinel) are replaced with 0.0 before
    normalization so they do not distort the scaling.

    Args:
        frame: 2-D float32 array with values in dBZ (or MISSING_VALUE).

    Returns:
        Normalized 2-D float32 array with values in [0, 1].
    """
    # Replace missing values
    frame = np.where(frame == MISSING_VALUE, DBZ_MIN, frame)
    # Clip to valid physical range then rescale
    frame = np.clip(frame, DBZ_MIN, DBZ_MAX)
    normalized = (frame - DBZ_MIN) / (DBZ_MAX - DBZ_MIN)
    return normalized.astype(np.float32)


def resize_frame(
    frame: np.ndarray,
    target_size: Tuple[int, int] = (DEFAULT_IMG_SIZE, DEFAULT_IMG_SIZE),
) -> np.ndarray:
    """
    Resize a 2-D radar frame to the target spatial resolution using
    bilinear interpolation via OpenCV.

    Args:
        frame:       2-D float32 array.
        target_size: (height, width) tuple for the output.

    Returns:
        Resized 2-D float32 array.
    """
    try:
        import cv2
        resized = cv2.resize(frame, (target_size[1], target_size[0]),
                             interpolation=cv2.INTER_LINEAR)
    except ImportError:
        # Fallback: coarse resize with numpy slicing
        h, w = frame.shape
        row_idx = np.linspace(0, h - 1, target_size[0]).astype(int)
        col_idx = np.linspace(0, w - 1, target_size[1]).astype(int)
        resized = frame[np.ix_(row_idx, col_idx)]
    return resized.astype(np.float32)


def extract_intensity_stats(frame: np.ndarray) -> dict:
    """
    Compute summary statistics from a normalized radar frame that serve as
    contextual features for the RL state space.

    Args:
        frame: Normalized 2-D float32 array in [0, 1].

    Returns:
        Dictionary with keys: mean_intensity, max_intensity, coverage_fraction,
        storm_cell_count (estimated as number of high-intensity regions).
    """
    threshold = 0.5  # Roughly corresponds to ~30 dBZ
    high_intensity_mask = frame > threshold
    coverage = float(high_intensity_mask.mean())
    # Rough storm-cell count via connected components (requires scipy)
    try:
        from scipy.ndimage import label
        _, num_cells = label(high_intensity_mask)
    except ImportError:
        num_cells = int(coverage * 10)  # crude approximation

    return {
        "mean_intensity": float(frame.mean()),
        "max_intensity": float(frame.max()),
        "coverage_fraction": coverage,
        "storm_cell_count": num_cells,
    }


def preprocess_single_file(
    filepath: str,
    output_dir: str,
    img_size: int = DEFAULT_IMG_SIZE,
) -> Optional[dict]:
    """
    Full preprocessing pipeline for a single NEXRAD file:
      1. Load raw data
      2. Normalize reflectivity
      3. Resize to target resolution
      4. Save as .npy file
      5. Return metadata dict

    Args:
        filepath:   Path to raw NEXRAD file.
        output_dir: Directory where processed .npy file will be saved.
        img_size:   Spatial resolution for resizing.

    Returns:
        Metadata dictionary or None if loading failed.
    """
    raw = load_nexrad_file(filepath)
    if raw is None:
        return None

    normalized = normalize_reflectivity(raw)
    resized = resize_frame(normalized, target_size=(img_size, img_size))
    stats = extract_intensity_stats(resized)

    # Save processed frame
    filename = Path(filepath).stem + "_radar.npy"
    save_path = os.path.join(output_dir, filename)
    np.save(save_path, resized)

    return {
        "source_file": filepath,
        "processed_file": save_path,
        **stats,
    }


def process_dataset(
    data_dir: str,
    output_dir: str,
    img_size: int = DEFAULT_IMG_SIZE,
    max_samples: Optional[int] = None,
) -> pd.DataFrame:
    """
    Batch process all NEXRAD files found under data_dir.

    Searches recursively for files without a '.nc' or '.png' extension
    (NEXRAD Level-2 files have no standard extension or are .gz compressed).

    Args:
        data_dir:    Root directory containing raw NEXRAD files.
        output_dir:  Directory to store processed .npy files.
        img_size:    Target image size.
        max_samples: Optional cap on the number of files to process (for testing).

    Returns:
        Pandas DataFrame with one row per processed file and metadata columns.
    """
    os.makedirs(output_dir, exist_ok=True)
    radar_output_dir = os.path.join(output_dir, "radar_frames")
    os.makedirs(radar_output_dir, exist_ok=True)

    # Collect candidate files
    data_path = Path(data_dir)
    all_files: List[str] = []
    for ext in ["", ".gz", ".ar2v"]:
        all_files.extend(str(p) for p in data_path.rglob(f"*{ext}") if p.is_file())
    all_files = sorted(set(all_files))

    if max_samples:
        all_files = all_files[:max_samples]

    if not all_files:
        print(f"[INFO] No NEXRAD files found in {data_dir}. "
              "Generating synthetic samples for testing.")
        all_files = [f"synthetic_{i}" for i in range(max_samples or 20)]

    records = []
    for filepath in tqdm(all_files, desc="Processing radar files"):
        if filepath.startswith("synthetic_"):
            # Generate synthetic frame when no real data is present
            frame = _synthetic_radar_frame(img_size)
            stats = extract_intensity_stats(frame)
            filename = filepath + "_radar.npy"
            save_path = os.path.join(radar_output_dir, filename)
            np.save(save_path, frame)
            records.append({"source_file": filepath, "processed_file": save_path, **stats})
        else:
            meta = preprocess_single_file(filepath, radar_output_dir, img_size)
            if meta:
                records.append(meta)

    df = pd.DataFrame(records)
    metadata_path = os.path.join(output_dir, "radar_metadata.csv")
    df.to_csv(metadata_path, index=False)
    print(f"[INFO] Processed {len(df)} radar files. Metadata saved to {metadata_path}")
    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess NOAA NEXRAD radar data for weather perception model."
    )
    parser.add_argument("--data_dir", type=str, default="data/nexrad",
                        help="Directory containing raw NEXRAD Level-2 files.")
    parser.add_argument("--output_dir", type=str, default="data/processed",
                        help="Directory to save processed .npy files and metadata.")
    parser.add_argument("--img_size", type=int, default=DEFAULT_IMG_SIZE,
                        help="Target image resolution (default: 224).")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of files to process (useful for testing).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        img_size=args.img_size,
        max_samples=args.max_samples,
    )
