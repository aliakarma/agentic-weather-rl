"""
Satellite Image Preprocessing Module
======================================
Purpose:
    Preprocess GOES-16/17 satellite imagery (NetCDF4) and SEVIR HDF5 files
    into normalized numpy arrays for the multi-modal perception model.

Input:
    Raw GOES NetCDF4 files or SEVIR HDF5 files from data/goes/ or data/sevir/

Output:
    Preprocessed satellite frames saved as .npy arrays in data/processed/satellite_imgs/
    A metadata CSV with paths and channel statistics.

Example usage:
    python preprocessing/process_satellite_images.py \
        --data_dir data/goes \
        --output_dir data/processed \
        --source goes \
        --img_size 224
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
DEFAULT_IMG_SIZE: int = 224

# GOES channel normalization bounds (brightness temperature in Kelvin or
# reflectance in [0, 1] depending on band)
GOES_CHANNEL_BOUNDS = {
    "C02": (0.0, 1.0),        # Visible reflectance
    "C09": (180.0, 320.0),    # Upper-level water vapour BT (K)
    "C13": (180.0, 320.0),    # Clean IR window BT (K)
}

# SEVIR channel statistics (approximate, from dataset documentation)
SEVIR_CHANNEL_BOUNDS = {
    "vil":   (0.0, 70.0),     # Vertically Integrated Liquid (dBZ equivalent)
    "ir069": (180.0, 320.0),  # 6.9 µm brightness temperature
    "vis":   (0.0, 1.0),      # Visible reflectance
    "ir107": (180.0, 320.0),  # 10.7 µm brightness temperature
}


# ---------------------------------------------------------------------------
# GOES preprocessing
# ---------------------------------------------------------------------------

def load_goes_netcdf(filepath: str, variable: str = "CMI") -> Optional[np.ndarray]:
    """
    Load a GOES ABI Level-2 NetCDF4 file and extract the specified variable.

    Args:
        filepath: Path to a GOES .nc file.
        variable: NetCDF4 variable name to extract (default: 'CMI').

    Returns:
        2-D float32 array or None on failure.
    """
    try:
        import netCDF4 as nc  # pip install netCDF4
        with nc.Dataset(filepath, "r") as ds:
            data = np.array(ds.variables[variable][:], dtype=np.float32)
        # Remove singleton dimensions if present (e.g., time axis)
        data = np.squeeze(data)
        return data
    except ImportError:
        print("[WARNING] netCDF4 not installed. Returning synthetic satellite frame.")
        return _synthetic_satellite_frame()
    except Exception as e:
        print(f"[WARNING] Could not load {filepath}: {e}")
        return None


def _synthetic_satellite_frame(
    size: int = DEFAULT_IMG_SIZE,
    n_channels: int = 3,
) -> np.ndarray:
    """
    Generate a synthetic multi-channel satellite frame for testing.

    Produces arrays that loosely resemble cloud-coverage patterns using
    Gaussian noise blurred with a large kernel.

    Args:
        size:       Spatial resolution.
        n_channels: Number of spectral channels to simulate.

    Returns:
        float32 array of shape (n_channels, size, size) in [0, 1].
    """
    rng = np.random.default_rng(seed=1)
    frame = rng.uniform(0.0, 1.0, size=(n_channels, size, size)).astype(np.float32)
    # Simulate cloud structure with a simple box blur
    try:
        import cv2
        for c in range(n_channels):
            frame[c] = cv2.GaussianBlur(frame[c], (15, 15), 0)
    except ImportError:
        pass
    return frame


# ---------------------------------------------------------------------------
# SEVIR preprocessing
# ---------------------------------------------------------------------------

def load_sevir_hdf5(
    filepath: str,
    channel: str = "vil",
    time_index: int = 0,
) -> Optional[np.ndarray]:
    """
    Load a single channel and time slice from a SEVIR HDF5 file.

    SEVIR HDF5 structure: dataset name equals the channel name;
    shape is (N_events, T, H, W) where T=49 time steps.

    Args:
        filepath:   Path to SEVIR .h5 file.
        channel:    Channel key (e.g., 'vil', 'ir069', 'vis').
        time_index: Temporal index to extract (0–48).

    Returns:
        3-D float32 array of shape (N_events, H, W) or None on failure.
    """
    try:
        import h5py
        with h5py.File(filepath, "r") as f:
            data = np.array(f[channel][:, time_index, :, :], dtype=np.float32)
        return data
    except ImportError:
        print("[WARNING] h5py not installed. Returning synthetic SEVIR frames.")
        return _synthetic_satellite_frame(DEFAULT_IMG_SIZE)[0:1]
    except Exception as e:
        print(f"[WARNING] Could not load {filepath} channel '{channel}': {e}")
        return None


# ---------------------------------------------------------------------------
# Shared preprocessing utilities
# ---------------------------------------------------------------------------

def normalize_channel(
    frame: np.ndarray,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> np.ndarray:
    """
    Normalize pixel values to [0, 1] using known physical bounds.

    Values outside [vmin, vmax] are clipped before normalization.

    Args:
        frame: 2-D float32 array.
        vmin:  Minimum physical value.
        vmax:  Maximum physical value.

    Returns:
        Normalized 2-D float32 array in [0, 1].
    """
    frame = np.clip(frame, vmin, vmax)
    return ((frame - vmin) / (vmax - vmin)).astype(np.float32)


def resize_image(
    frame: np.ndarray,
    target_size: Tuple[int, int] = (DEFAULT_IMG_SIZE, DEFAULT_IMG_SIZE),
) -> np.ndarray:
    """
    Resize a 2-D frame to the target spatial resolution.

    Args:
        frame:       2-D float32 array.
        target_size: (height, width) target dimensions.

    Returns:
        Resized float32 array.
    """
    try:
        import cv2
        return cv2.resize(frame, (target_size[1], target_size[0]),
                          interpolation=cv2.INTER_LINEAR).astype(np.float32)
    except ImportError:
        h, w = frame.shape
        row_idx = np.linspace(0, h - 1, target_size[0]).astype(int)
        col_idx = np.linspace(0, w - 1, target_size[1]).astype(int)
        return frame[np.ix_(row_idx, col_idx)].astype(np.float32)


def stack_channels(channels: List[np.ndarray]) -> np.ndarray:
    """
    Stack a list of 2-D single-channel arrays into a (C, H, W) tensor.

    Args:
        channels: List of 2-D float32 arrays, all the same shape.

    Returns:
        float32 array of shape (C, H, W).
    """
    return np.stack(channels, axis=0).astype(np.float32)


def extract_cloud_stats(frame: np.ndarray) -> dict:
    """
    Compute cloud-coverage statistics from a normalized satellite frame.

    Args:
        frame: 2-D normalized float32 array in [0, 1] (any channel).

    Returns:
        Dictionary with mean brightness, cloud fraction (pixels > 0.5),
        and peak brightness.
    """
    return {
        "mean_brightness": float(frame.mean()),
        "cloud_fraction": float((frame > 0.5).mean()),
        "peak_brightness": float(frame.max()),
    }


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def process_goes_dataset(
    data_dir: str,
    output_dir: str,
    img_size: int = DEFAULT_IMG_SIZE,
    max_samples: Optional[int] = None,
) -> pd.DataFrame:
    """
    Batch preprocess GOES NetCDF4 files.

    Args:
        data_dir:    Root directory containing .nc files.
        output_dir:  Where to save processed .npy files.
        img_size:    Target spatial resolution.
        max_samples: Optional limit on files processed.

    Returns:
        DataFrame with file paths and channel statistics.
    """
    sat_out = os.path.join(output_dir, "satellite_imgs")
    os.makedirs(sat_out, exist_ok=True)

    nc_files = sorted(Path(data_dir).rglob("*.nc"))
    if max_samples:
        nc_files = nc_files[:max_samples]

    records = []
    for fpath in tqdm(nc_files, desc="Processing GOES files"):
        raw = load_goes_netcdf(str(fpath))
        if raw is None:
            continue
        if raw.ndim == 2:
            raw = raw[np.newaxis, ...]  # add channel dim

        processed_channels = []
        for c_idx in range(raw.shape[0]):
            channel = resize_image(raw[c_idx], (img_size, img_size))
            channel = normalize_channel(channel, 0.0, 1.0)
            processed_channels.append(channel)

        stacked = stack_channels(processed_channels)
        save_name = fpath.stem + "_sat.npy"
        save_path = os.path.join(sat_out, save_name)
        np.save(save_path, stacked)

        stats = extract_cloud_stats(stacked[0])
        records.append({"source_file": str(fpath), "processed_file": save_path, **stats})

    if not records:
        print("[INFO] No GOES files found. Generating synthetic satellite samples.")
        for i in range(max_samples or 20):
            frame = _synthetic_satellite_frame(img_size)
            save_path = os.path.join(sat_out, f"synthetic_{i}_sat.npy")
            np.save(save_path, frame)
            stats = extract_cloud_stats(frame[0])
            records.append({"source_file": f"synthetic_{i}", "processed_file": save_path, **stats})

    df = pd.DataFrame(records)
    meta_path = os.path.join(output_dir, "satellite_metadata.csv")
    df.to_csv(meta_path, index=False)
    print(f"[INFO] Processed {len(df)} satellite images. Metadata saved to {meta_path}")
    return df


def process_sevir_dataset(
    data_dir: str,
    output_dir: str,
    channels: Optional[List[str]] = None,
    img_size: int = DEFAULT_IMG_SIZE,
    max_samples: Optional[int] = None,
) -> pd.DataFrame:
    """
    Batch preprocess SEVIR HDF5 storm-event files.

    Args:
        data_dir:    Directory containing SEVIR .h5 files.
        output_dir:  Where to save processed .npy files.
        channels:    List of SEVIR channel keys to include (default: all).
        img_size:    Target spatial resolution.
        max_samples: Optional limit on storm events extracted.

    Returns:
        DataFrame with file paths and per-channel statistics.
    """
    if channels is None:
        channels = list(SEVIR_CHANNEL_BOUNDS.keys())

    sat_out = os.path.join(output_dir, "satellite_imgs")
    os.makedirs(sat_out, exist_ok=True)

    h5_files = sorted(Path(data_dir).rglob("*.h5"))
    records = []

    for fpath in tqdm(h5_files, desc="Processing SEVIR files"):
        channel_arrays = []
        for ch in channels:
            data = load_sevir_hdf5(str(fpath), channel=ch)
            if data is None:
                break
            n = min(data.shape[0], max_samples or data.shape[0])
            for event_idx in range(n):
                frame = resize_image(data[event_idx], (img_size, img_size))
                vmin, vmax = SEVIR_CHANNEL_BOUNDS.get(ch, (0.0, 1.0))
                frame = normalize_channel(frame, vmin, vmax)
                channel_arrays.append((ch, event_idx, frame))

        for ch, idx, frame in channel_arrays:
            save_name = f"{fpath.stem}_{ch}_{idx:04d}_sat.npy"
            save_path = os.path.join(sat_out, save_name)
            np.save(save_path, frame[np.newaxis, ...])
            stats = extract_cloud_stats(frame)
            records.append({
                "source_file": str(fpath),
                "channel": ch,
                "event_index": idx,
                "processed_file": save_path,
                **stats,
            })

    if not records:
        print("[INFO] No SEVIR files found. Generating synthetic samples.")
        for i in range(max_samples or 20):
            frame = _synthetic_satellite_frame(img_size)
            save_path = os.path.join(sat_out, f"synthetic_sevir_{i}_sat.npy")
            np.save(save_path, frame)
            stats = extract_cloud_stats(frame[0])
            records.append({"source_file": f"synthetic_{i}", "processed_file": save_path, **stats})

    df = pd.DataFrame(records)
    meta_path = os.path.join(output_dir, "sevir_metadata.csv")
    df.to_csv(meta_path, index=False)
    print(f"[INFO] Processed {len(df)} SEVIR frames. Metadata saved to {meta_path}")
    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess GOES/SEVIR satellite imagery for the weather perception model."
    )
    parser.add_argument("--data_dir", type=str, default="data/goes",
                        help="Directory containing raw satellite files.")
    parser.add_argument("--output_dir", type=str, default="data/processed",
                        help="Directory to save processed .npy files and metadata.")
    parser.add_argument("--source", type=str, choices=["goes", "sevir"], default="goes",
                        help="Dataset source type.")
    parser.add_argument("--img_size", type=int, default=DEFAULT_IMG_SIZE,
                        help="Target image resolution.")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to process.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.source == "goes":
        process_goes_dataset(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            img_size=args.img_size,
            max_samples=args.max_samples,
        )
    elif args.source == "sevir":
        process_sevir_dataset(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            img_size=args.img_size,
            max_samples=args.max_samples,
        )
