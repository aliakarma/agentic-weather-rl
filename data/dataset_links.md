# Dataset Download Instructions

This project uses three publicly available meteorological datasets.
**No datasets are bundled in this repository.** Follow the instructions below to obtain them.

---

## 1. SEVIR — Storm Event Imagery Dataset

- **Source:** MIT Lincoln Laboratory / Amazon Open Data Registry
- **URL:** https://registry.opendata.aws/sevir/
- **Direct Download:** https://sevir.s3.amazonaws.com/
- **Size:** ~1 TB (full); ~50 GB (recommended subset for experiments)
- **Format:** HDF5 files containing aligned radar, satellite, and lightning channels

**Download via AWS CLI:**
```bash
# Install AWS CLI if not available
pip install awscli

# Download a subset (e.g., a single day of events)
aws s3 cp s3://sevir/ data/sevir/ --recursive --no-sign-request --exclude "*" --include "2019*"
```

**Description:** SEVIR provides temporally aligned multi-modal weather observations:
- `vis` — GOES-16 visible channel (0.64 µm)
- `ir069` — GOES-16 cloud-top temperature (6.9 µm)
- `ir107` — GOES-16 cloud-top temperature (10.7 µm)
- `vil` — NEXRAD vertically integrated liquid (radar)
- `lght` — GOES-16 lightning flashes

**Expected directory structure:**
```
data/sevir/
├── CATALOG.csv
├── data/
│   ├── vil/
│   ├── ir069/
│   ├── vis/
│   └── lght/
```

---

## 2. GOES — Geostationary Operational Environmental Satellite

- **Source:** NOAA / Amazon Open Data Registry
- **URL:** https://registry.opendata.aws/noaa-goes/
- **Buckets:**
  - GOES-16 (East): `s3://noaa-goes16`
  - GOES-17 (West): `s3://noaa-goes17`
- **Format:** NetCDF4 files

**Download via AWS CLI:**
```bash
# Download ABI Level 2 Cloud products
aws s3 sync s3://noaa-goes16/ABI-L2-CMIPC/ data/goes16/ \
  --no-sign-request \
  --exclude "*" \
  --include "2022/180/*"
```

**Key products used:**
| Product | Description |
|---|---|
| ABI-L2-CMIPC | Cloud and Moisture Imagery |
| ABI-L2-ACHTF | Cloud Top Height |
| ABI-L2-ACTPC | Cloud Top Phase |

---

## 3. NOAA NEXRAD — Next Generation Weather Radar

- **Source:** NOAA / Amazon Open Data Registry
- **URL:** https://registry.opendata.aws/noaa-nexrad/
- **Bucket:** `s3://noaa-nexrad-level2`
- **Format:** Level-2 binary radar files (readable with `pyart`)

**Download via AWS CLI:**
```bash
# Download NEXRAD data for a specific station and date
aws s3 sync s3://noaa-nexrad-level2/2022/06/15/KTLX/ data/nexrad/KTLX/ --no-sign-request
```

**Available stations:** ~160 CONUS WSR-88D radar sites  
**Key variables:** Reflectivity (DBZ), Radial Velocity (VEL), Spectrum Width

**Install pyart for reading NEXRAD files:**
```bash
pip install arm-pyart
```

---

## Recommended Minimal Dataset for Experiments

To run experiments without downloading the full datasets, prepare a minimal subset:

1. Download ~500 SEVIR storm events (HDF5)
2. Download 2–4 weeks of GOES ABI imagery for a target region
3. Download NEXRAD Level-2 files for 2–3 nearby stations over the same period

Expected total size for a minimal run: **~20–50 GB**

---

## Data Placement

After downloading, place processed data at:

```
data/
├── sevir/              # Raw SEVIR HDF5 files
├── goes/               # Raw GOES NetCDF4 files
├── nexrad/             # Raw NEXRAD Level-2 files
└── processed/          # Output of preprocessing scripts
    ├── radar_frames/   # Preprocessed radar arrays (.npy)
    ├── satellite_imgs/ # Preprocessed satellite arrays (.npy)
    └── metadata.csv    # Labels and contextual variables
```

Run preprocessing:
```bash
python preprocessing/process_radar_data.py --data_dir data/nexrad --output_dir data/processed
python preprocessing/process_satellite_images.py --data_dir data/goes --output_dir data/processed
```
