# sen-et-openeo
Run SenET workflow through OpenEO.

SenET was originally developed within the Sen4ET ESA project (https://www.esa-sen4et.org/).

The main output of this workflow is a thermal water stress indicator **LST-Ta** (Land Surface Temperature minus air temperature), as well as an evapotranspiration estimate computed with the **TSEB-PT** (Two-Source Energy Balance – Priestley-Taylor) model.

The script now runs a complete multi-step processing chain: OpenEO/CDSE download, preprocessing and harmonisation, Sentinel-3 LST sharpening, optional ECOSTRESS-based correction, LST-Ta and NDVI export, TSEB-PT ET modelling, and optional generation of LSTM-like 30 m LST, LST-Ta and ET products. For longer periods, downloads can be chunked into smaller OpenEO jobs, and each processing step is cached so interrupted runs can resume automatically.

In addition to the LST sharpening as developed within Sen4ET, additional bias and directionality corrections based on intercomparison of sharpened Sentinel-3 LST data with ECOSTRESS LST data have been added.
All scripts required to compute these correction coefficients can be found in the `scripts/corrections_ecostress/` folder.

At the end of the main script, all required files are prepared to upload the final results to the Food Security TEP platform.
Actual data upload is completely optional and can be done using the scripts located under `scripts/FSTEP_upload` script.

---

## Environment setup

A fully reproducible conda environment is provided. Requires **Python 3.11** and conda (or [mamba](https://mamba.readthedocs.io) for faster solving).

```bash
# Create the environment from the provided yml file
conda env create -f OpenEO_senet_20260312.yml

# Activate it
conda activate OpenEO_senet

# Install the package itself in editable mode
pip install -e .
```

> **Note:** The yml file was exported on 2026-03-12 and contains all pinned package versions.
> If you encounter solver conflicts, try replacing `conda` with `mamba` in the command above.

---

## Required user accounts
Before being able to execute the main script `scripts/run_sen-et.py`, the following user accounts are needed:

- **Copernicus Data Space Ecosystem (CDSE)** — register here:
  https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/auth?client_id=cdse-public&response_type=code&scope=openid&redirect_uri=https%3A//dataspace.copernicus.eu/account/confirmed/1

- **Copernicus Climate Data Store (CDS / ERA5)**:
  - Create an ECMWF account (login/register button top-right): https://cds-beta.climate.copernicus.eu/
  - Save your personal API token following the instructions here: https://cds-beta.climate.copernicus.eu/how-to-api
  - Accept the ERA5 dataset licence (scroll down, click "Accept licence"): https://cds-beta.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download

---

## Processing pipeline

The full workflow is executed by `scripts/run_sen-et.py` and consists of the following steps, implemented in `src/sen_et_openeo/`:

```
Step 0 — Download          (data_download.py  →  001_download/)
Step 1 — Preprocess        (data_download.py  →  002_preprocess/)
Step 2 — LST Sharpening    (data_download.py  →  003_sharpening/)
Step 3 — LST Correction    (data_download.py  →  004_lst-correction/)   [optional]
Step 4 — LST-Ta            (run_sen-et.py → 005_lst-ta/)
Step 5 — NDVI export       (run_sen-et.py → 006_ndvi/)
Step 6 — ET (TSEB-PT)      (tseb.py           →  007_et/)
Step 7 — LSTM-like LST + LST-Ta + ET (run_sen-et.py → 008_lstm-like/ + 008_lstm-ta/ + 008_lstm-et/) [optional]
```

Each step saves its results as a `.pkl` file so that a re-run automatically resumes from the last completed step.

---

### Step 0 — Download (`001_download/`)

Data is downloaded from **CDSE via OpenEO**. All jobs can be launched in parallel.
Results are stored in `001_download/` and catalogued in `download_<start>_<end>.pkl` (see [Temporal chunking](#temporal-chunking)).

| Dataset | Collection | Format | Notes |
|---|---|---|---|
| Sentinel-2 L2A | `SENTINEL2_L2A` | GeoTIFF (multi-band) | Bands B02–B12 + NDVI; optional SCL dilation masking, dekadal compositing, linear temporal interpolation |
| Sentinel-3 SLSTR LST | `SENTINEL3_SLSTR_L2_LST` | NetCDF | LST, LST_uncertainty, confidence_in, exception, sun/view angles; optional daytime/cloud filtering UDFs. File is named `datacube_s3_<start>_<end>.nc` so different time chunks do not overwrite each other. |
| Copernicus DEM | `COPERNICUS_30` | GeoTIFF | Single temporal max composite; shared across all time chunks |
| ESA WorldCover | `ESA_WORLDCOVER_10M_2020_V1` or `_2021_V2` | GeoTIFF | Version chosen automatically based on start year (< 2021 → 2020 product); shared across all time chunks |
| Biophysical vars | ESA-APEx `biopar` UDP | GeoTIFF | LAI, FAPAR, FCOVER — one OpenEO job per variable; optional dekadal compositing, linear temporal interpolation (same flags as S2/NDVI). **Skipped entirely when `compute_et_tseb = False`** to save time and compute credits. |

After downloading, scale/offset metadata is written into the Sentinel-2 GeoTIFFs.

When re-scanning an existing `001_download/` directory (no pkl present), the S2 and BIOPAR GeoTIFF file lists are filtered to only include dates that fall within the current `temporal_extent`, so files from a previous time chunk are never accidentally reused.

#### Chunked download via the Job Manager

For longer time periods or to avoid single large jobs timing out on CDSE, Sentinel-2 and BIOPAR downloads can be split into smaller temporal chunks and managed via the **OpenEO `MultiBackendJobManager`** (requires `openeo-python-client >= 0.31.0`).

This is controlled by two parameters in `download()`:

| Parameter | Default | Description |
|---|---|---|
| `s2_chunk_months` | `0` | When > 0, the S2 download is split into chunks of this many months, each submitted as a separate OpenEO batch job. `0` = single job (legacy behaviour). |
| `biopar_chunk_months` | `0` | Same for BIOPAR (LAI, FAPAR, FCOVER). When > 0, one job per *(variable × chunk)* is submitted. `0` = single job per variable. |
| `max_concurrent_jobs` | `2` | Maximum number of jobs running simultaneously on the CDSE backend. |

**How it works:**

1. The total temporal extent is split into chunks of `chunk_months` months.
2. One OpenEO batch job is created per chunk (and per BIOPAR variable). All jobs are tracked via a persistent **CSV job database** stored alongside the pkl files:
   - `s2_jobs_<start>_<end>.csv` — S2 job tracker
   - `biopar_jobs_<start>_<end>.csv` — BIOPAR job tracker
3. The `MultiBackendJobManager` polls CDSE until all jobs are `finished` (or `error`/`cancelled`). If the script is interrupted, **re-running it resumes from where it left off** — already-finished jobs are not re-submitted.
4. Downloaded GeoTIFFs are moved from the job manager's working directory into the standard `S2/` and `BIOPAR/<VAR>/` directories, with filenames normalised to `SENTINEL2_L2A_<date>Z.tif` and `BIOPAR_<VAR>_<date>Z.tif` respectively.
5. Failed or cancelled jobs are logged as warnings but do not abort the run — the remaining files are still collected and the workflow continues.

**Intermediate files** are written to:
```
001_download/
├── S2/_job_manager/            ← per-job download directories (S2)
└── BIOPAR/_job_manager/        ← per-job download directories (BIOPAR)
```

These can be deleted once the download is confirmed complete.

> **Note:** If output files recorded in the pkl are deleted from disk (e.g. to force a re-download), the cached entries are automatically invalidated and the download is re-triggered on the next run.

---

### Step 1 — Preprocess (`002_preprocess/`)

Raw downloaded files are converted, spatially aligned and quality-flagged.
Results are stored in `002_preprocess/` and catalogued in `preprocess_<start>_<end>.pkl`.

#### Sub-steps

**1.1 — Format conversion**

| Input | Operation |
|---|---|
| **Sentinel-3** NetCDF | Converted band-by-band to GeoTIFF per timestamp. The `confidence_in` band is additionally unpacked into 16 individual bit-layer GeoTIFFs (`confidence_in_bitlayers`). Longitude/latitude grids are also written as separate GeoTIFFs (`lon`, `lat`). |
| **Sentinel-2** multi-band GeoTIFF | Split into individual per-band, per-date GeoTIFFs (B02 … B12, NDVI). Scale/offset values are preserved per band. |
| **Copernicus DEM** GeoTIFF | Slope (radians) and aspect (radians) are derived with RichDEM. Three files are written: `DEM_alt.tif`, `DEM_slo.tif`, `DEM_asp.tif`. |

**1.2 — Spatial warping (GDAL)**

All layers are warped to a common grid using the Sentinel-2 B02 file as spatial reference (20 m UTM grid).

| Input | Target resolution | Resampling |
|---|---|---|
| **Sentinel-3** angle bands (`sunAzimuthAngles`, `sunZenithAngles`, `viewAzimuthAngles`, `viewZenithAngles`, `lat`, `lon`) | High-resolution S2 grid | Nearest neighbour |
| **Sentinel-3** all other bands (LST, uncertainty, flags, …) | ~1098 m (native S3 pixel size) | Nearest neighbour |
| **Copernicus DEM** (alt, slo, asp) | S2 20 m grid | Nearest neighbour |
| **BIOPAR** (LAI, FAPAR, FCOVER) | S2 20 m grid | Nearest neighbour | Already in S2 UTM CRS (computed from S2 by the `biopar` UDP); copied to `002_preprocess/BIOPAR/` then pixel-aligned to the S2 reference extent |
| **WorldCover** | S2 20 m grid | Mode resampling |

**1.3 — Quality flag generation**

A `quality_flag` raster is derived per S3 timestamp by combining two conditions (both must be satisfied for a pixel to be marked as good quality = 1):
- No cloud detected (`confidence_in` bit 14 = 0)
- LST uncertainty ≤ 1 K

**1.4 — Incidence angle calculation**

A cosine-of-incidence-angle (`inc`) raster is computed per S3 timestamp using sun position (day-of-year, fractional UTC hour), terrain slope, aspect and the high-resolution lat/lon grids.

**1.5 — Optional cleanup**

If `delete_unrequired_data=True`, intermediate bands not needed downstream are deleted to save disk space:
- S3: only `LST`, `inc`, `quality_flag`, `viewZenithAnglesHR`, `latHR`, `lonHR` are kept.
- S2: only `B02–B12` (sharpening inputs) and `NDVI` are kept.
- DEM: only `alt` is kept.

**1.6 — Scaling / dtype conversion**

All retained rasters are converted to compact integer/uint16 formats with a fixed scale and offset, as defined in `utils/geoloader.py` (`OUTPUT_SCALING`):

| Layer | Output dtype | Scale |
|---|---|---|
| S3 LST, LST_uncertainty | uint16 | 0.01 K |
| S3 angles | int16 | 0.01 ° |
| DEM altitude | int16 | 0.01 m |
| DEM slope | uint16 | 0.0001 |
| DEM aspect | int16 | 0.0001 rad |

---

### Step 2 — LST Sharpening (`003_sharpening/`)

Sentinel-3 LST (≈1 km) is downscaled to 20 m using a **Decision Tree Sharpener** (pyDMS — `utils/pyDMS.py`) trained on Sentinel-2 reflectance bands, DEM elevation and the S3 incidence angle.

For each S3 acquisition:
1. The closest available Sentinel-2 observation is selected.
2. A virtual raster (VRT) is built stacking all high-resolution predictor bands: S2 B02, B03, B04, B05, B06, B07, B08, B11, B12 + DEM altitude + S3 incidence angle.
3. A bagging ensemble of decision trees (30 estimators, 80% samples/features) is trained on the low-resolution S3 LST pixels that pass the quality flag.
4. The trained model is applied at 20 m resolution to produce the sharpened LST.
5. Optionally, a residual correction (block-level bias removal) is applied.
6. Optionally, the sharpened LST is masked to the original S3 coverage (see below).

Results are stored in `003_sharpening/` and catalogued in `sharpening_<start>_<end>_<params>.pkl`.

#### Output filename encoding

The sharpened LST filename encodes the two quality-control parameters so that runs with different settings produce distinct, non-conflicting files:

```
LST_SHARPENED_<timestamp>_minf<NNN>[_msk]_fin.tif
```

- `minf<NNN>` — minimum valid S3 fraction as a 3-digit integer percentage, e.g. `minf005` for `min_valid_s3_fraction = 0.05`.
- `_msk` — appended when `mask_to_s3_coverage = True`.

Examples:
- `LST_SHARPENED_20240515T103000_minf005_msk_fin.tif` — 5% threshold, S3 mask applied
- `LST_SHARPENED_20240515T103000_minf000_fin.tif` — no threshold, no mask (default behaviour)

#### Sharpening quality controls

Two user-configurable parameters in `run_sen-et.py` control sharpening behaviour for low-quality or partially-clouded S3 scenes:

**`min_valid_s3_fraction`** *(default: `0.0`)*

Minimum fraction of valid (cloud-free) S3 LST pixels required before sharpening is attempted.
Scenes below this threshold are skipped entirely — no sharpened output is produced for that timestamp.

> **Rationale:** The Decision Tree Sharpener is trained only on valid S3 pixels. If very few pixels are available (e.g. < 5% of the tile), the regression model is trained on a non-representative sample and will extrapolate LST values across the entire S2 tile, producing physically meaningless results. Setting `min_valid_s3_fraction = 0.05` (5%) is recommended to avoid this.

```python
# Skip scenes where fewer than 5% of S3 pixels are cloud-free
min_valid_s3_fraction = 0.05
```

**`mask_to_s3_coverage`** *(default: `False`)*

When `True`, the sharpened LST output is masked so that only pixels **co-located with a valid original S3 observation** are retained. Pixels where the S3 was cloud-masked, outside the swath, or otherwise invalid are set to nodata, regardless of whether the sharpener produced a prediction there.

> **Rationale:** Even when `min_valid_s3_fraction` allows a scene to be processed, the sharpener applies the trained regression to the full S2 tile. This means sharpened values are generated by extrapolation in areas where S3 had no valid observation (e.g. cloud gaps). Setting `mask_to_s3_coverage = True` restricts the output to only the area physically observed by S3, eliminating extrapolated values.

The mask is derived by reprojecting the binary S3 valid-pixel mask (nearest-neighbour) from the native S3 grid (~1098 m) to the 20 m output grid. Each valid S3 pixel maps to a ~55 × 55 block of 20 m pixels.

```python
# Restrict sharpened LST to only where S3 had valid observations
mask_to_s3_coverage = True
```

> **Note:** When re-running with `mask_to_s3_coverage = True` on a dataset where `_fin.tif` files already exist from a previous run, the mask is automatically (re-)applied to those existing files — no need to delete and reprocess from scratch.

---

### Step 3 — LST Correction (`004_lst-correction/`) *(optional)*

If correction parameters are provided (`corr_parameters` dict), a bias and directionality correction is applied to the sharpened LST, correcting for systematic offsets and view-angle-dependent effects derived from ECOSTRESS intercomparison.
If no parameters are provided this step is skipped and the sharpening output is used directly.

Results are catalogued in `lst_correction_<start>_<end>.pkl`.

---

### Step 4 — LST-Ta computation (`005_lst-ta/`)

Air temperature (T_a) is obtained from **ERA5 reanalysis** (`t2m` band), spatially interpolated to the tile grid and corrected for elevation using a moist adiabatic lapse rate.
LST-Ta (K) is then computed pixel-by-pixel as:

```
LST-Ta = LST_sharpened - T_air_ERA5
```

Output GeoTIFFs are written to `005_lst-ta/` as scaled `int16` (scale = 0.01 K).
A CSV file ready for Food Security TEP upload is also generated.

---

### Step 5 — NDVI export (`006_ndvi/`)

NDVI composites from the preprocessing step are copied to `006_ndvi/` with a standardised naming convention, and a CSV file for TEP upload is generated.

---

### Step 6 — Evapotranspiration with TSEB-PT (`007_et/`)

The **Two-Source Energy Balance – Priestley-Taylor** model (pyTSEB) is run for each S3 timestamp.
See the dedicated [TSEB-PT model section](#tseb-pt-model-inputs-and-outputs) below for full details of inputs and outputs.

Results are written as VRT files to `007_et/`.

---

### Step 7 — LSTM-like LST + LST-Ta + ET at 30 m (`008_lstm-like/` + `008_lstm-ta/` + `008_lstm-et/`) *(optional)*

Produces a 30 m LST product that mimics the spatial characteristics expected from a future **LSTM** (Land Surface Temperature and Microwave) sensor, and derives the corresponding TSEB-PT evapotranspiration at 30 m. Enabled by setting `generate_lstm_like = True` in the `__main__` block of `run_sen-et.py`.

**7a — LSTM-like LST (`008_lstm-like/`)**

Processing steps applied per timestamp:

1. Read sharpened (and optionally corrected) LST at 20 m; convert scaled integer → Kelvin.
2. Convert to **radiance** using the Stefan-Boltzmann law: $R = \sigma T^4$.
3. **Resample radiance** from 20 m → 30 m using cubic spline interpolation (smooth, preserves gradients).
4. **Resample VZA** (`viewZenithAnglesHR`) from 20 m → 30 m using nearest neighbour.
5. **Mask** pixels where VZA > 30° (off-nadir views introduce angular bias).
6. Convert resampled, masked radiance back to **LST** (K): $T = (R/\sigma)^{1/4}$.

Output: float32 GeoTIFF, LST in Kelvin, nodata = -9999, stored in `008_lstm-like/LST-LSTM_{timestamp}_{tile}.tif`.

> **Note:** Resampling is done through radiance space rather than directly on LST values to avoid physically incorrect averaging of temperature.

**7b — LSTM-like LST-Ta (`008_lstm-ta/`)**

ERA5 `t2m` air temperature is interpolated to the S3 overpass time and subtracted from the 30 m LSTM-like LST pixel-by-pixel, using the same lapse-rate height correction as Step 4. Output is a scaled `int16` GeoTIFF (scale = 0.01 K) stored in `008_lstm-ta/LST-Ta-LSTM_{timestamp}_{tile}.tif`.

**7c — LSTM-like ET (`008_lstm-et/`)**

The TSEB-PT model is run again using the 30 m LSTM-like LST as the radiometric surface temperature input (`T_R1`). All other inputs (ERA5 meteorology, BIOPAR, WorldCover, VZA, DEM) are identical to Step 6. Meteorological fields computed during Step 6 for the same timestamp are reused from an in-memory cache; if Step 6 was skipped for a given timestamp (output already existed), the meteo is recomputed on the fly.

Output: TSEB-PT VRT files stored in `008_lstm-et/TSEB-PT_{timestamp}_{tile}.vrt` (same band structure as `007_et/`).

---

## TSEB-PT model — inputs and outputs

The TSEB-PT (Two-Source Energy Balance – Priestley-Taylor) model is implemented in `src/sen_et_openeo/tseb.py` and orchestrated by `compute_et()`.
It is called once per S3 overpass timestamp (Step 6) and optionally again for the 30 m LSTM-like LST (Step 7c).

### Inputs

#### Radiometric / geometric

| Parameter | Symbol | Source | Units |
|---|---|---|---|
| Sharpened (and optionally corrected) LST | `T_R1` | `003_sharpening/` or `004_lst-correction/`; converted to float32 Kelvin before passing to pyTSEB | K |
| View zenith angle | `VZA` | `002_preprocess/S3/viewZenithAnglesHR` | ° |
| Latitude grid | `lat` | `002_preprocess/S3/latHR` | ° |
| Longitude grid | `lon` | `002_preprocess/S3/lonHR` | ° |
| DEM elevation | `alt` | `002_preprocess/DEM/alt` | m |
| Day of year | `DOY` | Derived from S3 overpass datetime | — |
| Decimal UTC hour | `time` | Derived from S3 overpass datetime | h |

Solar zenith (`SZA`) and azimuth (`SAA`) angles are computed internally at runtime from `lat`, `lon`, `DOY` and `time`.

#### Vegetation structure

| Parameter | Symbol | Source | Units |
|---|---|---|---|
| Leaf Area Index | `LAI` | `001_download/BIOPAR/LAI/` (ESA-APEx `biopar` UDP) | m²/m² |
| Fractional vegetation cover | `f_c` | `001_download/BIOPAR/FCOVER/` | — |

> **LAI capping:** The `biopar` UDP occasionally produces outlier LAI values up to ~14 m²/m². Values above **8 m²/m²** are clipped before use in `calc_fg()`, `calc_canopy_height()` and the block-wise TSEB loop. Without this cap, FAPAR → 1 which collapses `f_g` to FAPAR (losing the iterative correction), and very high LAI can also cause pyTSEB to become numerically unstable.
| Fraction of green vegetation | `f_g` | Computed by `calc_fg()` from FAPAR, LAI and SZA at solar noon; cached in `007_et/biopar/` | — |
| Canopy height | `h_C` | Computed by `calc_canopy_height()` from LAI, F_G and ESA WorldCover LUT; cached in `007_et/biopar/` | m |
| Land-cover class | `landcover` | ESA WorldCover remapped to pyTSEB internal codes (`remap_worldcover_for_pytseb()`); cached in `007_et/biopar/` | — |

> **Biopar date selection:** All biopar variables (LAI, FAPAR, FCOVER) share a single acquisition date per TSEB run. The available LAI dates are compared against the S3 overpass datetime and the date with the **smallest absolute time difference** is selected. The same closest date is then used for FAPAR and FCOVER. For the LSTM-like ET run (Step 7c), the same S3 overpass datetime is used, so biopar date selection is identical to Step 6.

#### Meteorology (from ERA5, interpolated to overpass time, height-corrected to 100 m blending height)

| Parameter | Symbol | ERA5 band(s) | Units |
|---|---|---|---|
| Air temperature at 100 m | `T_A1` | `t2m` + geopotential `z`; lapse-rate corrected to DEM + 100 m | K |
| Wind speed at 100 m | `u` | `u100` + `v100` (scalar magnitude) | m/s |
| Vapour pressure | `ea` | `d2m` (dewpoint → Magnus formula) | mb |
| Air pressure | `p` | `sp` | mb |
| Instantaneous shortwave downwelling radiation | `S_dn` | `ssrd` (1-hour accumulation ending at overpass) | W/m² |
| Daily mean shortwave downwelling radiation | `S_dn_24` | `ssrd` (24-hour daily integral) | W/m² |
| Longwave downwelling radiation | `L_dn` | Estimated internally from `ea`, `T_A1` and `p` via pyTSEB's `calc_longwave_irradiance()` | W/m² |

Height of the meteorological reference level (`z_T`, `z_u`) is fixed at **100 m** (ERA5 wind at 100 m).

#### Fixed model parameters

| Parameter | Value | Description |
|---|---|---|
| `alpha_PT` | 1.26 | Priestley-Taylor coefficient |
| `x_LAD` | 1.0 | Spherical leaf angle distribution |
| `emis_C` / `emis_S` | 0.99 / 0.97 | Canopy and soil emissivities |
| `rho_vis_C` / `tau_vis_C` | 0.07 / 0.08 | Canopy visible reflectance / transmittance |
| `rho_nir_C` / `tau_nir_C` | 0.32 / 0.33 | Canopy NIR reflectance / transmittance |
| `rho_vis_S` / `rho_nir_S` | 0.15 / 0.25 | Soil visible and NIR reflectance |
| `G_form` | `G = 0.35 × Rn` | Soil heat flux as a constant fraction of net radiation |
| `resistance_form` | 0 | Norman et al. (1995) resistance scheme |
| `KN_b`, `KN_c`, `KN_C_dash` | 0.012, 0.0038, 90 | Kustas-Norman resistance coefficients |

---

### Outputs

All output fields are written as individual compressed, tiled GeoTIFFs (float32, nodata = −9999) in `<output_vrt_stem>.data/` and assembled into a single **VRT** file:

```
007_et/
└── TSEB-PT_<timestamp>_<tile>.vrt
    └── TSEB-PT_<timestamp>_<tile>.data/
        ├── flag.tif
        ├── T_S1.tif
        ├── T_C1.tif
        ├── ...
        └── ET_day.tif
```

#### Primary output bands (`S_P`)

| Band | Description | Units |
|---|---|---|
| `flag` | Convergence flag: 0 = converged, 255 = failed (pyTSEB `F_INVALID`) | — |
| `T_S1` | Soil surface temperature | K |
| `T_C1` | Canopy temperature | K |
| `T_AC1` | Air temperature within the canopy layer | K |
| `H_C1` | Sensible heat flux from the canopy | W/m² |
| `H_S1` | Sensible heat flux from the soil | W/m² |
| `LE_C1` | Latent heat flux from the canopy | W/m² |
| `LE_S1` | Latent heat flux from the soil | W/m² |
| `LE_partition` | Latent heat partitioning coefficient | — |
| `H1` | Total sensible heat flux | W/m² |
| `LE1` | Total latent heat flux | W/m² |
| `R_ns1` | Net shortwave radiation | W/m² |
| `R_nl1` | Net longwave radiation | W/m² |
| `Rn1` | Total net radiation | W/m² |
| `G1` | Soil heat flux | W/m² |
| `ET_day` | Daily evapotranspiration (requires `S_dn_24`) | mm/day |

#### Ancillary output bands (`S_A`)

| Band | Description | Units |
|---|---|---|
| `z_0M` | Roughness length for momentum | m |
| `d_0` | Zero-plane displacement height | m |
| `R_A1` | Aerodynamic resistance | s/m |
| `R_x1` | Canopy boundary-layer resistance | s/m |
| `R_S1` | Soil resistance | s/m |
| `u_friction` | Friction velocity | m/s |
| `L` | Monin-Obukhov stability length | m |
| `n_iterations` | Number of iterations to convergence | — |

> **Note:** Pixels where the iteration diverged (`flag = 255`) are set to nodata (−9999) in all energy-flux and temperature bands; only the `flag` band retains the `255` value.

---

## Temporal chunking

To avoid memory and processing issues, it is recommended to limit each run to a maximum of **6 months**. Each time chunk is fully independent: all `.pkl` cache files include the temporal extent in their filename (e.g. `download_20240501_20240630.pkl`), so switching `temporal_extent` in the `__main__` block never triggers an extent-mismatch error and never accidentally reuses cached results from a different period.

**Running multiple chunks sequentially** (recommended):
```python
# Chunk 1
temporal_extent = ['2024-05-01', '2024-06-30']
# ... run script ...

# Chunk 2  — just update the date range and re-run
temporal_extent = ['2024-07-01', '2024-09-30']
```

Each chunk will:
- Create its own pkl files (`download_20240701_20240930.pkl`, etc.).
- Download its own S3 NetCDF (`datacube_s3_20240701_20240930.nc`).
- Reuse DEM and WorldCover files (time-independent).
- Skip already-processed per-date output files (`.tif`, `.vrt`) from any previous chunk.

> **Parallel execution across tiles is safe** (each tile writes to its own subdirectory).  
> **Parallel execution of two chunks on the same tile is risky** — there is no file locking, and shared output subdirectories (`005_lst-ta/`, `007_et/`, etc.) could be written to simultaneously.

---

## Output directory structure

```
<output_dir>/<tile>/
├── download_<start>_<end>.pkl
├── preprocess_<start>_<end>.pkl
├── sharpening_<start>_<end>_<params>.pkl
├── lst_correction_<start>_<end>.pkl  ← only if correction applied
├── biopar_<start>_<end>.pkl          ← only if compute_et_tseb=True
├── 001_download/
│   ├── S2/                           ← S2 multi-band GeoTIFFs
│   ├── S3/                           ← S3 NetCDF file per time chunk
│   ├── DEM/                          ← DEM GeoTIFF (shared across chunks)
│   ├── WorldCover/                   ← WorldCover GeoTIFF (shared)
│   └── BIOPAR/{LAI,FAPAR,FCOVER}/    ← only when compute_et_tseb=True
├── 002_preprocess/
│   ├── S2/                           ← per-band, per-date GeoTIFFs
│   ├── S3/                           ← per-variable, per-date GeoTIFFs
│   └── DEM/                          ← alt, slo, asp GeoTIFFs
├── 003_sharpening/                   ← sharpened LST at 20 m
├── 004_lst-correction/               ← corrected LST (if applicable)
├── 005_lst-ta/                       ← LST-Ta GeoTIFFs + CSV
├── 006_ndvi/                         ← NDVI GeoTIFFs + CSV
├── 007_et/                           ← TSEB-PT evapotranspiration VRTs
├── 008_lstm-like/                    ← LSTM-like 30 m LST GeoTIFFs (if enabled)
├── 008_lstm-ta/                      ← LST-Ta from LSTM-like LST (if enabled)
└── 008_lstm-et/                      ← TSEB-PT ET from LSTM-like LST (if enabled)
```