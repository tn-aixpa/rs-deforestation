# RS-Deforestation

This project implements a pipeline for deforestation using Sentinel-2 Level-2A imagery. It processes raw .SAFE or .zip Sentinel-2 inputs, extracts NDVI and BSI indices, interpolates them to a monthly time series, applies BFAST (Breaks For Additive Season and Trend), and outputs change detection and probability maps.

#### AIxPA

- `kind`: product-template
- `ai`: remote sensing
- `domain`: PA

The context in which this project was developed: The project pipeline downloads the indices of area of interest (Trentino) from the sentinel-2 download tool. The Trentino region covers several S2 tiles: T32TQS, T32TPR, T32TPS, T32TQR. These tiles can also be overlapped. The software process each downloaded tile separately, clip them using python procedure to convert the downloaded data to input files and then process the clipped tiles for the deforestation.

The product contains operations for

- Download Sentinel-2 data using tile-specific metadata (containing only two years).
- Perform elaboration
  - Compute NDVI and BSI indices from RED, NIR, and SWIR1 bands.
  - Apply cloud/shadow masks from precomputed binary mask files (MASK.npy).
  - Interpolate data to generate a complete 24-month time series (12 months/year).
  - Fuse features and reshape data into pixel-wise time series.
  - Run BFAST to detect change points across time.
  - Post-process change maps to remove isolated pixels and fill gaps.
- Log results as GeoTIFF raster files.

## Requirements

### Hardware Requirements

The pipelines takes around 8 hours to complete with 16 CPUs and 64GB Ram for 2 years of data which is the default period. It consists of two steps (download, elaboration). The download step is dependant on Sentinel Hub dataspace. It could happen that data download takes more time than usual due to various factors, including technical issues, data processing delays, and limitations in the data access infrastructure. The second step 'elaboration' consists of interpolation and post processing steps which are computationally heavy since it is pixel based analysis. It is based on python joblib library for optimizations of numpy arrays. With the use of more images the interpolation will be shorter. The amount of sentinal data is huge that is whay a volume of 250Gi of type 'persistent_volume_claim' is specified to ensure significant data space.

### General Requirements

- Register to the open data space copenicus(if not already) and get your credentials.

```
https://identity.dataspace.copernicus.eu/auth/realms/CDSE/login-actions/registration?client_id=cdse-public&tab_id=FIiRPJeoiX4
```

- Shape file can be downloaded from the [WebGIS Portal](https://webgis.provincia.tn.it/) confine del bosco layer or from https://siatservices.provincia.tn.it/idt/vector/p_TN_3d0874bc-7b9e-4c95-b885-0f7c610b08fa.zip. More details in download [step](./docs/howto/download.md)

## Usage

Tool usage documentation [here](./docs/usage.md).

## How To

- [Download and preprocess sentinel forest data](./docs/howto/download.md)
- [Run Deforesation Elaboration and log output ](./docs/howto/elaborate.md)
- [Workflow](./docs/howto/workflow.md)

## License

[Apache License 2.0](./LICENSE)
