# RS-Deforestation

## Usage Scenario

The main purpose of the tool is to provide perform the deforestation elaboration over the period of time (for e.g 2 years). This project implements a pipeline for deforestation using Sentinel-2 Level-2A imagery between two years. It processes raw .SAFE or .zip Sentinel-2 inputs, extracts NDVI and BSI indices, interpolates them to a monthly time series, applies BFAST (Breaks For Additive Season and Trend), and outputs change detection and probability maps.

## Input

- **Sentinel-2 L2A Data** in `.SAFE` folders or `.zip` format.
- **Forest Mask** in `.shp` or raster format.
  - Used to limit analysis to forested areas.
  - Can be downloaded from the [WebGIS Portal](https://webgis.provincia.tn.it/) confine del bosco layer or from https://siatservices.provincia.tn.it/idt/vector/p_TN_3d0874bc-7b9e-4c95-b885-0f7c610b08fa.zip.

## Output

GeoTIFF file for:

- **Change map** (e.g., `CD_2018_2019.tif`)

The change map has two bands for each pixel:

1. year of change (2018 or 2019)
2. probability of change ( between 0 to 1)

The output is logged as artifact in the project context.
