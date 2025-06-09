# RS-Deforestation

This project implements a pipeline for deforestation using Sentinel-2 Level-2A imagery. It processes raw .SAFE or .zip Sentinel-2 inputs, extracts NDVI and BSI indices, interpolates them to a monthly time series, applies BFAST (Breaks For Additive Season and Trend), and outputs change detection and probability maps.

#### AIxPA

- `kind`: product-template
- `ai`: Remote Sensing
- `domain`: PA

The context in which this project was developed: The project pipeline downloads the indices of area of interest (Trentino) from the sentinel-2 download tool. The Trentino region covers several S2 tiles: T32TQS, T32TPR, T32TPS, T32TQR. These tiles can also be overlapped. The software process each downloaded tile separately, clip them using python procedure to convert the downloaded data to input files and then process the clipped tiles for the deforestation.

The product contains operations for

- Download and preprocess the forest data
- Perform deforestation elaboration
- Log the output tiff image to datalake.

## Prequisites Note!

The pipelines takes around 5 hours to complete with 16 CPUs and 64GB Ram for 2 years of data which is the default period. It consists of interpolation and post processing steps which are computationally heavy since it is pixel based analysis. It is based on python joblib library for optimizations of numpy arrays. With the use of more images the interpolation will be shorter. The amount of sentinal data is huge that is whay a safe limit volume of 250Gi is specified to ensure significant data space.

## Usage

Tool usage documentation [here](./docs/usage.md).

## How To

- [Download and preprocess sentinel forest data](./docs/howto/download.md)
- [Run Deforesation Elaboration and log output ](./docs/howto/elaborate.md)

## License

[Apache License 2.0](./LICENSE)
