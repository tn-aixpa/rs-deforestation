import sys
import numpy as np
import os, time, shutil, json, zipfile
from os.path import abspath
import utils.filemanager as fm
from utils.S2L2A import L2Atile, getTileList
from utils.utils import _ndi, _bsi
from datetime import datetime
from joblib import Parallel, delayed
from utils.utils import run_bfast_parallel, fuse_features, interpolate_time_series
import utils.post_processing as pp
import utils.custom_bfast as bfast
from tqdm import tqdm
import digitalhub as dh
from utils.skd_handler import upload_artifact


def deforestation(sensor, years, maindir, boscopath, datapath, outpath):
    start_time = time.time()

    # Check sensor type and get tile list
    if sensor == 'S2':
        tiledict = getTileList(datapath)
    else:
        raise IOError('Invalid sensor')

    for k, tileDatapath in tiledict.items():
        print(f"Reading Tile-{k}.")
        if sensor == 'S2':
            tile = L2Atile(os.path.join(maindir, datapath), tileDatapath)

        # Storage for features
        feature_file = os.path.join(outpath, 'feature_all.dat')
        timestep_index = 0
        all_dates = []

        for y in years:
            y_int = int(y)
            temppath = fm.joinpath(maindir, datapath, 'numpy', k)

            # Get features for the year
            ts, _, _ = tile.gettimeseries(year=y_int, option='default')
            fn = [f for f in os.listdir(temppath)]
            if len(ts) == 0:
                continue

            print(f'Extracting features for {y_int}:')
            height, width = ts[0].feature('B04').shape
            geotransform, projection = fm.getGeoTIFFmeta(ts[0].featurepath()['B04'])

            if timestep_index == 0:
                # Initialize memory-mapped array with correct size
                n_timesteps_total = sum([len(tile.gettimeseries(year=int(yy), option='default')[0]) for yy in years])
                feature_all = np.memmap(feature_file, dtype='float16', mode='w+', shape=(height, width, n_timesteps_total))
                bosco_mask = fm.shapefile_to_array(boscopath, geotransform, projection, height, width, attribute='objectid')

            ts = sorted(ts, key=lambda x: x.InvalidPixNum())
            totimg = len(ts)

            for idx, img in enumerate(ts):
                print(f'.. {idx+1}/{totimg}      ', end='\r')

                # Bands
                b1 = img.feature('BLUE', dtype=np.float16)
                b3 = img.feature('RED', dtype=np.float16)
                b4 = img.feature('nir', dtype=np.float16)
                b5 = img.feature('SWIR1', dtype=np.float16)

                # Indices
                NDVI = _ndi(b4, b3)
                BSI = _bsi(b1, b3, b4, b5)
                fuse_feature = fuse_features(NDVI, BSI)

                # Mask
                name = fn[idx]
                maskpath = fm.joinpath(temppath, name, 'MASK.npy')
                msk = np.load(maskpath)
                feature_mask = np.where(msk, np.nan, fuse_feature)

                # Store
                feature_all[:, :, timestep_index] = feature_mask
                all_dates.append(img._metadata['date'])
                timestep_index += 1

                del b1, b3, b4, b5, NDVI, BSI, fuse_feature, feature_mask, msk

        # Flush features
        feature_all.flush()

        # Convert dates
        all_dates_datetime = [datetime.strptime(date, '%Y%m%d') for date in all_dates]

        # Group dates by year
        dates_by_year = {}
        for y in years:
            y_int = int(y)
            dates_by_year[y_int] = [date for date in all_dates_datetime if date.year == y_int]

        # Apply bosco mask
        feature_data = np.where(bosco_mask[:, :, np.newaxis] == 0, np.nan, feature_all).astype(np.float16)
        height, width, time_steps = feature_data.shape

        # Flatten
        flat_pixels = feature_data.reshape(-1, time_steps)
        valid_mask = ~np.isnan(flat_pixels).all(axis=1)
        valid_pixels = flat_pixels[valid_mask]

        print(f"Total pixels: {flat_pixels.shape[0]}, Valid pixels: {valid_pixels.shape[0]}")

        # Interpolation
        print('Generating monthly samples:')
        interpolated_feature_path = os.path.join(outpath, f"interpolated_feature_{k}.npy")

        if os.path.exists(interpolated_feature_path):
            print(f"Interpolated feature already exists: {interpolated_feature_path}")
            interpolated_feature = np.load(interpolated_feature_path)
        else:
            interpolated_valid = Parallel(n_jobs=6)(
                delayed(interpolate_time_series)(px, dates_by_year)
                for px in tqdm(valid_pixels, desc="Interpolating")
            )
            interpolated_valid = np.stack(interpolated_valid).astype(np.float16)

            new_time_steps = interpolated_valid.shape[1]
            interpolated_full = np.zeros((height * width, new_time_steps), dtype=np.float16)
            interpolated_full[valid_mask] = interpolated_valid
            interpolated_feature = interpolated_full.reshape(height, width, new_time_steps)

            np.save(interpolated_feature_path, interpolated_feature)
            print(f"Saved interpolated feature to: {interpolated_feature_path}")

        print(f"Interpolated feature shape: {interpolated_feature.shape}")

        # Reshape for BFAST
        totpixels = height * width
        fused_reshaped = interpolated_feature.reshape((totpixels, interpolated_feature.shape[2]))

        # Run BFAST
        print('Running break point detector:')
        startyear = int(years[0])
        endyear = int(years[-1])
        freq = 12
        nyear = endyear - startyear
        years_np = np.arange(startyear, endyear+1)
        dates = bfast.r_style_interval((startyear, 1), (startyear + nyear, 365), freq).reshape(fused_reshaped.shape[1], 1)

        changemaps_path = os.path.join(outpath, f"changemaps_year_{k}.npy")
        accuracymaps_path = os.path.join(outpath, f"accuracymaps_{k}.npy")

        if os.path.exists(changemaps_path) and os.path.exists(accuracymaps_path):
            print("Loading precomputed maps.")
            changemaps = np.load(changemaps_path)
            accuracymaps = np.load(accuracymaps_path)
        else:
            print(f"Input array shape : {fused_reshaped.shape}")
            batch_size = int(totpixels / 10)
            num_batches = int(np.ceil(fused_reshaped.shape[0] / batch_size))

            all_breaks, all_confidence = [], []
            for i in range(num_batches):
                start = i * batch_size
                end = min((i + 1) * batch_size, fused_reshaped.shape[0])
                print(f"Processing batch {i+1}/{num_batches} ({end - start} pixels)")

                batch_data = fused_reshaped[start:end]
                with Parallel(n_jobs=-1) as parallel:
                    breaks, confidence = run_bfast_parallel(parallel, batch_data, dates, freq)

                all_breaks.append(breaks)
                all_confidence.append(confidence)

            breaks = np.concatenate(all_breaks, axis=0)
            confidence = np.concatenate(all_confidence, axis=0)


            changemaps = breaks // freq
            accuracymaps = confidence
            changemaps = changemaps.reshape(height, width)
            accuracymaps = accuracymaps.reshape(height, width)


            # Convert index to year
            changemaps_year = np.zeros_like(changemaps, dtype=int)
            for i, year in enumerate(years_np):
                changemaps_year[changemaps == i] = year

        output_changemaps_year = fm.joinpath(outpath, f"Changemap_{k}_beforepostprocessing.tif")
        fm.writeGeoTIFFD(output_changemaps_year, np.stack([changemaps_year, accuracymaps], axis=-1), geotransform, projection)        



        # Post-processing
        print('Start post processing:')
        updated_change_array, updated_probability_array = pp.remove_isolated_pixels(changemaps_year, accuracymaps)

        print('Fill gaps and update probabilities:')
        final_change_array, final_probability_array = pp.fill_small_holes_and_update_probabilities(
            updated_change_array, updated_probability_array
        )

        final_change_array = final_change_array.astype(float)
        final_probability_array = final_probability_array.astype(float)
        final_change_array[final_change_array == 0] = 0
        final_probability_array[final_probability_array == 0] = 0

        for year in years_np:
            # Mask for change array: keep only the current year
            output_change = np.where(final_change_array == year, final_change_array, 0)

            # Mask for probability array: keep only where year matches
            output_prob = np.where(final_change_array == year, final_probability_array, 0)


            # Save output
            output_change_path = fm.joinpath(outpath, f"Change_{year}_{k}.tif")
            fm.writeGeoTIFF(output_change_path, output_change, geotransform, projection)

            output_prob_path = fm.joinpath(outpath, f"Probability_{year}_{k}.tif")
            fm.writeGeoTIFF(output_prob_path, output_prob, geotransform, projection)

        print("Processing complete!")
        print(f"Execution time: {(time.time() - start_time) / 60:.2f} minutes")


if __name__ == "__main__":
    args = sys.argv[1].replace("'", "\"")
    json_input = json.loads(args)

    sensor = 'S2'
    maindir = '.'
    boscopath = 'bosco'
    datapath = 'data'
    result_folder = 'output'
    outpath = os.path.join(maindir, datapath, result_folder)
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    shape = json_input['shapeArtifactName']
    project_name = os.environ["PROJECT_NAME"]
    data = json_input['dataArtifactName']
    years = json_input['years']
    output_artifact_name = json_input['outputArtifactName']

    print(f"shape: {shape}, data:{data}, years:{years}, output_artifact_name:{output_artifact_name}, project:{project_name}")

    project = dh.get_or_create_project(project_name)
    bosco_artifact = project.get_artifact(shape)
    boscopath = bosco_artifact.download(boscopath, overwrite=True)

    data = project.get_artifact(data)
    datapath = data.download(datapath, overwrite=True)

    deforestation(sensor, years, maindir, boscopath, datapath, outpath)

    # Zip results
    zip_file = os.path.join(outpath, output_artifact_name + '.zip')
    print(f"Creating zip file: {zip_file}")
    zf = zipfile.ZipFile(zip_file, "w")
    for dirname, subdirs, files in os.walk(outpath):
        for filename in files:
            if filename.endswith('.tif') or filename.endswith('.tiff'):
                print(f"Adding {filename} to the zip file")
                zf.write(os.path.join(dirname, filename), arcname=filename)
    zf.close()

    print(f"Uploading artifact: {zip_file}")
    upload_artifact(artifact_name=output_artifact_name, project_name=project_name, src_path=zip_file)
