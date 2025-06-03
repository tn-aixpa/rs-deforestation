import sys
import numpy as np
import sys, os, time, shutil, json
from os.path import abspath
import utils.filemanager as fm
from utils.S2L2A import L2Atile, getTileList
from utils.utils import _ndi, _bsi
from datetime import datetime
from joblib import Parallel, cpu_count, delayed
from utils.utils import run_bfast_parallel, get_month_numbers, interpolate_for_year, interpolate_time_series, fuse_features, parallel_interpolate
import utils.post_processing as pp
import utils.custom_bfast as bfast
from tqdm import tqdm
import digitalhub as dh
from utils.skd_handler import upload_artifact
import json


def deforestation(sensor, years, maindir, boscopath, datapath, outpath):
    start_time = time.time()
   
    # Check sensor type and get tile list
    if sensor == 'S2':
        tiledict = getTileList(datapath)
    else:
        raise IOError('Invalid sensor')
    
    keys = tiledict.keys()

    for k in keys:
        tileDatapath = tiledict[k]
        print(f"Reading Tile-{k}.")
        
        if sensor == 'S2':
            tile = L2Atile(maindir, tileDatapath)

        # Initialize empty storage for all years
        
        feature_file = os.path.join(outpath, 'feature_all.dat')

        timestep_index = 0  # to keep track of writing index

        all_dates = []    

        for y in years:
            # Set temporary path for the current year
            temppath = fm.joinpath(maindir, 'numpy', k)

            # Get features for the current year
            ts, _, _ = tile.gettimeseries(year=y, option='default')
            fn = [f for f in os.listdir(temppath)] 

            if len(ts) != 0:
                print(f'Extracting features for each image for year {y}:')
            
            # Get some information from data
            height, width = ts[0].feature('B04').shape
            geotransform, projection = fm.getGeoTIFFmeta(ts[0].featurepath()['B04'])
            ts_length = len(ts)

        
           

            if timestep_index == 0:
                # Initialize memory-mapped arrays with estimated total time steps
                n_timesteps_total = sum([len(tile.gettimeseries(year=y, option='default')[0]) for y in years])
                feature_all = np.memmap(feature_file, dtype='float16', mode='w+', shape=(height, width, n_timesteps_total))
                #read bosco map
                geotransform, projection = fm.getGeoTIFFmeta(ts[0].featurepath()['B04'])
                bosco_mask = fm.shapefile_to_array(boscopath, geotransform, projection, height, width, attribute='objectid')
            

            ts = sorted(ts, key=lambda x: x.InvalidPixNum())[0:ts_length]
            totimg = len(ts)

            dates = []
               
            # Compute Index Statistics
            for idx, img in enumerate(ts):        
                print(f'.. {idx+1}/{totimg}      ', end='\r')   
                        
                # Compute NDVI and BSI indices
                b1 = img.feature('BLUE', dtype=np.float16)
                b3 = img.feature('RED', dtype=np.float16)
                b4 = img.feature('nir', dtype=np.float16)
                b5 = img.feature('SWIR1', dtype=np.float16)
                
                
    
                NDVI = _ndi(b4, b3)
                BSI = _bsi(b1, b3, b4, b5)

                fuse_feature = fuse_features(NDVI,BSI)
    
                # Mask for valid values (update if needed)
                #fn = fn[1:]
                name = fn[idx]
                maskpath = fm.joinpath(temppath, name, 'MASK.npy')
                msk = np.load(maskpath)

                feature_mask = (np.where(msk, np.nan, fuse_feature))
                

                feature_all[:, :, timestep_index] = feature_mask


                all_dates.append(img._metadata['date'])
                timestep_index += 1
                

                # Delete intermediate arrays to free memory
                del b3, b4, b5, NDVI, BSI, msk, fuse_feature, feature_mask
                
        # Flush memory-mapped arrays to disk
        feature_all.flush()
        #read the dates
    # Convert the date strings to datetime objects
    all_dates_datetime = [datetime.strptime(date, '%Y%m%d') for date in all_dates]
    
    # Separate dates based on the year
    dates_2018 = [date for date in all_dates_datetime if date.year == 2018]
    dates_2019 = [date for date in all_dates_datetime if date.year == 2019]
    
    
    #feature data
    feature_data = feature_all
    
    #filter by bosco map
    feature_data = np.where(bosco_mask[:,:,np.newaxis] == 0, np.nan, feature_data).astype(np.float16)
    height, width, time_steps = feature_data.shape
    
    # Flatten image and get valid pixel indices (not NaN across all time steps)
    flat_pixels = feature_data.reshape(-1, time_steps)
    valid_mask = ~np.isnan(flat_pixels).all(axis=1)
    valid_pixels = flat_pixels[valid_mask]
    
    print(f"Total pixels: {flat_pixels.shape[0]}, Valid pixels: {valid_pixels.shape[0]}")

    # Interpolation
    print('Generating monthly samples:')

    # Define output file path
    interpolated_feature_path = os.path.join(outpath, "interpolated_feature.npy")

    # Check if file exists
    if os.path.exists(interpolated_feature_path):
        print(f"Interpolated feature already exists at: {interpolated_feature_path}")
        interpolated_feature = np.load(interpolated_feature_path)
    else:
        # Interpolation process
        interpolated_valid = Parallel(n_jobs=6)(
            delayed(interpolate_time_series)(px, dates_2018, dates_2019)
            for px in tqdm(valid_pixels, desc="Interpolating")
        )
        interpolated_valid = np.stack(interpolated_valid).astype(np.float16)

        # Create full 3D array with NaNs
        new_time_steps = interpolated_valid.shape[1]
        interpolated_full = np.zeros((height * width, 24), dtype=np.float16)

        # Fill valid positions
        interpolated_full[valid_mask] = interpolated_valid

        # Reshape back to 3D image
        interpolated_feature = interpolated_full.reshape(height, width, new_time_steps)

        # Save to output
        np.save(interpolated_feature_path, interpolated_feature)
        print(f"Saved interpolated feature to: {interpolated_feature_path}")
        

    print(f"Interpolated feature shape: {interpolated_feature.shape}")  


    # Reshape for BFAST
    totpixels = height * width
    fused_reshaped = interpolated_feature.reshape((totpixels, 24))
   
    
    # Run BFAST
    print('Running break point detector:')

    startyear = int(years[0])
    endyear = int(years[-1]) 
    freq = 12 #monthly data
    nyear = endyear - startyear 
    years_np = np.arange(startyear, endyear+1)
    
    #Save as numpy array

    changemaps_path = os.path.join(outpath, "changemaps_year.npy")
    accuracymaps_path = os.path.join(outpath, "accuracymaps.npy")


    # Check if both files exist
    if os.path.exists(changemaps_path) and os.path.exists(accuracymaps_path):
        print("Loading precomputed change maps and accuracy maps.")
        changemaps_year = np.load(changemaps_path)
        accuracymaps = np.load(accuracymaps_path)

    else:    

        print(f"Input array shape : {fused_reshaped.shape}")
        batch_size = int(totpixels/10)  # Try 1M, 2M, etc.
        num_batches = int(np.ceil(fused_reshaped.shape[0] / batch_size))
        
        all_breaks = []
        all_confidence = []
        dates = bfast.r_style_interval((startyear, 1), (startyear + nyear, 365), freq).reshape(fused_reshaped.shape[1], 1)

        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, fused_reshaped.shape[0])
            print(f"Processing batch {i+1}/{num_batches} ({end - start} pixels)")
            
            batch_data = fused_reshaped[start:end]
        
            with Parallel(n_jobs=-1) as parallel:
                
                breaks, confidence = run_bfast_parallel(parallel, batch_data, dates, freq)
                
            all_breaks.append(breaks)
            all_confidence.append(confidence)    
            
        # Combine all results
        breaks = np.concatenate(all_breaks, axis=0)
        confidence = np.concatenate(all_confidence, axis=0) 

        # Compute change maps and accuracy maps
        changemaps = breaks // freq
        accuracymaps = confidence
        changemaps = changemaps.reshape(height, width)
        accuracymaps = accuracymaps.reshape(height, width)

        # Convert index to year
        changemaps_year = np.zeros_like(changemaps, dtype=int)
        for i, year in enumerate(years_np):
            changemaps_year[changemaps == i] = year


        # Save results
        np.save(changemaps_path, changemaps_year)
        np.save(accuracymaps_path, accuracymaps)
        print(f"Saved changemaps_year to: {changemaps_path}")
        print(f"Saved accuracymaps to: {accuracymaps_path}")      
        
    print('Start post processing:')
    # Remove isolated pixels
    updated_change_array, updated_probability_array = pp.remove_isolated_pixels(changemaps_year, accuracymaps)
    
    print('Fill gaps and update probabilities:')
    # Fill gaps and update probabilities
    final_change_array, final_probability_array = pp.fill_small_holes_and_update_probabilities(updated_change_array, updated_probability_array) 

    final_change_array = final_change_array.astype(float)
    final_probability_array = final_probability_array.astype(float)
    final_change_array[final_change_array ==0 ] = np.nan
    final_probability_array[final_probability_array ==0 ] = np.nan    
    
    # Save output 
    output_filename_process = fm.joinpath(outpath,"CD_2018_2019.tif")
    
    fm.writeGeoTIFFD(output_filename_process, np.stack([final_change_array, final_probability_array], axis=-1), geotransform, projection) 

    print("Processing complete!") 
    
    # End timing
    end_time = time.time()
    
    # Calculate and print time in minutes
    minutes = (end_time - start_time) / 60
    print(f"Execution time: {minutes:.2f} minutes")
         
# "{'shape':'bosco', 'data': 'data', 'years':['2018', '2019'], 'outputArtifactName': 'deforestation_output'}"

if __name__ == "__main__":
    args = sys.argv[1].replace("'","\"")
    json_input = json.loads(args)
    #PREPARE SOME TOOLBOX PARAMETERS
    sensor = 'S2'
    #tilename = 'T32TPR' # must match with tile type in the downloaded sentinel data.
    maindir = '.'
    boscopath = 'bosco'
    datapath = 'data'
    outpath = 'output'
    temppath = fm.joinpath(maindir, 'numpy')
    
    shape = json_input['shapeArtifactName'] #shape artifact name (e.g., 'bosco')
    project_name=os.environ["PROJECT_NAME"] #project name (e.g., 'deforestation')
    data = json_input['dataArtifactName'] #data artifact name (e.g., 'data')
    years = json_input['years'] # list of years to process (e.g., ['2018', '2019'])
    output_artifact_name=json_input['outputArtifactName'] #output artifact name (e.g., 'deforestation_output')

    print(f"shape: {shape}, data:{data}, years:{years}, output_artifact_name:{output_artifact_name}, project:{project_name}")
    #print(f"type shape: {type(shape)}, type data:{type(data)}, type years:{type(years)}, type output_artifact_name:{type(output_artifact_name)}, project:{project_name}")

    # download shape
    project = dh.get_or_create_project(project_name)
    bosco_artifact = project.get_artifact(shape)
    boscopath = bosco_artifact.download(boscopath, overwrite=True)

    # download data
    data = project.get_artifact(data)
    datapath =  data.download(datapath, overwrite=True)
    deforestation(sensor, years, maindir, boscopath, datapath, outpath)
    
    #upload output artifact
    print(f"Upoading artifact: {output_artifact_name}, {output_artifact_name}")
    upload_artifact(artifact_name=output_artifact_name,project_name=project_name,src_path=outpath)