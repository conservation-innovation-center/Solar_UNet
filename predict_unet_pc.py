import argparse
import os
# import glob
from os.path import join
import sys
from io import BytesIO
from dotenv import load_dotenv, dotenv_values
import multiprocessing
# from datetime import datetime
from pathlib import Path
import json
from datetime import datetime, timedelta
import numpy as np
from azure.storage.blob import ContainerClient

from osgeo import gdal
from shapely.geometry import Polygon, Point, box
import geopandas as gpd
# import dask_geopandas
import dask.dataframe as dd
# import zarr
import pandas as pd
import xarray as xr
import rioxarray
import rasterio as rio
from rasterio import RasterioIOError
from rasterio.transform import xy, Affine, rowcol
# from rasterio.windows import Window, from_bounds, transform, shape
# from rasterio.vrt import WarpedVRT
from pyproj import CRS
import planetary_computer
import pystac_client
import stackstac
# import stac_vrt
# import tempfile
from scipy.ndimage import zoom
from tensorflow.keras import models # was commented out
import tensorflow as tf

print('contents of root dir', os.listdir('.'))

DIR = Path().resolve()
sys.path.append(str(DIR/'scv'))
S2IMGIDS = []

from utils import pc_tools, raster_tools, processing #, model_tools
from importlib import reload
import sienna_model_tools

def recursive_date_search(dates, aoi = None, collection = ['sentinel-2-l2a'], query={"eo:cloud_cover": {"lt": 10}}):
        # connect to the planetary computer catalog
    catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                                        modifier = planetary_computer.sign_inplace)
    datetimes = [datetime.strptime(date, "%Y-%m-%d") for date in dates.split('/')]
    year = int(dates[0:4])
    delta = timedelta(days = 5)
    search = catalog.search(
        collections = collection,
        datetime = dates,
        intersects = aoi,
        query= query
    )
    items = [item.to_dict() for item in list(search.items())]
    if len(items) > 0: # base-case: we have at least one item matching our criteria
        print(f'found {len(items)} matching items')
        # sort items so those collected closest to original dates is first
        # items.sort(key = lambda x: abs(datetime.strptime(x['properties']['datetime'][0:10], '%Y-%m-%d') - datetimes[-1]))
        # sort the items so the most recent is first
        items.sort(key = lambda x: datetime.strptime(x['properties']['datetime'][0:10], '%Y-%m-%d'), reverse = True)
        return items
    else: # if nothing matches, expand the date range by 5 days and try again
        datetimes = [datetimes[0] - delta, datetimes[1] + delta]
        if (datetimes[0] < datetime(year, 4, 1)) or (datetimes[1] > datetime(year, 10, 31)): # if we get earlier than APril or later than Oc
            datetimes = [datetime.strptime(date, "%Y-%m-%d") for date in dates.split('/')] # revert to original dates
            query['eo:cloud_cover']['lt'] += 5 # increase acceptable cloud cover by 5 # sentinel-2 imagery only available after June 23, 2015
        newdates = f'{datetimes[0]:%Y-%m-%d}/{datetimes[1]:%Y-%m-%d}'
        print(f'trying again with new dates: {newdates}')
        return recursive_date_search(dates = newdates, aoi = aoi, collection = collection, query = query)

# breakpoint()
def predict_chunk(arr, m: models.Model, buff: int, imgsz: int):
    """Prediction function to be used with Dask map_overlaps2"""
    H, W, C = arr.shape
    print(f'chip shape: {(H, W, C)}')
    try:
        assert (H,W,C) == (imgsz + (2*buff), imgsz + (2*buff), 6), "not a full chip"
        preds = m.predict(np.array([arr]), verbose = 0) # return the probability of solar
        # # trim the buffer from prediction trip
        # prediction = preds[0][0,buff:(imgsz + buff),buff:(imgsz + buff),:]
        prediction = preds[0][0,:,:,:]
    except Exception as e:
        print(e)
        prediction = np.full((H,W,1), 2.0)
    finally: 
        print(f'prediction shape {prediction.shape}')
    return prediction

def get_idx(filename):
    """Return the '{x}_{y}' identifying string at the end of a numpy file"""
    path, ext = os.path.splitext(filename)
    if ext != '.json':
        base = os.path.basename(path)
        pieces = base.split('_')
        idx = '_'.join(pieces[-2:])
        return idx
    else:
        pass

def get_existing_files(path):
    generator = Path(path).glob('*.tiff')
    tiffs = [url for url in generator]
    ids = set([get_idx(tiff) for tiff in tiffs])
    return ids

def get_existing_blobs(container_client, path = 'test/train/label/'):
    """Return a list of '{x}_{y}' identifying strings from list of blobs"""
    generator = container_client.list_blobs(name_starts_with = path)
    blobs = [f'{blob.name}' for blob in generator]
    ids = set([get_idx(blob) for blob in blobs])
    return ids

def run(dates, name:str, imgsz:int, buff:int, nclasses:int, nchannels:int, container_client: ContainerClient, out_dir:str, weights = None, sas_token = None, existing = [], multi = False): # ssurgo_table, dem_file,    
    year = dates[0:4]

    # BANDS = json.loads(args.bands)
    OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999)

    METRICS = {
            'logits':[tf.keras.metrics.MeanSquaredError(name='mse'), tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')],
            'classes':[tf.keras.metrics.MeanIoU(num_classes=2, name = 'mean_iou')]
            }

    def get_weighted_bce(y_true, y_pred):
        return sienna_model_tools.weighted_bce(y_true, y_pred, 1)

    m = sienna_model_tools.get_binary_model(depth = nchannels, optim = OPTIMIZER, loss = get_weighted_bce, mets = METRICS, bias = None) # depth = len(BANDS) # not right arguments for current get_binary_model go to satellite computer vision repo and look back for previous commits where get_binary_model has those arguments and copy the function in here 

    if weights:
        if 'https' in weights: # if our weights are in blob storage
            weights = f'{weights}?{sas_token}'
            m = sienna_model_tools.get_blob_weights(m = m, hdf5_url = weights)
        else: # otherwise if weights on local file system
            m.load_weights(weights,by_name=True)
    m.summary()

    side = imgsz + (buff*2)
    query = {"s2:mgrs_tile": {"eq":name}, "eo:cloud_cover":{"lte":5}}
    # retrieve Sentinel-2 imagery from MPC 
    def get_s2_data(dates, query = query): # could just call this in a try catch method on line thats complaining
        s2items = recursive_date_search(dates = dates, collection = ['sentinel-2-l2a'], query = query)
        s2item = s2items[0]
        logInfo = pd.DataFrame(
            {'tile': name,
             'year': year,
             'S2ID': s2item['id'],
             'CloudCover':  s2item['properties']['eo:cloud_cover'],
             'Date':s2item['properties']['datetime'][0:10],
             'Imgsz': imgsz,
             'Buff': buff
             },
             index = [0]) 
        s2Stac = (
            stackstac.stack(s2item,
                            epsg = s2item['properties']['proj:epsg'],
                            assets = ["B02", "B03", "B04", "B08", "B11", "B12"],
                            resolution = 10).where(lambda x: x>0, other = np.nan)
                            ) # xarray dataarray
        
        s2Img = pc_tools.harmonize_to_old(s2Stac) # xarray dataarray

        # median = s2Img.median(dim="time").transpose('y', 'x', 'band')#.rio.write_crs(s2Img.attrs['crs'])
        median = s2Img[0].transpose('y', 'x', 'band') # get single 3D array from only timetsep
        # .rio.write_crs(s2Img.attrs['crs']).rio.clip([aoi.geometry.iloc[0]], crs = 4326)
        s2HWC = pc_tools.normalize_dataArray(median, 'band').rio.write_crs(s2Stac.attrs['crs'])
        return median, s2HWC, logInfo
    
    # s2Img = pc_tools.get_s2_stac(
    #     dates = dates,
    #     aoi = aoi.geometry.iloc[0], 
    #     cloud_thresh = 10,
    #     bands = ["B02", "B03", "B04", "B08", "B11", "B12"],
    #     epsg = None) # xarray dataarray
    

    median, s2HWC, logInfo = get_s2_data(dates = dates, query= query)
    s2Transform = s2HWC.rio.transform() # the math object (matrix w 9 vals) that actually translates pixels of rastor to location  # recalc = True
    s2Res = s2Transform[0]
    s2CRS = s2HWC.rio.crs
    s2EPSG = s2CRS.to_epsg()
    H,W,C = s2HWC.shape

    print('Sentinel-2 epsg', s2EPSG)
    print('Sentinel-2 res', s2Res)
    print('Sentinel-2 shape', s2HWC.shape)

    # aoi_reproj = aoi.to_crs(s2CRS)
    # bounds = aoi_reproj.total_bounds # minx, miny, maxx, maxy
    # print('aoi bounds', bounds)

    # window = from_bounds(*bounds, transform = s2Transform) # the window contains pixels only in the county (a subset of original)
    # H, W = shape(window) # left, bottom, right, top
    # print(H,W)

    # breakpoint()
  
    # trimmed_transform = transform(window, s2Transform)

    # chip indices are row(y), col(x) pixel coordinates relative to window
    chip_indices = raster_tools.generate_chip_indices(round(H), round(W), buff = buff, kernel = imgsz)
    # convert chip indices to absolute UTM coordinates
    geometries = [Point(xy(s2Transform, rows = y, cols = x)) for y,x, in chip_indices]
    # geometries = [Point(xy(trimmed_transform, rows = y, cols = x)) for y,x in chip_indices] 
    print(f'{len(chip_indices)} chip indices')
    # chip indices shifted are row(y), col(x) pixel coordinates of the original s2 img
    chip_indices_shifted = [rowcol(s2Transform, xs = pt.x, ys = pt.y) for pt in geometries]
    # chip_indices_shifted = [(round(y+window.row_off), round(x+window.col_off)) for y, x in chip_indices]

    # breakpoint()
    
    coords = gpd.GeoDataFrame({
      'indices':chip_indices,
      'geometry': geometries, # geom = Point(x,y)
      's2_coords':chip_indices_shifted, #s2_coords is formatted (y,x) 
      'idx':[f"{int(chip_indices[i][1])}_{int(chip_indices[i][0])}" for i in range(len(chip_indices))] # this is going over y, x so it only really iterates over y keeping x 0 or 1
      },
      geometry = 'geometry',
      crs=s2CRS)
    print(f'{len(coords)} Sentinel-2 pts')

    # create a mixer dictionary so we can reconstruct the outputs
    mixer = dict({
        "rows": round(H), 
        "cols": round(W),
        "crs": s2EPSG,
        "size": imgsz,
        "transform": s2Transform
    })
    mixer_client = container_client.get_blob_client(f'{out_dir}/mixer.json')

    with BytesIO() as buffer:
        # json.dump(mixer, buffer).encode()
        buffer.write(json.dumps(mixer).encode())
        buffer.seek(0)
        mixer_client.upload_blob(buffer, overwrite = True) 

    # dask_arr = da.from_array(s2HWC, chunks = (imgsz, imgsz, nchannels))
    # preds = dask_arr.map_overlap(
    #     func = lambda x:predict_chunk(x, m, buff = buff, imgsz=imgsz),
    #     trim = True,
    #     boundary = 'reflect',
    #     depth = (buff, buff, 0),
    #     meta = np.array([[[]]], dtype = 'float32')
    #     ).compute()
    
    # predictions = xr.DataArray(
    #     preds,
    #     coords = {'y':s2HWC.coords['y'][6829:6829+(256*3)], 'x':s2HWC.coords['x'][1987:1987+(256*3)]},
    #     dims = ('y', 'x', 'band')
    # ).rio.set_crs(s2CRS)

    # predictions.rio.set_spatial_dims(x_dim='x', y_dim='y', inplace = True)
    #         # add spatial reference info
    # predictions.rio.write_crs(f"{s2CRS}", inplace = True)

    def predict_chips(row):
        # nonlocal s2HWC
        # nonlocal median
        y, x = row['indices'] # pixel coords relative to window
        # y, x = index
        print('S2 point', row['s2_coords']) 
        Y, X = row['s2_coords'] # pixel coords relative to original s2 image
        try:
            # print(s2HWC)
            s2Chip = s2HWC[Y - buff: Y + imgsz + buff, X - buff: X + imgsz + buff, :].values # we start with Y? See this syntax -- also implies s2HWC is y, x
            print(type(s2Chip))
            assert s2Chip.shape == (side, side, nchannels), f'S2 chip not expected shape ({s2Chip.shape})'
            # get model predictions for current chip
            preds = m.predict(np.array([s2Chip]), verbose = 0) # return the probability of solar <--------------------------------------------------------------------------------------------------------------------- np.array on transforms is now deprecated, is s2chip a transform?
            # trim the buffer from prediction trip
            prediction = preds[0][0,buff:(imgsz + buff),buff:(imgsz + buff),:] # error? maybe the solar model just gives you propabilityies or prob & classes, may say something about the shapes lengths not being good, might have to test manually structure of the prediction
            # s2Chip_corrected = s2Chip[buff:(imgsz + buff),buff:(imgsz + buff),0:3]
            print(prediction.shape) 
            return prediction, y, x
        
        except AssertionError as msg:
            print(msg)
            # get_s2_data(dates, query = query)
            # predict_chips(row)
            return None, None, None
        
        # except Exception as e:
        #     # median, s2HWC = get_s2_data(dates = dates, query = query)
        #     return predict_chips(row)

    def predict_chips_rio(row):
        prediction, y, x = predict_chips(row)
        if prediction is None:
            print(f'skipping {row}')
            pass
        else:
            # stack = np.concatenate([prediction, naip[600:1800,600:1800,:], dem[600:1800,600:1800,:]], axis = -1)
            arr = np.moveaxis(prediction, -1, 0) #CHW
            # s2Chip = np.moveaxis(s2Chip, -1, 0)
            print("array shape", arr.shape)
            # print("s2 shape", s2Chip.shape)
            # print("s2Chip type", s2Chip.dtype)
            # print(s2Chip.min(), s2Chip.max())
            # s2Chip_vis = (s2Chip * 255.0).clip(0, 255).astype(np.uint8)
            # concated = np.concatenate([arr, s2Chip], axis=0)
            C,H,W = arr.shape
            band_list = list(range(1,C+1))
            lon, lat = xy(Affine(*s2Transform), rows = np.arange(y,y+imgsz), cols = np.arange(x, x + imgsz))
            # _, lat = xy(Affine(*trimmed_transform), rows = np.arange(y,y+imgsz), cols = np.repeat(0, imgsz)) # changed from 0
            # lon, _ = xy(Affine(*trimmed_transform), np.repeat(0, imgsz), np.arange(x, x + imgsz)) # changed from 0

            da = xr.DataArray(
                arr,
                coords = {
                    'band': list(range(C)),
                    # 'y':lat, # changed from lat 
                    'y':s2HWC.coords['y'][y:y+imgsz],
                    # 'x':lon # changed from lon
                    'x':s2HWC.coords['x'][x:x+imgsz]
                },
                dims = ('band', 'y', 'x')
            )

            da.rio.set_spatial_dims(x_dim='x', y_dim='y', inplace = True)
            # add spatial reference info
            da.rio.write_crs(f"{s2CRS}", inplace = True)
            # write to cog
            # da.rio.to_raster(f'//chesconse-fs/K/GIS/CBT_NonTidalWetlands/Analysis/Intersection_over_Union/{name}/unet{epoch_id}/tiff/{x}_{y}.tif', driver = 'GTiff', windowed = True)
            with BytesIO() as buffer:
                da.rio.to_raster(buffer, driver = "GTiff", windowed = True)
                buffer.seek(0)
                year = dates[0:4] # added this
                blob_client = container_client.get_blob_client(f'{out_dir}/{year}/tiff/{x}_{y}.tif')
                blob_client.upload_blob(buffer, overwrite = True)

    # identify sampling points that need dem data exported
    # existing = s1_blobs.intersection(naip_blobs, dem_blobs, ssurgo_blobs)
    print("existing", existing)

    # subset coordinates to those falling within the aoi
    # coords_within = coords[coords.geometry.within(aoi_reproj.geometry.iloc[0])]
    # print(coords_within)
    # to_process = coords_within[~coords_within['idx'].isin(existing)]
    to_process = coords[~coords['idx'].isin(existing)] 
    print(f'{len(existing)} out of {len(coords)} existing, {len(to_process)} to go') 
    
    # Create a MirroredStrategy.
    gpus = tf.config.list_physical_devices('GPU')
    print(f'Number of devices: {len(gpus)}')

    if multi:
        to_process_ddf = dd.from_pandas(to_process, npartitions = multiprocessing.cpu_count())
        to_process_ddf.apply(predict_chips_rio, axis = 1, meta = {'':None}).compute()
    else:
        to_process.apply(predict_chips_rio, axis = 1)

    return logInfo

if __name__ == '__main__':
    gdal.UseExceptions()

    parser = argparse.ArgumentParser()
    parser.add_argument('--run_config', type = str, required = True)
    parser.add_argument('--multi', dest = 'multi', action = 'store_true', default = False)
    args = parser.parse_args()

    # setup geospatial engines
    import fiona
    fiona.supported_drivers['KML'] = 'rw'

    # Add PC dask cluster jupyterhub token to environemnt
    env_config = dotenv_values(".env")
    os.environ["CPL_VSIL_USE_TEMP_FILE_FOR_RANDOM_WRITE"] = "YES" # added this since it said "w+ not supported for /vsiaz unless CPL... set to YES"
    os.environ['JUPYTERHUB_API_TOKEN'] = env_config['JUPYTERHUB_API_TOKEN']
    os.environ['PC_SDK_SUBSCRIPTION_KEY'] = env_config['JUPYTERHUB_API_TOKEN']
    sas_token = env_config['SAS_TOKEN']

    # create special folders './outputs' and './logs' which automatically get saved
    os.makedirs('outputs', exist_ok = True)
    os.makedirs('logs', exist_ok = True)
    out_dir = './outputs'
    log_dir = './logs'

    # this contianer is connected to ArcPro
    container_client = ContainerClient.from_container_url(
        container_url = f'https://aiprojects.blob.core.windows.net/solar?{sas_token}')
    
    with open(args.run_config) as f:
        run_config = json.load(f)
    
    # name = run_config["name"]
    year = run_config['dates'][0:4]

    # s2grid = gpd.read_file(f'https://aiprojects.blob.core.windows.net/solar/CPK_solar/boundaries/S2A_tile_grid.kml?{sas_token}',
    #                         driver = 'KML',
    #                         engine = 'fiona')

    local_dir = run_config["data_dir"]
    unet_vars = run_config['unet_vars']
    nchannels = sum([unet_vars[var]['nchannels'] for var in unet_vars.keys() if unet_vars[var]['files'] is not None])
    weights = run_config["weights"]
    epoch_id = Path(weights).stem[-3:]
    data_dir = run_config['data_dir']
    dates = run_config["dates"]
    imgsz = run_config["imgsz"]
    buff = run_config["buff"]
    tile = run_config["tile"]

    # get a list of our existing unique ids for which we have already made predictions
    existing = get_existing_blobs(container_client, f'{data_dir}/{tile}/{year}/tiff/') 

    # existing = get_existing_files(f'{local_dir}/unet{epoch_id}/tiff/')
    print(f'already completed {len(existing)} chips')
    logInfo = run(
        dates = dates,
        imgsz = imgsz,
        buff = buff,
        name = tile,
        nclasses = run_config['nclasses'],
        nchannels = nchannels,
        weights = weights,
        container_client = container_client,
        sas_token = sas_token,
        out_dir = f'{data_dir}/{tile}',
        existing = existing,
        multi = False#args.multi
    )

    log_file = f'{log_dir}/run_metadata.csv'
    if Path(log_file).is_file(): # if our log file exists, append to it, otherwise create it
        log = pd.read_csv(log_file)
        if len(existing) == 0:
            pd.concat([log, logInfo], axis = 0, ignore_index = True).to_csv(log_file, index = False)
    else:
        logInfo.to_csv(log_file, index = False)

    # # set azure credentials as environment variables - this lets gdal interface with blob storage
    os.environ["AZURE_STORAGE_CONNECTION_STRING"]=env_config["AZURE_STORAGE_CONNECTION_STRING"]
    # os.environ["AZURE_STORAGE_ACCOUNT"] = env_config["AZURE_STORAGE_ACCOUNT"] # need the azure storage account + storage key
    # os.environ["AZURE_STORAGE_KEY"] = env_config["AZURE_STORAGE_KEY"]

    # here, i can check which year we are looking at and then see if its the most recent year and if it is not, check any years after and see if this year has any fields that the years after do not
    # pseudcode
    # if year is not most recent then get cogs in years afterward
    # compare this cog to cogs of years afterwards, if the current cog has a 1 where the other field has a 0, get rid of the 1's in the current cog 

    # get current year, get all years, if any years > current year then set that year into the years array ex: ['2024', '2025'] 

    # read rasters into numpy arrays and then use np.stack([]) <- what rasters am i even reading? why don't i try stacking the numpy arrays? (s2HWC is the current year, how do i get the older years? from blob storage?)
    # how do i get the older years?
    # path_to_vrt = f"/vsiaz/solar/output/{name}/{year}/vrt_regen.vrt"
    blob_generator = container_client.list_blobs(name_starts_with = f'{data_dir}/{tile}/{year}/tiff/')
    blobs = [blob.name for blob in blob_generator]
    # tif_list = [f'/vsiaz/solar/{blob}' for blob in blobs if '.tif' in blob if '.tif' in blob]
    tiff_list = [f'/vsiaz/solar/{blob}' for blob in blobs if '.tif' in blob]
    print(tiff_list)
    vrt_file = f'/vsiaz/solar/{data_dir}/{tile}/{year}/{tile}.vrt'
    # vrt_file = '/home/azureuser/Solar_UNet/vrt.vrt'
    print('writing VRT', vrt_file)
    vrt_options = gdal.BuildVRTOptions(bandList=[1])
    vrt = gdal.BuildVRT(vrt_file, tiff_list, options=vrt_options)
    vrt = None
    check = gdal.Open(vrt_file)
    if check is None:
        raise RuntimeError(f"VRT written but unreadable: {vrt_file}")
    cog_file = f'/vsiaz/solar/{data_dir}/{tile}/{year}/{tile}_cog.tif'
    # cog_file = '/home/azureuser/Solar_UNet/cog.tif'
    gdal.Translate(cog_file, vrt_file)

    # blob_generator = container_client.list_blobs(name_starts_with = f'{data_dir}/{name}/{year}/tiff/')
    # blobs = [blob.name for blob in blob_generator]
    # tif_list = [f'/vsiaz/solar/{blob}' for blob in blobs if '.tif' in blob if '.tif' in blob]
    # print('found', len(tif_list), 'tifs. writing vrt')
    # # first write a vrt file aggregating all the tifs
    # if len(tif_list) > 2000:
    #     vrt_file1 = f'/vsiaz/solar/{local_dir}/unet{epoch_id}/tiff/vrt1.vrt'
    #     print('writing VRT', vrt_file1)
    #     vrt = gdal.BuildVRT(vrt_file1, tif_list[0:500])
    #     vrt = None
    #     vrt_file2 = f'/vsiaz/solar/{local_dir}/unet{epoch_id}/tiff/vrt2.vrt'
    #     print('writing VRT', vrt_file2)
    #     vrt = gdal.BuildVRT(vrt_file2, tif_list[500:1000])
    #     vrt = None
    #     vrt_file3 = f'/vsiaz/solar/{local_dir}/unet{epoch_id}/tiff/vrt3.vrt'
    #     print('writing VRT', vrt_file3)
    #     vrt = gdal.BuildVRT(vrt_file3, tif_list[1000:1500])
    #     vrt = None
    #     vrt_file4 = f'/vsiaz/solar/{local_dir}/unet{epoch_id}/tiff/vrt4.vrt'
    #     print('writing VRT', vrt_file4)
    #     vrt = gdal.BuildVRT(vrt_file4, tif_list[1500:2000])
    #     vrt = None
    #     vrt_file5 = f'/vsiaz/solar/{local_dir}/unet{epoch_id}/tiff/vrt5.vrt'
    #     print('writing VRT', vrt_file5)
    #     vrt = gdal.BuildVRT(vrt_file5, tif_list[2000:2500])
    #     vrt = None
    #     vrt_file6 = f'/vsiaz/solar/{local_dir}/unet{epoch_id}/tiff/vrt6.vrt'
    #     print('writing VRT', vrt_file6)
    #     vrt = gdal.BuildVRT(vrt_file6, tif_list[2500:])
    #     vrt = None        
    # else:
    #     vrt_file = f'/vsiaz/solar/{data_dir}/{name}/{year}/vrt.vrt'
    #     print('writing VRT', vrt_file)
    #     vrt = gdal.BuildVRT(vrt_file, tif_list)
    #     vrt = None
    # # now build a COG from the VRT
    # cog_file = f'/vsiaz/solar/{data_dir}/{name}/{year}/cog.tif'
    # gdal.Translate(cog_file, vrt_file) # THIS IS BLOWING MEMORY


