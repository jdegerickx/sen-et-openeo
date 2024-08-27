from sen_et_openeo.data_download import SenETDownload, BlockHelper
from pathlib import Path
import json


def test_data_download():
    spatial_extent_mol = {
        "west": 5.102634,
        "east": 5.432847,
        "south": 51.175899,
        "north": 51.389673,
    }
    temporal_extent = ["2019-06-01", "2019-06-30"]
    output_dir = Path('/data/validation/unit_tests/data_download_mol')

    data_download = SenETDownload(spatial_extent_mol, temporal_extent)
    return_dict = data_download.download(output_dir, output_format='gtiff')


def test_data_download_aoi_file():
    aoi_file = Path('/data/beresilient/lst_results/openeo_test/AOI_34HBH.gpkg')
    temporal_extent = ["2019-06-01", "2019-06-30"]
    output_dir = Path('/data/validation/unit_tests/data_download_aoi')

    data_download = SenETDownload(aoi_file, temporal_extent)
    return_dict = data_download.download(output_dir, output_format='gtiff')


def test_data_download_s2_tile():
    tile = '34HBH'
    temporal_extent = ["2019-07-01", "2019-07-31"]
    output_dir = Path(
        f'/data/validation/unit_tests/et_calc/data_download_{tile}')

    data_download = SenETDownload(tile, temporal_extent)

    data_download.s2_jobid = 'j-24070895b4dd471eb7278ada8a194531'
    data_download.s3_jobid = 'j-240604accf7046fb852ee2312cd5f3c4'
    data_download.dem_jobid = 'j-240708ed28534e96b5d274289b11035d'

    output_files = data_download.run(output_dir)

    # download_dict = data_download.download(output_dir, output_format='gtiff')
    # preprocess_dict = data_download.preprocess(output_dir, delete_unrequired_data=True)
    # sharpening_dict = data_download.sharpening(output_dir)

    test = 1


def test_warp():
    src_ds = Path(
        '/data/validation/unit_tests/data_download/datacube_dem/openEO.tif')
    ref_ds = Path(
        '/data/validation/unit_tests/data_download/datacube_s2/openEO_2019-06-03Z.tif')
    out_ds = Path(
        '/data/validation/unit_tests/data_download/datacube_dem/openEO_warped.tif')

    SenETDownload.reference_warp2(src_ds, out_ds, ref_ds)


def test_netcdf_to_geotiff():
    input_file = Path(
        '/data/validation/unit_tests/data_download_34HBH/34HBH/S3/datacube_s3.nc')
    output_path = Path(
        '/data/validation/unit_tests/data_download_34HBH/34HBH/S3/geotiff')

    return_dict = SenETDownload.netcdf_to_geotiff(input_file, output_path)

    for f in return_dict['confidence_in'].values():
        output_confidence = f.parent / \
            f"confidence_in_converted_{f.stem.split('-')[1]}.tif"
        SenETDownload.convert_s3_confidence_in(f, output_confidence)


def test_mask_extractor():
    input_file = Path('/data/validation/unit_tests/data_download_34HBH/34HBH/S3/geotiff/'
                      'confidence_in-20190701T080000Z.tif')
    output_file = Path('/data/validation/unit_tests/data_download_34HBH/34HBH/S3/geotiff/'
                       'confidence_in_converted-20190701T080000Z.tif')

    SenETDownload.convert_s3_confidence_in(input_file, output_file)


def print_output_dict_as_help():
    return_dict = dict()
    return_dict['SENTINEL2_L2A'] = dict()

    s2_keys = ['B02', 'B03', 'B04', 'B05',
               'B06', 'B07', 'B08', 'B8A', 'B11', 'SCL']

    for key in s2_keys:
        return_dict['SENTINEL2_L2A'][key] = dict()
        return_dict['SENTINEL2_L2A'][key]['<datetime>'] = '<Path>'
        return_dict['SENTINEL2_L2A'][key]['...'] = '...'

    return_dict['SENTINEL3_SLSTR_L2_LST'] = dict()

    slstr_keys = ['LST', 'LST_uncertainty', 'exception', 'confidence_in', 'sunAzimuthAngles', 'sunZenithAngles',
                  'viewAzimuthAngles', 'viewZenithAngles', 'confidence_in_bitlayers', 'lat', 'lon', 'sunAzimuthAngles',
                  'sunZenithAnglesHR', 'viewAzimuthAnglesHR', 'viewZenithAnglesHR', 'latHR', 'lonHR']
    for key in slstr_keys:
        return_dict['SENTINEL3_SLSTR_L2_LST'][key] = dict()
        return_dict['SENTINEL3_SLSTR_L2_LST'][key]['<datetime>'] = '<Path>'
        return_dict['SENTINEL3_SLSTR_L2_LST'][key]['...'] = '...'

    return_dict['COPERNICUS_30'] = {
        'ALT': '<Path>', 'SLO': '<Path>', 'ASP': '<Path>'}
    return_dict['info'] = dict()
    return_dict['info']['spatial_extent'] = {'east': '<float>',
                                             'north': '<float>',
                                             'south': '<float>',
                                             'west': '<float>'}
    return_dict['info']['temporal_extent'] = ['<datetime>', '<datetime>']

    print(json.dumps(return_dict, indent=4))


def test_split_s2():
    dir = Path('/data/validation/unit_tests/data_download_34HBH_final1/34HBH/S2/')
    filelist = [dir / 'SENTINEL2_L2A_2019-07-01Z.tif',
                dir / 'SENTINEL2_L2A_2019-07-11Z.tif',
                dir / 'SENTINEL2_L2A_2019-07-21Z.tif']

    output_dict = SenETDownload._split_geotiff_s2(filelist)
    test = 1


def test_scaled():
    f_in = Path(
        '/data/validation/unit_tests/data_download_34HBH_final1/34HBH/S3/geotiff/LST-20190701T080000Z.tif')
    f_out = Path(
        '/data/validation/unit_tests/data_download_34HBH_final1/34HBH/S3/geotiff/LST-20190701T080000Z_scaled.tif')

    SenETDownload.to_scaled_raster(
        f_in, scale=1.0 / 100.0, offset=0.0, nodata=0, dst_ds=f_out)


def test_dem_features():
    f_in = Path(
        '/data/validation/unit_tests/data_download_34HBH/34HBH/DEM/COPERNICUS_30.tif')
    SenETDownload._extract_dem_features(f_in)
    test = 1


def testblockhelper():
    lores = Path(
        '/data/validation/unit_tests/et_calc/data_download_34HBH/34HBH/002_preprocess/S3/LST-20190701T080000Z.tif')
    hires = Path(
        '/data/validation/unit_tests/et_calc/data_download_34HBH/34HBH/002_preprocess/S2/B02_20190701Z.tif')
    bh = BlockHelper(lores, hires, low_res_x_block_size=35,
                     low_res_y_block_size=35)
    bh.get_hi_res_file(2)
    bh.get_lo_res_file(2)
    test = 1


def testwarp2():
    f_in = Path(
        '/data/validation/unit_tests/et_calc/data_download_34HBH/34HBH/002_preprocess/S3/LST-20190702T070000Z.tif')
    f_ref = Path(
        '/data/validation/unit_tests/et_calc/data_download_34HBH/34HBH/002_preprocess/S2/B02_20190701Z.tif')
    f_out = Path(
        '/data/validation/unit_tests/et_calc/data_download_34HBH/34HBH/002_preprocess/S3/LST-20190702T070000Z_TMTMT.tif')
    SenETDownload.reference_warp_gdal(
        f_in, f_ref, res=(1098.0, -1098.0), dst_ds=f_out)


def test_remove_key_and_file():
    file1 = Path('/tmp/file1')
    file2 = Path('/tmp/file2')

    file1.touch()
    file2.touch()

    test_dict = {'S3': {'1': file1, '2': file2}}
    output_dict = SenETDownload.remove_key_and_file(test_dict, 'S3')
    is_file1 = file1.is_file()
    is_file2 = file2.is_file()
    test = 1


test_data_download_s2_tile()
