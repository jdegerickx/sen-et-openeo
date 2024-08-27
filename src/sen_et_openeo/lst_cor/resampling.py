from osgeo import gdal
import os

from scipy.constants import Stefan_Boltzmann as sigma

os.environ['PROJ_LIB'] = '/home/louis_snyders/miniconda3/envs/wasdi_conda_upscaling/share/proj'


def WarpS3HRtoValidation(infile, outfile, template, nodatavalue=None, additional_bands=None, convert_to_radiance=True):
    """
    :param convert_to_radiance: True if warping is based on Radiance instead of LST.
    :param additional_bands: List of file paths for additional bands.
    :param infile: Location of geotiff you want to resample
    :param outfile: Location of resampled geotiff
    :param template: Desired Geotiff resampling example
    :param nodatavalue: Nodatavalue to be used, extracted from the input file if not provided
    :return:
    """

    # If nodatavalue is not provided, extract it from the input file
    if nodatavalue is None:
        in_ds = gdal.Open(infile)
        in_proj = in_ds.GetProjection()
        nodatavalue = in_ds.GetRasterBand(1).GetNoDataValue()
        in_ds = None  # Close the input dataset

    if convert_to_radiance:
        # First, creata radiance TIF from the thermal sharpened S3 LST
        in_ds = gdal.Open(infile)
        band = in_ds.GetRasterBand(1)
        proj_or = in_ds.GetProjection()
        gt_or = in_ds.GetGeoTransform()
        sizeX_or = in_ds.RasterXSize
        sizeY_or = in_ds.RasterYSize
        bands_or = in_ds.RasterCount

        radiance_ds = gdal.GetDriverByName('GTiff').Create('/vitodata/aries/test.tif', sizeX_or, sizeY_or, bands_or,
                                                           gdal.GDT_Float32)
        radiance_ds.SetProjection(proj_or)
        radiance_ds.SetGeoTransform(gt_or)
        radiance_band = radiance_ds.GetRasterBand(1)
        data = band.ReadAsArray()
        radiance_data = sigma * ((0.01 * data) ** 4)
        radiance_band.WriteArray(radiance_data)
        radiance_band.SetNoDataValue(band.GetNoDataValue())
        radiance_ds = None
        in_ds = None
        # Radiance TIF created as '/vitodata/aries/test.tif' from thermal sharpened S3 LST

        # Change the infile
        infile = '/vitodata/aries/test.tif'

    r = gdal.Open(template)
    proj = r.GetProjection()
    if proj == '':
        proj = in_proj
    gt = r.GetGeoTransform()
    sizeX = r.RasterXSize
    sizeY = r.RasterYSize
    extent = [gt[0], gt[3] + gt[5] * sizeY, gt[0] + gt[1] * sizeX, gt[3]]

    out_ds = gdal.Warp(outfile,
                       infile,
                       format="GTiff",
                       dstSRS=proj,
                       xRes=gt[1],
                       yRes=gt[5],
                       outputBounds=extent,
                       resampleAlg='cubicspline',
                       dstNodata=nodatavalue,
                       srcNodata=nodatavalue,
                       multithread=True
                       )

    if convert_to_radiance:
        # open the outfile
        out_ds = gdal.Open(outfile, gdal.GA_Update)
        # convert the raster values with the formula: (values/sigma)**(1/4)
        band = out_ds.GetRasterBand(1)
        data = band.ReadAsArray()
        LST = (data / sigma) ** (1 / 4)
        band.WriteArray(LST)
        band.SetNoDataValue(band.GetNoDataValue())  # Set the NoData value
        os.remove('/vitodata/aries/test.tif')

    out_ds = None

    # Add additional bands to the VRT
    if additional_bands:
        for band_name, geotiff_path in additional_bands.items():
            if "MSK" in band_name:
                nodatavalue_MSK = 1
                nodatavalue_add = nodatavalue_MSK
                resample_mode = "max"

            else:
                nodatavalue_add = nodatavalue
                resample_mode = 'cubicspline'

            anglefile_out = os.path.join(os.path.dirname(outfile),
                                         os.path.basename(outfile).replace("upscaled_S3-LSTHR", band_name))
            out_ds = gdal.Warp(anglefile_out,
                               geotiff_path,
                               format="GTiff",
                               dstSRS=proj,
                               xRes=gt[1],
                               yRes=gt[5],
                               outputBounds=extent,
                               resampleAlg=resample_mode,
                               dstNodata=nodatavalue_add,
                               srcNodata=nodatavalue_add,
                               multithread=True
                               )
            # Close the datasets
            out_ds = None

    return


def get_acquistion_angles_files_S3(file_path_upscaled, original_S3_folder_path_senet, include_cloud_MSK=True):
    basename = os.path.basename(file_path_upscaled)
    tile, _, S3_LSTHR, date, _, _, _ = basename.split("_")
    year = date[0:4]
    subfolder = f"S3-SL-2-LST_{date}_{tile}"
    subfolder_path = os.path.join(original_S3_folder_path_senet, tile[0:2], tile[2:3], tile[3:], year, date[:8],
                                  subfolder)

    if include_cloud_MSK:
        file_paths = {f"original_S3_{angle}": os.path.join(subfolder_path, file) for angle in
                      ["vza", "vaa", "sza", "saa", "MSK"] for file in os.listdir(subfolder_path) if angle in file and "aux" not in file}
    else:
        file_paths = {f"original_S3_{angle}": os.path.join(subfolder_path, file) for angle in
                      ["vza", "vaa", "sza", "saa"] for file in os.listdir(subfolder_path) if angle in file and "aux" not in file}

    return file_paths
