from osgeo import gdal
from numba import njit, uint16
import numpy as np


from sen_et_openeo.utils.geoloader import (saveImgMem,
                                           getrasterinfo)


@njit(uint16[:, :](uint16[:, :]))
def block_downsample_band_majority(band):
    new_shape = (band.shape[0] // 2, band.shape[1] // 2)
    new_band = np.zeros(new_shape, dtype=np.uint16)
    for i in range(0, band.shape[0], 2):
        for j in range(0, band.shape[1], 2):
            values = band[i:i+2, j:j+2].copy().reshape(4)
            new_band[i//2, j // 2] = np.uint16(
                np.bincount(values).argmax())
    band = None
    return new_band


def split_raster(infile, bandnames, replace):
    ''' splits a multiband raster in separate files,
        each file containing one band

        :param infile: (str) filename of original dataset to be split
        :param bandnames: list of band names to be written
        :param replace: (str) part of the original filename
            which should be replaced by band name
    '''
    # open source file
    src_ds = gdal.Open(infile)

    # write each band
    for i in range(1, src_ds.RasterCount + 1):
        outfile = infile.replace(replace, bandnames[i-1])
        out_ds = gdal.Translate(outfile, src_ds, format='GTiff', bandList=[i])
        out_ds = None


def WarpS3toS2(infile, outfile, template, nodatavalue):
    ''' Execute gdalwarp from the command line instead of using the package,
    due to some unexplained artefacts'''

    # retrieve properties from template
    r = gdal.Open(template)
    proj = r.GetProjection()
    gt = r.GetGeoTransform()
    sizeX = r.RasterXSize
    sizeY = r.RasterYSize
    extent = [gt[0], gt[3]+gt[5]*sizeY, gt[0]+gt[1]*sizeX, gt[3]]
    # bands = r.RasterCount

    out_ds = gdal.Warp(outfile,
                       infile,
                       format="GTiff",
                       dstSRS=proj,
                       xRes=gt[1],
                       yRes=gt[5],
                       outputBounds=extent,
                       resampleAlg='cubicspline',
                       dstNodata=nodatavalue,
                       srcNodata=nodatavalue)

    out_ds = None
    # fsplit = geom_tif.split('.')
    # f_out = fsplit[0] + '-HR.' + fsplit[1]
    # cmd = 'gdalwarp -r cubicspline -tr ' + \
    #     str(res) + ' ' + str(res) + ' ' + \
    #     geom_tif + ' ' + f_out + ' -overwrite'
    # HR_geom = os.system(cmd)

    # return HR_geom


def warp_in_memory(data, gt, proj, template_file,
                   resample_alg='cubicspline'):
    '''Subset and reproject to the template file extent and projection
    using gdal'''

    # save input to memory
    ds_in = saveImgMem(data, gt, proj, "MEM")

    # Get template projection, extent and resolution
    t_proj, extent, t_gt, sizeX, sizeY, nbands, _ = getrasterinfo(
        template_file)

    # Resample with GDAL warp
    t_proj = f'EPSG:{t_proj}'
    warpoptions = gdal.WarpOptions(format="MEM",
                                   dstSRS=t_proj,
                                   xRes=sizeX,
                                   yRes=sizeY,
                                   outputBounds=extent,
                                   resampleAlg=resample_alg)
    out_ds = gdal.Warp("",
                       ds_in,
                       options=warpoptions)

    data = []
    for b in range(ds_in.RasterCount):
        data.append(out_ds.GetRasterBand(b + 1).ReadAsArray())
    out_ds = None
    data = np.stack(data, axis=0)

    return data
