import os
import re

import numpy as np
from osgeo import gdal, gdalconst
from osgeo import osr


def get_year_from_path_file(path_file):
    match = re.search(r'(\d{4})\d{4}T\d{6}', path_file)

    if match:
        return match.group(1)


def reproject_and_mosaic(ref_file, processing_tiles):
    ref_ds = gdal.Open(ref_file)
    if not ref_ds:
        return None

    dst_crs = ref_ds.GetProjectionRef()
    dst_transform = ref_ds.GetGeoTransform()
    dst_width, dst_height = ref_ds.RasterXSize, ref_ds.RasterYSize
    dst_datatype = ref_ds.GetRasterBand(1).DataType
    ref_ds = None  # Close the reference dataset explicitly

    first_tile = True

    for tile_path in processing_tiles:
        tile_ds = gdal.Open(tile_path)
        if not tile_ds:
            continue

        reprojected_tile_ds = gdal.GetDriverByName('MEM').Create(
            '', dst_width, dst_height, 1, dst_datatype)
        reprojected_tile_ds.SetProjection(dst_crs)
        reprojected_tile_ds.SetGeoTransform(dst_transform)

        gdal.ReprojectImage(tile_ds, reprojected_tile_ds,
                            None, None, gdalconst.GRA_Mode)

        reprojected_tile = np.array(reprojected_tile_ds.ReadAsArray())
        reprojected_tile_ds = None
        tile_ds = None  # Close the tile dataset explicitly

        if first_tile:
            mosaic = np.array(reprojected_tile)
            first_tile = False
        else:
            valid_indices = mosaic == 0
            mosaic += reprojected_tile * valid_indices

    return mosaic


def boundaries_geotiff(file_path):
    dataset = gdal.Open(file_path)

    if not dataset:
        return None

    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromWkt(dataset.GetProjection())

    ulx, pixel_width, _, uly, _, pixel_height = dataset.GetGeoTransform()

    lrx = ulx + (dataset.RasterXSize * pixel_width)
    lry = uly + (dataset.RasterYSize * pixel_height)

    # Create a transformation object
    transform = osr.CoordinateTransformation(
        spatial_ref, spatial_ref.CloneGeogCS())

    # Transform the corner coordinates
    ulx, uly, _ = transform.TransformPoint(ulx, uly)
    lrx, lry, _ = transform.TransformPoint(lrx, lry)

    dataset = None
    return ulx, lry, lrx, uly


def get_matching_world_cover_raster(file):
    world_cover_year = get_year_from_path_file(file)

    # Images before 2020 receive worldcover2020, images after 2021 receive worldcover2021
    if int(world_cover_year) <= int("2020"):
        version = "V100"
        world_cover_year = "2020"

    if int(world_cover_year) >= int("2021"):
        version = "V200"
        world_cover_year = "2021"

    # Path to the folder containing the World Cover maps
    world_cover_folder = "/data/MTDA/WORLDCOVER/ESA_WORLDCOVER_10M_YEAR_VERSION/MAP/"
    base_filename = "ESA_WorldCover_10m_YEAR_VERSION_N00E000_Map"

    # Replace 'YEAR' in the folder path with the actual year
    world_cover_folder = world_cover_folder.replace(
        'YEAR', str(world_cover_year))
    world_cover_folder = world_cover_folder.replace('VERSION', version)
    world_cover_files = []
    base_filename = base_filename.replace("YEAR", str(world_cover_year))
    base_filename = base_filename.replace('VERSION', version.lower())

    file_bounds = boundaries_geotiff(file)

    # Calculate the bounds of the world cover tiles that cover the file
    world_cover_bounds = (
        int(file_bounds[0] // 3) * 3,
        int(file_bounds[1] // 3) * 3,
        int(file_bounds[2] // 3) * 3 + 3,
        int(file_bounds[3] // 3) * 3 + 3
    )

    for left_bound in range(world_cover_bounds[0], world_cover_bounds[2], 3):
        if left_bound < 0:
            lon_or = "W"
            left_bound = abs(left_bound)
        else:
            lon_or = "E"

        for bottom_bound in range(world_cover_bounds[1], world_cover_bounds[3], 3):
            if bottom_bound < 0:
                lat_or = "S"
                bottom_bound = abs(bottom_bound)
            else:
                lat_or = "N"

            world_cover_filename = base_filename.replace(
                "N00E000", f"{lat_or}{bottom_bound:02d}{lon_or}{left_bound:03d}")
            full_name = os.path.join(
                world_cover_folder, world_cover_filename, f"{world_cover_filename}.tif")
            world_cover_files.append(full_name)

    mosaic = reproject_and_mosaic(file, world_cover_files)

    return mosaic
