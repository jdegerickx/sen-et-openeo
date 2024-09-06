
# IMPORTS
import numpy as np
import pandas as pd
from pathlib import Path
import rasterio
import glob
import geopandas as gpd
from matplotlib import pyplot as plt
from numba import guvectorize

RIO_GDAL_OPTIONS = {'GDAL_CACHEMAX': 0}

# FUNCTIONS


def readraster_points(infile, gdf):

    # Prepare coordinates
    coord_list = [(x, y) for x, y in zip(gdf["geometry"].x,
                                         gdf["geometry"].y)]

    # read data
    with rasterio.Env(**RIO_GDAL_OPTIONS):
        with rasterio.open(infile) as src:
            # nbands = src.count
            nodata = src.nodata
            scale = src.scales[0]
            offset = src.offsets[0]
            vals = np.array([x for x in src.sample(coord_list)])

    # scale and apply nodata value
    if nodata is not None:
        vals = vals.astype(np.float32)
        vals[vals == nodata] = np.nan
    vals = (vals * scale) + offset

    vals = np.expand_dims(np.squeeze(vals), axis=0)

    return vals


def check_geom(row):
    try:
        result = row["geometry"].contains(row["centroid"])
    except:
        result = False
    return result


def read_data_files(infiles, gdf):

    # ensure CRS matches
    with rasterio.open(infiles[0]) as src:
        gdf = gdf.to_crs(src.crs)

    data = []
    dates = []

    for infile in infiles:
        data.append(readraster_points(infile, gdf))
        dates.append(pd.to_datetime(infile.split("_")[-2]))

    data = np.concatenate(data, axis=0)

    return data, dates


def plot_timeseries_single_field(plot_data, sample_id, outfile):

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    for label, data in plot_data.items():
        dates, vals = data
        # get style and color
        if label == "NDVI":
            color = "green"
            style = "dashed"
            ax1.plot(dates, vals, label=label,
                     color=color, linestyle=style,
                     marker="o", markerfacecolor=color)
            ax1.set_ylabel(label, color=color)
        elif label == "LST-Ta":
            color = "red"
            style = "solid"
            ax2.plot(dates, vals, label=label,
                     color=color, linestyle=style,
                     marker="o", markerfacecolor=color)
            ax2.set_ylabel(label, color=color)
        else:
            raise ValueError(f"Unknown label {label}")

    ax1.set_xlabel("Date")
    for label in ax1.get_xticklabels():
        label.set_rotation(45)
    plt.title(f"Time series of LST-Ta and NDVI for plot {sample_id}")
    plt.tight_layout()

    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile)
    plt.close('all')


def plot_timeseries(shp_path, data_dir, tile,
                    start_date, end_date):

    # Read the shapefile
    gdf = gpd.read_file(shp_path)

    # simplify geometry if necessary
    if gdf.geom_type.values[0] in ['Polygon',
                                   'MultiPolygon']:
        gdf["centroid"] = gdf["geometry"].centroid
        # check whether centroid is in the original geometry
        gdf["centroid_in"] = gdf.apply(lambda x: check_geom(x),
                                       axis=1)
        gdf = gdf[gdf["centroid_in"]]
        gdf.drop(columns=["geometry", "centroid_in"], inplace=True)
        gdf.rename(columns={"centroid": "geometry"}, inplace=True)

    # Get the LST and NDVI files
    lst_files = sorted(
        glob.glob(str(data_dir / f"{tile}" / "005_lst-ta" / f"LST-Ta_*_{tile}.tif")))
    ndvi_files = sorted(
        glob.glob(str(data_dir / f"{tile}" / "006_ndvi" / f"NDVI_*_{tile}.tif")))

    # filter files for dates
    lst_files = [f for f in lst_files
                 if pd.to_datetime(start_date) <=
                 pd.to_datetime(f.split('_')[-2]) <=
                 pd.to_datetime(end_date)]
    ndvi_files = [f for f in ndvi_files
                  if pd.to_datetime(start_date) <=
                  pd.to_datetime(f.split('_')[-2]) <=
                  pd.to_datetime(end_date)]

    # Read the LST and NDVI values
    lst_data, lst_dates = read_data_files(lst_files, gdf)
    ndvi_data, ndvi_dates = read_data_files(ndvi_files, gdf)

    # Plot the time series for each plot separately
    sample_ids = gdf['ID'].values
    for idx, sample_id in enumerate(sample_ids):
        lst_vals = lst_data[:, idx]
        ndvi_vals = ndvi_data[:, idx]
        plot_data = {"LST-Ta": [lst_dates, lst_vals],
                     "NDVI": [ndvi_dates, ndvi_vals]}
        outfile = data_dir / f"{tile}" / "plots" / \
            f"{sample_id}_{tile}_timeseries.png"
        plot_timeseries_single_field(plot_data, sample_id, outfile)
        plt.show()
    fig, ax = plt.subplots()
    for lst_val, ndvi_val in zip(lst_vals, ndvi_vals):
        ax.plot(lst_val, ndvi_val, 'o', alpha=0.5)
    ax.set_xlabel("LST")
    ax.set_ylabel("NDVI")
    ax.set_title(f"Time series of LST and NDVI for {tile}")

    return fig, ax


if __name__ == "__main__":
    # SCRIPT TO RUN

    # Define the paths
    shp_path = Path("/vitodata/aries/test_points_aktc.gpkg")
    # data_dir = Path("/vitodata/aries/Zambia_2")
    data_dir = Path("/vitodata/aries/Zambia_res-corr")
    tile = "35LPD"
    start_date = "2023-09-01"
    end_date = "2024-07-31"

    # # Define the paths
    # basedir = Path('/vitodata/aries/data/ref/ACF/ACF-Mali-Crop-Survey')
    # tile = '30QVD'
    # shp_path = str(basedir / 'Shape' /
    #                f'SurvAgri_Tombouctou_Youwarou_Parcelles_drought_{tile}.gpkg')
    # data_dir = Path("/vitodata/aries/Mali_4")
    # start_date = "2023-06-01"
    # end_date = "2024-04-15"
    # outfile = str(basedir / 'Shape' /
    #               'SurvAgri_Tombouctou_Youwarou_Parcelles_drought.gpkg')

    plot_timeseries(shp_path, data_dir, tile, start_date, end_date)
