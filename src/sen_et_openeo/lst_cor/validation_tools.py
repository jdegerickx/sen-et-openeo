import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal


def get_basic_stats(geotiff_file_path, band_index=1):
    ds = gdal.Open(geotiff_file_path)
    if ds is None:
        print("Error opening GeoTIFF file.")
        return

    # Read the data from the specified band
    band = ds.GetRasterBand(band_index)
    data = band.ReadAsArray()

    # Flatten the 2D array to 1D for histogram plotting
    data_flat = data.flatten()

    # Plot the histogram
    # data_flat[data_flat < -100] = np.nan
    # data_flat[data_flat > 100] = np.nan

    mean = np.nanmean(data_flat)
    std = np.nanstd(data_flat)
    ds = None
    return mean, std


def display_histogram_geotiff(geotiff_file_path, band_index=1):
    # Open the GeoTIFF file
    ds = gdal.Open(geotiff_file_path)
    if ds is None:
        print("Error opening GeoTIFF file.")
        return

    # Read the data from the specified band
    band = ds.GetRasterBand(band_index)
    data = band.ReadAsArray()

    # Flatten the 2D array to 1D for histogram plotting
    data_flat = data.flatten()

    # Plot the histogram
    data_flat[data_flat < -100] = np.nan
    data_flat[data_flat > 100] = np.nan

    print(f"mean: {np.nanmean(data_flat)}")
    print(f"standard deviation: {np.nanstd(data_flat)}")
    plt.hist(data_flat, bins=1000, color='blue', alpha=0.7, range=(-4, 4))
    plt.title(f'')
    plt.xlabel('$\Delta T [K]$')
    plt.ylabel('Pixels')
    plt.show()

    # Close the dataset
    ds = None


def scatter_plot_two_geotiffs(original_file, validation_file, original_band_index=1, validation_band_index=1):
    original_ds = gdal.Open(original_file)
    original_band = original_ds.GetRasterBand(original_band_index)
    original_scale_factor = original_band.GetScale() if original_band.GetScale() else 1

    validation_ds = gdal.Open(validation_file)
    validation_band = validation_ds.GetRasterBand(validation_band_index)
    validation_scale_factor = validation_band.GetScale(
    ) if validation_band.GetScale() else 1

    validation_array = validation_band.ReadAsArray() * validation_scale_factor
    original_array = original_band.ReadAsArray() * original_scale_factor

    original_array = np.where(
        (original_array > 250) & (validation_array > 250) & (
            validation_array < 380) & (original_array < 380),
        original_array, np.nan)
    validation_array = np.where(
        (original_array > 250) & (validation_array > 250) & (
            validation_array < 380) & (original_array < 380),
        validation_array, np.nan)

    validation_data = validation_array.flatten()
    original_data = original_array.flatten()

    original_data = original_data[~np.isnan(original_data)]
    validation_data = validation_data[~np.isnan(validation_data)]

    x = np.linspace(300, 315, 10)
    # Create a scatter plot
    plt.scatter(original_data, validation_data, marker='.',
                s=0.0001, color='blue', alpha=0.7)
    plt.plot(x, x, color='r')
    plt.ylim([304, 311])
    plt.xlim([304, 311])
    plt.title("")
    plt.xlabel('High-Resolution S3 LST [K] resampled to ECOSTRESS resolution')
    plt.ylabel('ECOSTRESS LST [K]')
    plt.grid(True)
    plt.show()


def get_difference_geotiff(original_file, validation_file, output_file, original_band_index=1, validation_band_index=1):
    # Open the original and validation GeoTIFF files
    original_ds = gdal.Open(original_file)
    original_band = original_ds.GetRasterBand(original_band_index)
    original_scale_factor = original_band.GetScale() if original_band.GetScale() else 1

    validation_ds = gdal.Open(validation_file)
    validation_band = validation_ds.GetRasterBand(validation_band_index)
    validation_scale_factor = validation_band.GetScale(
    ) if validation_band.GetScale() else 1

    if original_ds is None or validation_ds is None:
        print("Error opening GeoTIFF files.")
        return

    validation_array = validation_band.ReadAsArray() * validation_scale_factor
    original_array = original_band.ReadAsArray() * original_scale_factor
    # # Mask out areas outside the intersection
    original_array = np.where((original_array > 0) & (
        validation_array > 0), original_array, np.nan)
    validation_array = np.where((original_array > 0) & (
        validation_array > 0), validation_array, np.nan)

    # Calculate the difference
    difference_array = original_array - validation_array

    # Create the output GeoTIFF file
    driver = gdal.GetDriverByName("GTiff")
    output_ds = driver.Create(
        output_file, original_ds.RasterXSize, original_ds.RasterYSize, 1, gdal.GDT_Float32)
    output_ds.SetGeoTransform(original_ds.GetGeoTransform())
    output_ds.SetProjection(original_ds.GetProjection())

    # Write the difference to the output band
    output_band = output_ds.GetRasterBand(1)
    output_band.SetNoDataValue(np.nan)
    output_band.WriteArray(np.squeeze(difference_array))

    # Close the datasets
    original_ds = None
    validation_ds = None
    output_ds = None
