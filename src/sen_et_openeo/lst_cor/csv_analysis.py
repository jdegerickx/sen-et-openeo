# USE aries_data_analysis as interpreter
import math
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LinearRegression, RANSACRegressor

from sen_et_openeo.lst_cor.angles import sat1_earth_sat2_angle

COLUMNS = ["lst_S3hr", "original_S3_vza", "original_S3_vaa", "original_S3_sza", "original_S3_saa",
           "validation_LST", "validation_vza", "validation_vaa",
           "validation_sza", "validation_saa", "land_cover", "S3_time", "validation_CloudMask"]


def weight_function(sample_size, full_sample=10000):
    if sample_size <= full_sample:
        return 1 / full_sample
    else:
        return (1 + math.log10(sample_size / full_sample))/sample_size


def create_sample_csv(csv_files, out_path, file_name="all_data_sample.csv", filters=None,
                      remove_hotspot=True,
                      remove_clouds=True,
                      only_morning=False,
                      only_evening=False,
                      seed=None):
    columns = ["lst_S3hr", "original_S3_vza", "original_S3_vaa", "original_S3_sza", "original_S3_saa",
               "validation_LST", "validation_vza", "validation_vaa",
               "validation_sza", "validation_saa", "land_cover", "S3_time", "validation_CloudMask"]
    sample_pd = load_csv_as_pd(csv_files, columns, filters=filters,
                               remove_hotspot=remove_hotspot,
                               remove_clouds=remove_clouds, only_morning=only_morning, only_evening=only_evening,
                               seed=seed, apply_sampling=True)

    sample_pd.to_csv(os.path.join(out_path, file_name), index=False)
    return


def create_sample_csvs_from_validation_folder(validation_folder, number_of_samples):
    tiles = [item for item in os.listdir(
        validation_folder) if re.match(r'\d{2}[A-Z]{3}', item)]
    difference_folders = [os.path.join(
        validation_folder, tile, "S3_validation_difference") for tile in tiles]

    csv_files = list_csv_files(difference_folders)

    for i in range(1, 1 + number_of_samples):
        create_sample_csv(csv_files, validation_folder,
                          file_name=f"all_data_sample_{i}.csv", seed=i)

    return


def classes_dictionary():
    classes = {"Tree Cover": 10,
               "Schrubland": 20,
               "Grassland": 30,
               "Cropland": 40,
               "Built-up": 50,
               "Bare/Sparse Vegetation": 60,
               "Snow and Ice": 70,
               "Permanent Water Bodies": 80,
               "Herbaceous Wetland": 90}
    return classes


def inverse_classes_dictionary():
    classes = classes_dictionary()
    inverse_classes = {v: k for k, v in classes.items()}
    return inverse_classes


def remove_hotspot_data(dataframe, dist_to_sun=10):
    def far_from_hotspot(vza, vaa, sza, saa):
        sun_sensor_angle = sat1_earth_sat2_angle(vza, sza, vaa, saa)
        return sun_sensor_angle > dist_to_sun

    condition_s3 = far_from_hotspot(dataframe['original_S3_vza'], dataframe['original_S3_vaa'],
                                    dataframe['original_S3_sza'], dataframe['original_S3_saa'])
    condition_val = far_from_hotspot(dataframe['validation_vza'], dataframe['validation_vaa'],
                                     dataframe['validation_sza'], dataframe['validation_saa'])

    # Query DataFrame based on conditions
    dataframe_without_hotspot = dataframe[condition_s3 & condition_val]
    hotspot_pixels = len(dataframe) - len(dataframe_without_hotspot)

    return dataframe_without_hotspot


def list_csv_files(folders):
    csv_files = []
    for folder in folders:
        # List files and directories in the current folder
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.endswith(".csv"):
                    # Append the file path to csv_files
                    csv_files.append(os.path.join(root, file))

    # Remove duplicates
    csv_files = list(set(csv_files))

    # Print and remove duplicate files
    for file in csv_files:
        if csv_files.count(file) > 1:
            print(f"Duplicate file found: {file}")
            csv_files.remove(file)

    return csv_files


def load_csv_as_pd(csv_files, columns, filters=None, sample_fraction=1, remove_hotspot=True, remove_clouds=True,
                   only_morning=False, only_evening=False, apply_sampling=False, seed=None):
    combined_data = []
    for csv_file in csv_files:
        file_data = []
        pixels_file = 0
        original_columns = columns

        if "PDQ" in csv_file or "PDR" in csv_file:
            columns = ["lst_S3hr", "original_S3_vza", "original_S3_vaa", "original_S3_sza", "original_S3_saa",
                       "validation_LST", "validation_view_zenith", "validation_view_azimuth",
                       "validation_solar_zenith", "validation_solar_azimuth", "land_cover", "S3_time",
                       "validation_CloudMask"]
        try:
            # Adjust chunk size as needed
            for chunk in pd.read_csv(csv_file, usecols=columns, chunksize=10000):
                chunk = chunk.sample(frac=sample_fraction)

                if "PDQ" in csv_file or "PDR" in csv_file:
                    chunk = chunk.rename(columns={"validation_view_zenith": "validation_vza",
                                                  "validation_view_azimuth": "validation_vaa",
                                                  "validation_solar_zenith": "validation_sza",
                                                  "validation_solar_azimuth": "validation_saa"})
                if filters:
                    for filter in filters:
                        chunk = chunk.query(filter)

                if remove_clouds:
                    chunk = chunk[chunk['validation_CloudMask'] != 1]

                if remove_hotspot:
                    chunk = remove_hotspot_data(chunk)

                if not only_evening and not only_morning:
                    pass
                elif only_morning and not only_evening:
                    # Only keep data for which int(chunk['"S3_time"'][9:11]) < 12
                    chunk = chunk[chunk["S3_time"].str[9:11].astype(int) < 12]

                elif only_evening and not only_morning:
                    # Only keep data for which int(chunk['"S3_time"'][9:11]) > 12
                    chunk = chunk[chunk['"S3_time"'].str[9:11].astype(
                        int) > 12]

                else:
                    raise Exception(
                        "You can not set only_morning and only_evening True")

                file_data.append(chunk)
            file_df = pd.concat(file_data)

            if apply_sampling:
                pixels_file = len(file_df)

                full_sample = 10000
                if pixels_file == 0:
                    pass

                elif pixels_file < full_sample:
                    combined_data.append(file_df)
                else:
                    sample_size = int(weight_function(
                        pixels_file) * full_sample * pixels_file)
                    combined_data.append(file_df.sample(
                        sample_size, random_state=seed))

            else:
                combined_data.append(file_df)

        except pd.errors.EmptyDataError:
            print(f"The file {csv_file} is empty.")
        #
        # for chunk in pd.read_csv(csv_file, usecols=columns, chunksize=10000):  # Adjust chunk size as needed
        #     chunk = chunk.sample(frac=sample_fraction)
        #
        #     if "PDQ" in csv_file or "PDR" in csv_file:
        #         chunk = chunk.rename(columns={"validation_view_zenith": "validation_vza",
        #                                       "validation_view_azimuth": "validation_vaa",
        #                                       "validation_solar_zenith": "validation_sza",
        #                                       "validation_solar_azimuth": "validation_saa"})
        #     if filters:
        #         for filter in filters:
        #             chunk = chunk.query(filter)
        #
        #     if remove_clouds:
        #         chunk = chunk[chunk['validation_CloudMask'] != 1]
        #
        #     if remove_hotspot:
        #         chunk = remove_hotspot_data(chunk)
        #
        #     if not only_evening and not only_morning:
        #         pass
        #     elif only_morning and not only_evening:
        #         # Only keep data for which int(chunk['"S3_time"'][9:11]) < 12
        #         chunk = chunk[chunk["S3_time"].str[9:11].astype(int) < 12]
        #
        #     elif only_evening and not only_morning:
        #         # Only keep data for which int(chunk['"S3_time"'][9:11]) > 12
        #         chunk = chunk[chunk['"S3_time"'].str[9:11].astype(int) > 12]
        #
        #     else:
        #         raise Exception("You can not set only_morning and only_evening True")
        #
        #     file_data.append(chunk)
        # file_df = pd.concat(file_data)
        # pixels_file = len(file_df)
        #
        # if pixels_file ==0:
        #     pass
        #
        # elif pixels_file < 1000:
        #     combined_data.append(file_df.sample(1000))
        # else:
        #     sample_size = int(weight_function(pixels_file)*1000* pixels_file)
        #     combined_data.append(file_df.sample(sample_size))
        #
        # columns = original_columns

    combined_data = pd.concat(combined_data)

    return combined_data


def init_data_analysis(csv_files, already_processed=False, max_vza_difference=None, remove_clouds=True, remove_hotspot=False, sample_fraction=1,
                       title=None, only_morning=False, only_evening=False):

    if already_processed:
        combined_data = pd.read_csv(csv_files[0], usecols=COLUMNS)
        title = None

    else:
        tiles = [os.path.dirname(file_path).split("/")[6]
                 for file_path in csv_files]
        if title is None:
            title = f"Flanders [{'+'.join(tiles)}] (|$\\Delta$VZA|<{str(max_vza_difference)}°)"

        columns = ["lst_S3hr", "original_S3_vza", "original_S3_vaa", "original_S3_sza", "original_S3_saa",
                   "validation_LST", "validation_vza", "validation_vaa",
                   "validation_sza", "validation_saa", "land_cover", "S3_time", "validation_CloudMask"]

        if max_vza_difference:
            filters = [
                f'abs(original_S3_vza - validation_vza) < {str(max_vza_difference)}']
        else:
            filters = None

        combined_data = load_csv_as_pd(csv_files, columns, filters=filters, sample_fraction=sample_fraction,
                                       remove_hotspot=remove_hotspot,
                                       remove_clouds=remove_clouds, only_morning=only_morning, only_evening=only_evening)

    return combined_data, title


def regression_lines(dataframe, x_column, y_column, classification, cmap, single_regression_line=False,
                     remove_outliers=False):
    if single_regression_line:
        x_all = dataframe[x_column].values.reshape(-1, 1)
        y_all = dataframe[y_column].values.reshape(-1, 1)

        if remove_outliers:
            model_all = RANSACRegressor()
            model_all.fit(x_all, y_all)
        else:
            model_all = LinearRegression()
            model_all.fit(x_all, y_all)

        if not remove_outliers:
            # Print overall regression line statistics
            print("Overall Regression Line:")
            print(f"Slope (Coefficient): {model_all.coef_[0][0]}")
            print(f"Intercept: {model_all.intercept_[0]}")
            print(f"Overall R-squared: {model_all.score(x_all, y_all)}\n")

        # Plot the linear fits on the scatter plot
        plt.plot(x_all, model_all.predict(x_all), color="r", linewidth=2)

        for class_index in range(min(dataframe[classification]), max(dataframe[classification]) + 1):
            subset_data = dataframe[dataframe[classification] == class_index]
            x_subset = subset_data[x_column].values.reshape(-1, 1)
            y_subset = subset_data[y_column].values.reshape(-1, 1)

            # Calculate R-squared value for each subclass
            r_squared_subset = model_all.score(x_subset, y_subset)

            if not remove_outliers:
                # Print subclass-specific information
                print(f"Class Index: {class_index}")
                print(f"R-squared for Subclass: {r_squared_subset}")
                print(f"Number of Scatter Points: {len(subset_data)}\n")

    else:
        linear_fits = {}
        for class_index in range(min(dataframe[classification]), max(dataframe[classification] + 1)):
            subset_data = dataframe[dataframe[classification] == class_index]
            x = subset_data[x_column].values.reshape(-1, 1)
            y = subset_data[y_column].values.reshape(-1, 1)

            if remove_outliers:
                model = RANSACRegressor()
                model.fit(x, y)
            else:
                model = LinearRegression()
                # model = LinearRegression(fit_intercept=False)
                model.fit(x, y)

            linear_fits[class_index] = model

            l_cover = np.mean(subset_data["land_cover"])

            # Print regression line statistics
            if not remove_outliers:
                print(f"Class Index: {l_cover}")
                print(f"Slope (Coefficient): {model.coef_[0][0]}")
                print(f"Intercept: {model.intercept_[0]}")
                print(f"R-squared: {model.score(x, y)}\n")
                print(f"Number of Scatter Points: {len(subset_data)}\n")

            # Plot the linear fits on the scatter plot
            plt.plot(x, model.predict(x), color=cmap(class_index), linewidth=2)
    return


def scatter_plot_from_csv(csv_files, max_vza_difference=None, sample_fraction=1, title=None, remove_hotspot=False,
                          remove_clouds=True,
                          classification=None, save_filepath=None, remove_outliers=False, single_regression_line=True):
    """
    Generate a scatter plot from two columns of CSV files.

    Parameters:
    - csv_files (list): List of CSV file paths.
    - x_col (str): The column name for the x-axis.
    - y_col (str): The column name for the y-axis.
    - title (str): The title of the plot (default is "Scatter Plot").
    - xlabel (str): Label for the x-axis (default is None, uses the column name).
    - ylabel (str): Label for the y-axis (default is None, uses the column name).
    """
    combined_data, title = (
        init_data_analysis(csv_files, max_vza_difference=max_vza_difference, remove_clouds=remove_clouds,
                           remove_hotspot=remove_hotspot, sample_fraction=sample_fraction, title=title))

    colors = combined_data[classification].astype('category').cat.codes
    combined_data['colors'] = combined_data[classification].astype(
        'category').cat.codes

    num_unique_dates = len(combined_data[classification].unique())
    cmap = ListedColormap(plt.get_cmap(
        'tab10', num_unique_dates)(range(num_unique_dates)))

    scatter_plot = combined_data.plot.scatter(x="lst_S3hr", y="validation_LST", s=0.01, c=colors, alpha=0.7, cmap=cmap,
                                              colorbar=False)

    # Create a legend using the unique date strings and corresponding colors
    legend_labels = combined_data[classification].unique()
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(i), markersize=5,
                                 label=list(classes_dictionary.keys())[
                                     list(classes_dictionary.values()).index(int(class_label))])
                      for i, class_label in zip(colors.unique(), legend_labels)]

    regression_lines(combined_data, "lst_S3hr", "validation_LST", "colors", cmap,
                     single_regression_line=single_regression_line,
                     remove_outliers=remove_outliers)

    plt.plot(np.linspace(0, 400, 2), np.linspace(
        0, 400, 2), color='k', label="1:1 line")
    scatter_plot.legend(handles=legend_handles + [plt.Line2D([0], [0], color='k', label='1:1 Line')]
                        + [plt.Line2D([0], [0], color='r', label='Regression Line')])

    plt.title(title)

    # Adjust x-axis and y-axis limits based on scatter points
    xlim_min, xlim_max = combined_data['lst_S3hr'].min(
    ), combined_data['lst_S3hr'].max()
    ylim_min, ylim_max = combined_data['validation_LST'].min(
    ), combined_data['validation_LST'].max()
    plt.xlim(xlim_min, xlim_max)
    plt.ylim(ylim_min, ylim_max)

    plt.ylabel("ECOSTRESS Land Surface Temperature [K]")
    plt.xlabel("High-Resolution Sentinel-3 Land Surface Temperature [K]")

    if save_filepath:
        # Create directories if they do not exist
        directory = os.path.dirname(save_filepath)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.rcParams.update({'font.size': 18})
        plt.gcf().set_size_inches(12, 8)
        plt.savefig(save_filepath, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def histograms(csv_files, time_slots_header, classes_header, max_vza_difference=None, sample_fraction=1,
               remove_hotspot=False, remove_clouds=True, save_filepath=None):
    """
    This function generates subfigures for every class.
    Each subfigure shows a normalized smoothed KDE line for each day. The line shows the "LST_S3 - LST_val" data.
    All lines share the same x-axis limits. Zero is the middle of the x-axis.
    A single legend shows the different days. Text annotations display the total number of pixels used for each histogram.

    :param csv_files:
    :param columns:
    :param time_slots_header: This is the header of the column that represents all the dates.
                              The number of different dates in the column corresponds to the number of lines per subfigure.
    :param classes_header: This is the header of the column that represents all the classes.
                           The number of different classes in the column corresponds to the number of subfigures.
    :param classes_dictionary: A dictionary mapping numerical labels to land cover names.
    :param filters:
    :param remove_hotspot:
    :return:
    """
    pd_data, title = (
        init_data_analysis(csv_files, max_vza_difference=max_vza_difference, remove_clouds=remove_clouds,
                           remove_hotspot=remove_hotspot, sample_fraction=sample_fraction, title=None))
    pd_data["LST_S3 - LST_val"] = pd_data["lst_S3hr"] - \
        pd_data["validation_LST"]

    # Get unique dates and classes
    unique_dates = pd_data[time_slots_header].unique()
    unique_classes = pd_data[classes_header].unique()

    # Set up subplots
    num_classes = len(unique_classes)
    num_subplots = num_classes

    # Ensure an even number of subplots
    if num_subplots % 2 != 0:
        num_subplots += 1

    fig, axes = plt.subplots(nrows=num_subplots // 2, ncols=2, figsize=(15, 5 * (num_subplots // 2)), sharey=False,
                             sharex=True)
    fig.suptitle('Smoothened Frequency Histograms (KDE) 31UDS', fontsize=16)

    # Define a color map for days
    cmap = plt.get_cmap('tab20')

    # Find common x-axis limits
    x_min = -7.5
    x_max = 7.5

    for i, class_label in enumerate(unique_classes):
        # Filter data for the current class
        subset_data = pd_data[pd_data[classes_header] == class_label]

        # Clear previous legends and content
        axes[i // 2, i % 2].clear()

        # Plot normalized smoothed KDE line for each day using seaborn
        for j, date in enumerate(unique_dates):
            day_data = subset_data[subset_data[time_slots_header]
                                   == date]["LST_S3 - LST_val"]
            # Ensure the color is consistent for each day across subfigures
            color = cmap(j % 20)

            total_pixels = len(day_data)
            # KDE plot with seaborn
            if int(date[9:11]) < 12:
                linestyle = '--'
            else:
                linestyle = '-'

            sns.kdeplot(day_data, color=color, ax=axes[i // 2, i % 2], label=f'{date} ({total_pixels} p.)', linewidth=2,
                        linestyle=linestyle)

        # Set subplot properties
        axes[i // 2, i % 2].set_title(
            list(classes_dictionary.keys())[list(classes_dictionary.values()).index(int(class_label))])
        axes[i // 2, i % 2].set_xlim(x_min, x_max)
        axes[i // 2, i % 2].legend()
        axes[i // 2, i %
             2].set_xlabel("Temperature Difference (LST_S3 - LST_val) [K]")

    if save_filepath:
        # Create directories if they do not exist
        directory = os.path.dirname(save_filepath)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.rcParams.update({'font.size': 18})
        plt.gcf().set_size_inches(16, 20)
        plt.savefig(save_filepath, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def time_series_plot(csv_files, max_vza_difference=None, sample_fraction=1, title=None, remove_hotspot=False,
                     remove_clouds=True, use_boxplots=False):
    combined_data, title = init_data_analysis(csv_files, max_vza_difference=max_vza_difference,
                                              remove_clouds=remove_clouds,
                                              remove_hotspot=remove_hotspot, sample_fraction=sample_fraction,
                                              title=title)

    combined_data['S3_time'] = pd.to_datetime(
        combined_data['S3_time'], format='%Y%m%dT%H%M%S')

    # Calculate the differences
    combined_data['difference'] = combined_data['lst_S3hr'] - \
        combined_data['validation_LST']

    # Plot scatter figure
    plt.figure(figsize=(10, 6))
    if use_boxplots:
        # grouped_data = combined_data.groupby(combined_data['S3_time'].dt.date)['difference']
        grouped_data = combined_data.groupby(
            combined_data['S3_time'])['difference']

        boxplot_data = [grouped_data.get_group(
            date) for date in grouped_data.groups]
        dates = pd.to_datetime(
            [date for date in list(grouped_data.groups.keys())])

        data_amount = [len(data) for data in boxplot_data]
        weights = np.array([(data / np.mean(data_amount)) ** (1 / 2)
                           for data in data_amount])
        x_data = pd.date_range(start='2019-01-01', end='2022-01-01', freq='D')
        x1 = pd.to_datetime('2019-01-01')
        x2 = pd.to_datetime('2022-01-01')
        pos = (dates - x1).days
        markersize = 0.05

        plt.boxplot(boxplot_data, positions=pos, widths=5 * weights, whis=(10, 90),
                    flierprops=dict(marker='.', markersize=markersize))
        plt.xlim([0, (x2 - x1).days])
        plt.xticks(rotation=45)

        date_range = pd.date_range(
            start=min(x_data), end=max(x_data), freq='MS')
        tick_positions = (date_range - x1).days

        plt.xticks(tick_positions,
                   [date.strftime('%b %Y') for date in pd.date_range(
                       start=min(x_data), end=max(x_data), freq='MS')],
                   rotation=45)
        #
        # # Define sizes for the legend
        # legend_sizes = [100, 1000, 10000]
        # ref_samples = data_amount[0]
        # ref_width = weights[0]*5
        # legend_widths = [np.sqrt(legend_size/ref_samples) * ref_width for legend_size in legend_sizes]
        #
        # # Visual legend for sample size and corresponding box widths
        # legend_handles = []
        # legend_labels= []
        # for index , legend_width in enumerate(legend_widths):
        #     line = plt.Line2D([], [], marker='|', linestyle='None',color='black',markeredgewidth=legend_width/markersize, fillstyle=None)
        #     legend_handles.append(line)
        #     legend_labels.append(f'Sample Size: {legend_sizes[index]}')
        #
        # legend = plt.legend(legend_handles, legend_labels,loc='upper left')

    else:
        plt.scatter(
            combined_data['S3_time'], combined_data['difference'], color='blue', alpha=0.5, s=0.01)

    plt.xlabel('Date')
    plt.ylabel('Difference (lst_S3hr - validation_LST)')
    plt.title('Difference between lst_S3hr and validation_LST over time')

    plt.grid(True)
    plt.show()
