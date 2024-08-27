import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import curve_fit

from sen_et_openeo.lst_cor.csv_analysis import (init_data_analysis,
                                                inverse_classes_dictionary,
                                                load_csv_as_pd)


def lsf_function(vza):
    lsf = ((1 + 2 * np.cos(vza)) / (np.sqrt(0.96) + 1.92 * np.cos(vza))
           - 1 / 4 * np.cos(vza) / (1 + 2 * np.cos(vza))
           + 0.15 * (1 - np.exp(-0.75 / np.cos(vza)))
           - 1.0304)

    return lsf


def vinnikov_function(vza):
    return 1 - np.cos(vza)


def lsf_model(data, A=1):
    vza_S3, vza_val = np.radians(data[0]), np.radians(data[2])
    lsf_S3 = lsf_function(vza_S3)
    lsf_val = lsf_function(vza_val)

    return A * (lsf_S3 - lsf_val)


def vinnikov_rl_model(data, A, B, k):
    vza_S3, vaa_S3, vza_val, vaa_val = np.radians(data[0]), np.radians(data[1]), np.radians(data[2]), np.radians(
        data[3])
    sza_S3, saa_S3, sza_val, saa_val = np.radians(data[4]), np.radians(data[5]), np.radians(data[6]), np.radians(
        data[7])

    f_n_S3 = np.tan(sza_S3)
    f_S3 = np.sqrt(
        np.power(np.tan(sza_S3), 2) +
        np.power(np.tan(vza_S3), 2) -
        2 * np.tan(sza_S3) * np.tan(vza_S3) * np.cos(vaa_S3 - saa_S3)
    )

    f_n_val = np.tan(sza_val)
    f_val = np.sqrt(
        np.power(np.tan(sza_val), 2) +
        np.power(np.tan(vza_val), 2) -
        2 * np.tan(sza_val) * np.tan(vza_val) * np.cos(vaa_val - saa_val)
    )
    vin = np.cos(vza_val) - np.cos(vza_S3)
    rl_S3 = (np.exp(-k * f_S3) - np.exp(-k * f_n_S3)) / \
        (1 - np.exp(-k * f_n_S3))
    rl_val = (np.exp(-k * f_val) - np.exp(-k * f_n_val)) / \
        (1 - np.exp(-k * f_n_val))

    return A * vin + B * (rl_S3 - rl_val)


def vinnikov_model(data, A=1):
    vza_S3, vza_val = np.radians(data[0]), np.radians(data[2])
    return A * (np.cos(vza_val) - np.cos(vza_S3))


def directional_effects(csv_files, already_processed=False, offset=0, gain=1, sample_fraction=1,
                        land_cover_specific=False,
                        separate_morning_and_evening=False, temperature_range=None, remove_hotspot=True,
                        remove_outliers=True):
    df, _ = init_data_analysis(csv_files, already_processed=already_processed, max_vza_difference=1000,
                               remove_clouds=True,
                               remove_hotspot=remove_hotspot, sample_fraction=sample_fraction)

    df["T difference"] = (df['lst_S3hr'] - offset) / \
        gain - df['validation_LST']
    grouped_data = df.groupby(df['S3_time'])

    model = vinnikov_model

    # Filter each group to select only the [10% : 90%] interval
    # filtered_data = grouped_data.apply(lambda x: x[(x['T difference'] >= x['T difference'].quantile(conf_int_low)) &
    #                                                (x['T difference'] <= x['T difference'].quantile(conf_int_up))])

    # Use the filtered data for further calculations
    # df = filtered_data.reset_index(drop=True)

    print(df[df["T difference"] > 100])

    df["VZA difference"] = df["original_S3_vza"] - df["validation_vza"]

    df_angles = (df["original_S3_vza"], df["original_S3_vaa"],
                 df["validation_vza"], df["validation_vaa"],
                 df["original_S3_sza"], df["original_S3_saa"],
                 df["validation_sza"], df["validation_saa"])
    df["directional kernel difference"] = model(df_angles)

    # df["cos(VZA) difference"] = np.cos(np.radians(df["validation_vza"])) - np.cos(np.radians(df["original_S3_vza"]))
    # if remove_outliers:
    #     # Group data by "S3_time"
    #     grouped = df.groupby("S3_time")
    #     # Calculate standard deviation and mean for "T difference" for each group
    #     std_devs = grouped["T difference"].transform("std")
    #     mean = grouped["T difference"].transform("mean")
    #
    #     # Calculate z-scores
    #     z_scores = (df["T difference"] - mean)/std_devs
    #
    #     # Add z-scores as a new column
    #     df["sigma T difference"] = z_scores
    #     # df["std T difference"] = df["std T difference"]

    #     ransac = linear_model.RANSACRegressor(residual_threshold=20)
    #     ransac.fit(df[["directional kernel difference"]],
    #                df[["T difference"]],
    #                sample_weight=None)
    #     inlier_mask = ransac.inlier_mask_
    #     df = df[inlier_mask]

    if temperature_range is not None:
        t_min, t_max = temperature_range[0], temperature_range[1]
        df = df[(t_min < ((df["validation_LST"]) + (df['lst_S3hr'] - offset) / gain) / 2) &
                (t_max > ((df["validation_LST"]) + (df['lst_S3hr'] - offset) / gain) / 2)]

    if separate_morning_and_evening:
        df_morning = df[df["S3_time"].str[9:11].astype(int) < 12]
        df_evening = df[df["S3_time"].str[9:11].astype(int) > 12]
        data = [df_morning, df_evening]
    else:
        data = [df]

    for pd_df in data:
        plt.figure()
        pd_angles = \
            (pd_df["original_S3_vza"], pd_df["original_S3_vaa"],
             pd_df["validation_vza"], pd_df["validation_vaa"],
             pd_df["original_S3_sza"], pd_df["original_S3_saa"],
             pd_df["validation_sza"], pd_df["validation_saa"])

        if land_cover_specific:
            inverse_classes = inverse_classes_dictionary()

            # Group data by land cover
            grouped = pd_df.groupby("land_cover")

            # Define colors for each land cover
            colors = plt.cm.tab10(np.linspace(0, 1, len(grouped)))

            # Iterate over each group
            for (group_name, group_df), color in zip(grouped, colors):
                angles = \
                    (group_df["original_S3_vza"], group_df["original_S3_vaa"],
                     group_df["validation_vza"], group_df["validation_vaa"],
                     group_df["original_S3_sza"], group_df["original_S3_saa"],
                     group_df["validation_sza"], group_df["validation_saa"])

                # Perform curve fitting for each group
                popt, pcov = curve_fit(model, angles, group_df["T difference"])

                best_A = popt
                std = pcov[0] ** (1 / 2)
                print("Best fit parameter A for land cover",
                      inverse_classes[group_name], ":", best_A, f"(+/-{std})")

                # Plot scatter plot for each group with color and trendline
                plt.scatter(group_df["directional kernel difference"], group_df["T difference"], s=0.05,
                            label=inverse_classes[group_name], color=color)
                plt.plot(group_df["directional kernel difference"], model(
                    angles, popt[0]), color=color)

            popt, pcov = curve_fit(model, pd_angles, pd_df["T difference"])
            # best_A = popt[0]
            std = pcov[0] ** (1 / 2)
            print(f"Best fit parameter A: {popt} +/-{std})")

        else:

            # Perform curve fitting for all data
            popt, pcov = curve_fit(model, df_angles, df["T difference"])

            print("Best fit parameters:", popt)

            # Plot scatter plot for all data with trendline
            plt.scatter(df["directional kernel difference"],
                        df["T difference"], s=0.05, color='blue')
            # plt.plot(df["directional kernel difference"], vinnikov_model(angles, popt[0]),
            #          color='r', label="Vinnikov parameter A")

        plt.xlabel("cos(VZA_ECOSTRESS) - cos(VZA_S3)")
        plt.ylabel("LST S3 - LST ECOSTRESS [K]")
        plt.legend()
    plt.show()


def periodic_weight_function(x, specific_date, data_start_date, data_end_date):
    six_months_later = specific_date + pd.DateOffset(months=6)
    min_date = six_months_later - pd.DateOffset(months=6)
    phase = (x - min_date).days % (6 * 30)  # 6 months is 6*30 days
    # Cosine periodic weight function
    weight = 0.5 - 0.5 * np.cos(2 * np.pi * phase / (6 * 30))
    return weight


def directional_effects_for_each_observation(csv_files, offset=0, gain=1, sample_fraction=1, land_cover_specific=False,
                                             temperature_range=None, remove_hotspot=True,
                                             conf_int_low=0, conf_int_up=1):
    df, _ = init_data_analysis(csv_files, max_vza_difference=1000,
                               remove_clouds=True,
                               remove_hotspot=remove_hotspot, sample_fraction=sample_fraction)
    df["T difference"] = (df['lst_S3hr'] - offset) / \
        gain - df['validation_LST']

    df['S3_time'] = pd.to_datetime(df['S3_time'], format='%Y%m%dT%H%M%S')

    # Calculate weights based on the periodic weight function
    max_date = df['S3_time'].max()
    df['weights'] = df['S3_time'].apply(
        lambda x: periodic_weight_function(x, max_date, df['S3_time'].min(), df['S3_time'].max()))

    # Group data by S3_time
    grouped = df.groupby("S3_time")

    observations = []
    vinnikov_parameters = []
    stds = []
    morning_observations = []

    # Iterate over each group
    for group_name, group_df in grouped:
        observations.append(group_name)
        morning_observations.append(group_name.hour > 12)
        angles = \
            (group_df["original_S3_vza"], group_df["original_S3_vaa"],
             group_df["validation_vza"], group_df["validation_vaa"],
             group_df["original_S3_sza"], group_df["original_S3_saa"],
             group_df["validation_sza"], group_df["validation_saa"])

        # Perform curve fitting for each group with weights
        popt, pcov = curve_fit(
            vinnikov_model, angles, group_df["T difference"], sigma=group_df['weights'], absolute_sigma=True)
        vinnikov_parameters.append(popt)
        stds.append(np.sqrt(np.diag(pcov)))

    dates = pd.to_datetime(observations)

    x_data = pd.date_range(start='2019-01-01', end='2022-01-01', freq='D')
    x1 = pd.to_datetime('2019-01-01')
    x2 = pd.to_datetime('2022-01-01')
    pos = (dates - x1).days

    colors = [
        'r' if morning_observation else 'b' for morning_observation in morning_observations]
    plt.figure()
    for pos, v_params, std, color in zip(pos, vinnikov_parameters, stds, colors):
        plt.errorbar(pos, v_params, yerr=std, color=color, fmt='o')

    plt.xlim([0, (x2 - x1).days])
    plt.xticks(rotation=45)

    date_range = pd.date_range(start=min(x_data), end=max(x_data), freq='MS')
    tick_positions = (date_range - x1).days

    plt.xticks(tick_positions,
               [date.strftime('%b %Y') for date in pd.date_range(
                   start=min(x_data), end=max(x_data), freq='MS')],
               rotation=45)

    plt.show()


def get_baseshape_par(df, gain=1, offset=0, dir_model=vinnikov_model):
    df["T difference"] = (df['lst_S3hr'] - offset) / \
        gain - df['validation_LST']
    model = dir_model

    df_angles = (df["original_S3_vza"], df["original_S3_vaa"],
                 df["validation_vza"], df["validation_vaa"],
                 df["original_S3_sza"], df["original_S3_saa"],
                 df["validation_sza"], df["validation_saa"])
    df["directional kernel difference"] = model(df_angles)

    # Perform curve fitting for all data
    popt, pcov = curve_fit(model, df_angles, df["T difference"])
    std = pcov ** (1 / 2)

    return popt, std


def apply_dir_cor(df, baseshape_par, dir_function=vinnikov_function):
    df["directionally corrected lst_S3hr"] = df["S3 cross-calibrated"] - \
        baseshape_par * dir_function(np.radians(df["original_S3_vza"]))
    df["directionally corrected validation_LST"] = df["validation_LST"] - \
        baseshape_par * dir_function(np.radians(df["validation_vza"]))
    return df


def vinnikov_time_series(csv_files, bias=2.13, sample_fraction=1):
    df, _ = init_data_analysis(csv_files, max_vza_difference=1000,
                               remove_clouds=True,
                               remove_hotspot=True, sample_fraction=sample_fraction)

    # df = df[df["land_cover"] == classes_dictionary["Cropland"]]
    df['S3_time'] = pd.to_datetime(df['S3_time'], format='%Y%m%dT%H%M%S')

    df['T difference'] = df['lst_S3hr'] - df['validation_LST'] - bias
    df['cos VZA difference'] = np.cos(
        df["original_S3_vza"]) - np.cos(df["validation_vza"])

    groups = df.groupby('S3_time')

    coefficients = []
    lower_cis = []
    upper_cis = []
    time_values = []

    for time, group in groups:
        X = group['cos VZA difference']
        y = group['T difference']

        # Fit linear regression model without intercept
        model = sm.OLS(y, X).fit()

        # Get coefficient for 'cos VZA difference' (slope)
        coefficient = model.params

        # Get confidence interval for the coefficient
        ci = model.conf_int(
            alpha=0.05, cols=None).loc['cos VZA difference'].values

        coefficients.append(coefficient)
        lower_cis.append(ci[0])
        upper_cis.append(ci[1])
        time_values.append(time)

    # Convert lists to arrays for easier manipulation
    coefficients = np.array(coefficients)
    lower_cis = np.array(lower_cis)
    upper_cis = np.array(upper_cis)
    time_values = np.array(time_values)

    # Plot time series of mean coefficient A and confidence interval
    plt.figure(figsize=(10, 6))
    plt.plot(time_values, coefficients,
             label='Mean Coefficient A', color='blue')
    plt.fill_between(time_values, lower_cis, upper_cis,
                     color='blue', alpha=0.3, label='95% CI')
    plt.xlabel('Time')
    plt.ylabel('Coefficient A')
    plt.title('Time Series of Coefficient A with 95% CI')
    plt.legend()
    plt.grid(True)
    plt.show()


def scatter_plot_difference_VZA(csv_files, columns, filters=None, title="Scatter Plot", remove_hotspot=False,
                                remove_clouds=True):
    combined_data = load_csv_as_pd(csv_files, columns, filters, remove_hotspot=remove_hotspot,
                                   remove_clouds=remove_clouds)
    combined_data["VZA_S3 - VZA_val"] = combined_data["original_S3_vza"] - \
        combined_data["validation_vza"]

    print(np.mean(combined_data["VZA_S3 - VZA_val"]))
    # Define the color ranges and corresponding colors
    color_ranges = [(-np.inf, -10), (-10, 10), (10, 20), (20, np.inf)]
    colors = ['blue', 'purple', 'orange', 'r']

    def map_color(value):
        for i, (start, end) in enumerate(color_ranges):
            if start <= value < end:
                return colors[i]

    # Apply the color mapping function to create a new column "color"
    combined_data['color'] = combined_data['VZA_S3 - VZA_val'].apply(map_color)
    min_count = combined_data['color'].value_counts().min()

    sample_data = combined_data.groupby(
        'color', group_keys=False).apply(lambda x: x.sample(min_count))
    scatter = sample_data.plot.scatter(
        x="lst_S3hr", y="validation_LST", s=0.05, c='color', alpha=0.6)
    # Store line objects for legend
    legend_lines = []
    # Add linear fitting lines for each color group
    for color, group in sample_data.groupby('color'):
        x = group['lst_S3hr']
        y = group['validation_LST']
        intercept = np.mean(y - x)  # Calculate intercept to make slope 1
        plt.plot(x, x + intercept, color=color)
        legend_lines.append(plt.Line2D(
            [0], [0], color=color, label=f"linear fit with slope 1"))
        # coeffs = np.polyfit(x, y, 1)
        # plt.plot(x, np.polyval(coeffs, x), color=color)

    # Create legend patches
    legend_patches = [mpatches.Patch(color=colors[i], label=f"{range_[0]} to {range_[1]}") for i, range_ in
                      enumerate(color_ranges)]
    legend_patches.extend(legend_lines)

    ref_line = plt.plot(np.linspace(280, 312, 100), np.linspace(280, 312, 100), color='k',
                        label='LST S3 = LST validation')
    plt.legend(handles=legend_patches)

    # Add legend with custom patches
    # plt.legend(handles=legend_patches)
    # plt.plot(np.linspace(280, 312, 100) + 1, np.linspace(280, 312, 100))
    # plt.plot(np.linspace(280, 312, 100) - 1, np.linspace(280, 312, 100))

    plt.title(
        "Scatter Plot of LST Validation vs. S3hr LST with Linear Fits for Different VZA Difference Ranges")
    plt.xlim([292, 310])
    plt.ylim([285, 312])

    plt.show()
