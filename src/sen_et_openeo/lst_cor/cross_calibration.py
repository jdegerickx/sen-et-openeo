import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sen_et_openeo.lst_cor.csv_analysis import init_data_analysis, weight_function, COLUMNS
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'


def s3_lst_bias_df(df, conf_int_low=0.1, conf_int_up=0.9):
    columns = ['lst_S3hr', 'validation_LST',
               'S3_time', 'original_S3_vza', 'validation_vza']
    df = df.filter(columns)
    df_no_dir_effects = df[abs(
        df['original_S3_vza'] - df['validation_vza']) < 10]
    # TODO: remove hotspot as well

    grouped_data = df_no_dir_effects.groupby(df['S3_time'])
    df_no_dir_effects['T difference'] = df_no_dir_effects['lst_S3hr'] - \
        df_no_dir_effects['validation_LST']

    # Filter each group to select only the [10% : 90%] interval
    filtered_data = grouped_data.apply(lambda x: x[(x['T difference'] >= x['T difference'].quantile(conf_int_low)) &
                                                   (x['T difference'] <= x['T difference'].quantile(conf_int_up))])

    # Use the filtered data for further calculations
    df_no_dir_effects = filtered_data.reset_index(drop=True)

    model = LinearRegression()
    model.fit(df_no_dir_effects[["validation_LST"]],
              df_no_dir_effects["lst_S3hr"])
    gain, offset, r_squared = model.coef_, model.intercept_, model.score(
        df_no_dir_effects[["validation_LST"]], df_no_dir_effects["lst_S3hr"])

    return gain, offset, r_squared


def s3_lst_bias(csv_files,
                already_processed=False,
                max_vza_difference=None,
                remove_clouds=True,
                remove_hotspot=True, sample_fraction=1, constant=False, plot_regression=False,
                conf_int_low=0.1, conf_int_up=0.9):
    """
    Calculate the bias between two sets of observations (lst_S3hr and validation_LST)
    and compute the weighted mean based on the number of observations for each data point.

    Parameters:
    - df: DataFrame containing the observations
    - weighting_option: Integer specifying the method for weighting the observations.
                        1: Each data point has the same weight.
                        2: Each data point is less important if there are more data points for that observation.
                        3: Each data point is square root less important if there are more data points for that observation.

    Returns:
    - Weighted mean of the bias between lst_S3hr and validation_LST.
    """
    # TODO remove outliers
    if already_processed:
        df = pd.read_csv(csv_files[0], usecols=COLUMNS)
    else:
        df, _ = init_data_analysis(csv_files, max_vza_difference=max_vza_difference,
                                   remove_clouds=remove_clouds,
                                   remove_hotspot=remove_hotspot, sample_fraction=sample_fraction)

    columns = ['lst_S3hr', 'validation_LST', 'S3_time']
    df = df.filter(columns)
    df['difference'] = df['lst_S3hr'] - df['validation_LST']

    grouped_data = df.groupby(df['S3_time'])
    weights = grouped_data['S3_time'].transform(
        lambda x: weight_function(len(x)))

    # Assign the weights to a new column in the DataFrame
    df['observation_weight'] = weights

    # Filter each group to select only the [10% : 90%] interval
    filtered_data = grouped_data.apply(lambda x: x[(x['difference'] >= x['difference'].quantile(conf_int_low)) &
                                                   (x['difference'] <= x['difference'].quantile(conf_int_up))])

    # Use the filtered data for further calculations
    df = filtered_data.reset_index(drop=True)

    # LST S3 = gain * Validation LST + offset
    if constant:
        # Option 1: calculate the weighted mean where each data point has the same weight
        weighted_gain = 1
        weighted_offset = np.average(
            df['difference'], df["observation_weight"])

    else:
        model = LinearRegression()
        # Option 1: calculate the weighted mean where each data point has the same weight
        model.fit(df[["validation_LST"]], df["lst_S3hr"],
                  sample_weight=df['observation_weight'])
        weighted_gain, weighted_offset = model.coef_, model.intercept_

    if plot_regression:
        # Plotting
        plt.figure(figsize=(10, 6))
        plt.scatter(df["validation_LST"], df["lst_S3hr"],
                    c=df["difference"], cmap="coolwarm", alpha=0.6, s=0.01)
        plt.colorbar(label="Bias (S3hr - Validation)")
        plt.plot(df["validation_LST"], df["validation_LST"] * weighted_gain + weighted_offset, label="Equal weight",
                 color="blue")

        plt.xlabel("Validation LST")
        plt.ylabel("LST S3hr")
        plt.title("Bias between LST S3hr and Validation LST")
        plt.legend()
        plt.grid(True)
        plt.show()

    return weighted_gain, weighted_offset


def correct_s3(df, gain, offset):
    df['corrected lst_S3hr'] = (df['lst_S3hr'] - offset)/gain
    plt.scatter(df["validation_LST"],
                df["corrected lst_S3hr"], alpha=0.6, s=0.01)
    plt.show()
    return df
