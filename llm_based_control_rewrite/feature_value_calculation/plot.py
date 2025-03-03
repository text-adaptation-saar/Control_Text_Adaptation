import argparse
import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# pip install seaborn matplotlib
# pip install statsmodels

def generate_lineplot_and_histogram(file_path, abs_src_feature_list, ratio_list, save_plot_path, is_filtering=False,
                                    x_min_value_list=None, x_max_value_list=None, y_max_value=None):
    print(file_path)
    print(abs_src_feature_list)
    print(ratio_list)
    print(x_min_value_list)
    print(x_max_value_list)
    print(type(file_path))
    print(type(abs_src_feature_list))
    print(type(ratio_list))
    print(type(x_min_value_list))
    print(type(x_max_value_list))

    if not os.path.exists(save_plot_path):
        os.makedirs(save_plot_path)

    df = pd.read_csv(file_path)
    print("generate_lineplot_and_histogram: %s vs %s, From: %s, Save:%s\n" % (ratio_list, abs_src_feature_list, file_path, save_plot_path))

    if is_filtering:
        df = filtering(df, abs_src_feature_list, ratio_list, x_min_value_list, x_max_value_list, y_max_value)
        # Save the DataFrame to a CSV file
        csv_file = save_plot_path + "/filtered_ratio_stats.csv"
        df.to_csv(csv_file, index=False)  # Set index=False to exclude the index column in the CSV file

    # generate_plots(df, abs_src_feature_list, ratio_list, save_plot_path)
    generate_plots(df, abs_src_feature_list, ratio_list, save_plot_path)

def filtering(df, abs_src_feature_list, ratio_list, x_min_value_list, x_max_value_list, y_max_value):

    # for abs_src_feature, ratio_feature in zip(abs_src_feature_list, ratio_list):
    for abs_src_feature, ratio_feature, x_min, x_max in zip(abs_src_feature_list, ratio_list, x_min_value_list, x_max_value_list):
        print("in filtering loop: %s" % (abs_src_feature))

        # filtered_df = filtered_df[(filtered_df[ratio_feature] <= 2.0)]
        filtered_df = df[(df[ratio_feature] <= y_max_value)]
        print("Filter with %s, now size %d" % (ratio_feature, len(filtered_df)))

        # Remove rows where df[source_sentence_length] < min or df[source_sentence_length] > max
        filtered_df = filtered_df[(filtered_df[abs_src_feature] >= x_min) & (filtered_df[abs_src_feature] <= x_max)]
        print("Filter with %s X_min:%d and x_max:%d, now size %d" % (abs_src_feature, x_min, x_max, len(filtered_df)))

        df = filtered_df
    return df

def generate_plots_old(df, abs_src_feature_list, ratio_list, save_plot_path):

    print("\nGenerate the plots: %s vs %s, Save:%s" % (ratio_list, abs_src_feature_list, save_plot_path))

    for abs_src_feature, ratio_feature in zip(abs_src_feature_list, ratio_list):
        print("abs_src_feature:%s and ratio_feature:%s " % (abs_src_feature, ratio_feature))
        # Create the main figure and axes
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Determine the x-axis range from the line plot data
        x_min, x_max = min(df[abs_src_feature].round(0)), max(df[abs_src_feature].round(0))
        print("Determine the x-axis range from the line plot data: x_min = %s, x_max = %s" % (x_min, x_max))
        bins = np.arange(x_min - 1, x_max + 2)  # +2 to include the upper edge for the last bin
        print("bins for histogram: %s" % (bins))

        # Plotting the line graph on ax1 with a label
        # sns.lineplot(x=df[abs_src_feature], y=df[ratio_feature], ax=ax1, color='tab:blue', linewidth=1.0, label=ratio_feature)
        # sns.regplot(x=df[abs_src_feature], y=df[ratio_feature], scatter=False, ci=None, color='tab:red', order=2)

        # Create the secondary y-axis
        ax2 = ax1.twinx()

        # Plotting the histogram on ax2 with a label (no log scale for frequencies)
        sns.histplot(df[abs_src_feature], ax=ax2, bins=bins, color='#FDF3F3', kde=False, stat="count",
                     edgecolor='#F2C0BD', label='No. of ' + abs_src_feature)

        # Annotate each count vertically inside each histogram block
        for p in ax2.patches:
            ax2.annotate(f'{int(p.get_height())}',
                         (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                         ha='center', va='center', fontsize=5, color='black', rotation='vertical')

        # Set the same x-axis range for both plots
        ax1.set_xlim(x_min - 1, x_max + 1)
        ax2.set_xlim(x_min -1, x_max + 1)

        # Bring the lineplot (ax1) to the front
        ax1.set_zorder(ax2.get_zorder() + 1)
        ax1.patch.set_visible(False)  # Make ax1 background transparent

        # Labels, titles, and legend
        ax1.set_xlabel(abs_src_feature)
        ax1.set_ylabel(ratio_feature)
        ax2.set_ylabel('No. of ' + abs_src_feature)
        # Combine legends from both axes
        handles, labels = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles + handles2, labels + labels2, loc='upper right')

        fig.savefig(save_plot_path + "/" + ratio_feature + "_vs_" + abs_src_feature, dpi=300, bbox_inches='tight')


def generate_plots(df, abs_src_feature_list, ratio_feature_list, save_plot_path):

    print("\nGenerate histogram plots! Save:%s" % (save_plot_path))

    for abs_src_feature, ratio_feature in zip(abs_src_feature_list, ratio_feature_list):
        abs_tgt_feature = abs_src_feature.replace("src", "tgt")
        print("abs_src_feature:%s\tabs_tgt_feature:%s\t ratio_feature:%s " % (abs_src_feature, abs_tgt_feature, ratio_feature))
        # Create the main figure and axes
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Determine the x-axis range from the line plot data
        x_min, x_max = min(df[abs_src_feature].round(0)), max(df[abs_src_feature].round(0))
        print("Determine the x-axis range from the line plot data: x_min = %s, x_max = %s" % (x_min, x_max))
        bins = np.arange(x_min - 1, x_max + 2)  # +2 to include the upper edge for the last bin
        # if abs_src_feature == "abs_src_FreqRank":
        #     bins = np.arange(x_min - 1, x_max + 2, 0.5)
        print("bins for histogram: %s" % (bins))

        # Plotting the line graph on ax1 with a label
        sns.lineplot(x=df[abs_src_feature], y=df[ratio_feature], ax=ax1, color='tab:blue', linewidth=1.0, label=ratio_feature)
        # sns.regplot(x=df[abs_src_feature], y=df[ratio_feature], scatter=False, ci=None, color='tab:red', order=2)

        # Create the secondary y-axis
        ax2 = ax1.twinx()

        # Plotting the histogram on ax2 with a label (no log scale for frequencies)
        sns.histplot(df[abs_src_feature], ax=ax2, bins=bins, color='#eaa6ff', kde=False, stat="count",
                     edgecolor='#52006b', label='No. of ' + abs_src_feature)

        # Plotting the histogram on ax2 with a label (no log scale for frequencies)
        sns.histplot(df[abs_tgt_feature], ax=ax2, bins=bins, color='#fdf3f3', kde=False, stat="count",
                     edgecolor='#F2C0BD', label='No. of ' + abs_tgt_feature)

        # Annotate each count vertically inside each histogram block
        for p in ax2.patches:
            ax2.annotate(f'{int(p.get_height())}',
                         (p.get_x() + p.get_width() / 2., int(p.get_height()) + 5),
                         ha='center', va='center', fontsize=5, color='black', rotation='horizontal')


        # Set the same x-axis range for both plots
        ax1.set_xlim(x_min - 1, x_max + 1)
        ax2.set_xlim(x_min -1, x_max + 1)

        # Bring the lineplot (ax1) to the front
        ax1.set_zorder(ax2.get_zorder() + 1)
        ax1.patch.set_visible(False)  # Make ax1 background transparent

        # Labels, titles, and legend
        ax1.set_xlabel(abs_src_feature) #.replace("src_", "")
        ax1.set_ylabel(ratio_feature)
        ax2.set_ylabel('No. of ' + abs_src_feature.replace("src_", ""))
        # Combine legends from both axes
        handles, labels = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles + handles2, labels + labels2, loc='upper right')

        fig.savefig(save_plot_path + "/" + ratio_feature + "_vs_" + abs_src_feature, dpi=300, bbox_inches='tight')


def plot_gold_predicted_actual_values_of_test(test_file_path_with_predicted_ratio, test_file_path_actual_output_ratio,
                                              x_axis_feature, y_axis_feature, save_plot_path):

    print("Plot results %s vs %s, From: %s,%s, Save:%s" % (y_axis_feature, x_axis_feature,
                                                              test_file_path_with_predicted_ratio,
                                                              test_file_path_actual_output_ratio, save_plot_path))
    # Read test actual output ratios
    # Line,abs_src_Length,abs_src_MaxDepDepth,abs_src_MaxDepLength,abs_src_FreqRank,abs_src_Leven,abs_src_WordCount,abs_tgt_Length,abs_tgt_MaxDepDepth,abs_tgt_MaxDepLength,abs_tgt_FreqRank,abs_tgt_Leven,abs_tgt_WordCount,Length_ratio,MaxDepDepth_ratio,MaxDepLength_ratio,FreqRank_ratio,Leven_ratio,WordCount_ratio
    df_test_actual_output_ratio = pd.read_csv(test_file_path_actual_output_ratio)
    no_of_rows, no_of_columns = df_test_actual_output_ratio.shape
    print("no_of_rows: %s, no_of_columns:%s, file:%s" % (no_of_rows, no_of_columns, test_file_path_actual_output_ratio))
    print(df_test_actual_output_ratio.columns)

    # Read LR predicted ratios
    df_test_with_predicted_ratio = pd.read_csv(test_file_path_with_predicted_ratio, nrows=no_of_rows)
    print(df_test_with_predicted_ratio.columns)

    # Sample data
    X_test = np.array(df_test_with_predicted_ratio[x_axis_feature])  # Source sentence word count
    if "ratio" in x_axis_feature:
        X_test_ori = X_test
        X_test = np.round(X_test_ori, 1)
    Y_test = np.array(df_test_with_predicted_ratio[y_axis_feature])  # Word count ratio
    test_predictions_ratio = df_test_with_predicted_ratio["predicted_" + y_axis_feature]
    actual_test_ratio = df_test_actual_output_ratio[y_axis_feature]

    # Visualize the training data and regression line
    # Set a custom color palette with a light color for the histogram
    custom_palette = ["#FDF3F3",  # Light color for the histogram
                      "#F2C0BD",
                      # Add other colors as needed
                      ]
    sns.set_palette(custom_palette)

    # Create the main figure and axes
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Determine the x-axis range from the line plot data
    x_min, x_max = min(X_test), max(X_test)
    print("x_min: " + str(x_min), ", x_max: " + str(x_max))
    print("Determine the x-axis range from the line plot data: x_min = %s, x_max = %s" % (x_min, x_max))
    bins = np.arange(x_min - 1, x_max + 2)  # +2 to include the upper edge for the last bin
    if "ratio" in x_axis_feature:
        bins = np.arange(start=x_min - 0.1, stop=x_max + 0.2, step=0.1)  # +2 to include the upper edge for the last bin
    print("bins for histogram: %s" % (bins))

    # Create a avg line plot
    sns.lineplot(x=X_test, y=test_predictions_ratio, color='red', label='LR Predicted',
                 err_style="band", errorbar=("sd"), estimator='mean')
                 # marker='o', markersize=6) errorbar=(lambda y: (y.min(), y.max()))
    sns.lineplot(x=X_test, y=Y_test, color='gold', label='Gold reference',
                 err_style="band", errorbar=("sd"), estimator='mean')
    sns.lineplot(x=X_test, y=actual_test_ratio, color='blue', label='Obtained by CTG',
                 err_style="band", errorbar=("sd"), estimator='mean')
    # sns.pointplot(x=X_test, y=actual_test_ratio, color='blue', join=False)
    sns.scatterplot(x=X_test, y=actual_test_ratio, color='blue')

    # Create the secondary y-axis
    ax2 = ax1.twinx()

    # Plotting the histogram on ax2 with a label (no log scale for frequencies)
    sns.histplot(X_test, ax=ax2, bins=bins, color='#FDF3F3', kde=False, stat="count",
                 edgecolor='#F2C0BD', label='No. of ' + x_axis_feature)

    # Annotate each count vertically inside each histogram block
    for p in ax2.patches:
        ax2.annotate(f'{int(p.get_height())}',
                     (p.get_x() + p.get_width() / 2., 1),
                     ha='center', va='center', fontsize=5, color='black', rotation='vertical')

    # Set the same x-axis range for both plots
    ax1.set_xlim(x_min - 1, x_max + 1)
    ax2.set_xlim(x_min - 1, x_max + 1)
    if "ratio" in x_axis_feature:
        ax1.set_xlim(x_min - 0.1, x_max + 0.1)
        ax2.set_xlim(x_min - 0.1, x_max + 0.1)

    # Bring the lineplot (ax1) to the front
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)  # Make ax1 background transparent

    # Labels, titles, and legend
    plt.title("%s vs %s" % (y_axis_feature, x_axis_feature))
    ax1.set_xlabel(x_axis_feature)
    ax1.set_ylabel(y_axis_feature)
    ax2.set_ylabel('No. of ' + x_axis_feature)
    # Combine legends from both axes
    handles, labels = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles + handles2, labels + labels2, loc='upper right')

    fig.savefig(save_plot_path + "/plot_" + y_axis_feature + "_vs_" + x_axis_feature, dpi=300, bbox_inches='tight')

def plot_gold_predicted_actual_values_of_test_with_feature_list(test_files_list_path_with_predicted_ratio, test_file_path_actual_output_ratio,
                                              abs_src_feature_list, ratio_feature_list, save_plot_path):

    print("Plot results %s vs %s, From: %s,%s, Save:%s" % (ratio_feature_list, abs_src_feature_list,
                                                           test_files_list_path_with_predicted_ratio,
                                                           test_file_path_actual_output_ratio, save_plot_path))
    # Read test actual output ratios
    # Line,abs_src_Length,abs_src_MaxDepDepth,abs_src_MaxDepLength,abs_src_FreqRank,abs_src_Leven,abs_src_WordCount,abs_tgt_Length,abs_tgt_MaxDepDepth,abs_tgt_MaxDepLength,abs_tgt_FreqRank,abs_tgt_Leven,abs_tgt_WordCount,Length_ratio,MaxDepDepth_ratio,MaxDepLength_ratio,FreqRank_ratio,Leven_ratio,WordCount_ratio
    df_test_actual_output_ratio = pd.read_csv(test_file_path_actual_output_ratio)
    no_of_rows, no_of_columns = df_test_actual_output_ratio.shape
    print("no_of_rows: %s, no_of_columns:%s, file:%s" % (no_of_rows, no_of_columns, test_file_path_actual_output_ratio))
    print(df_test_actual_output_ratio.columns)


    for test_file_path_with_predicted_ratio, abs_src_feature, ratio_feature in zip(test_files_list_path_with_predicted_ratio, abs_src_feature_list, ratio_feature_list):
        print(f"Reading src_feature: {abs_src_feature}, ratio: {ratio_feature} and LR-predicted ratio file: {test_file_path_with_predicted_ratio}, ")
        # Read LR predicted ratios
        df_test_with_predicted_ratio = pd.read_csv(test_file_path_with_predicted_ratio, nrows=no_of_rows)
        print(df_test_with_predicted_ratio.columns)
        # Sample data
        X_test = np.array(df_test_with_predicted_ratio[abs_src_feature])  # Source sentence word count
        Y_test = np.array(df_test_with_predicted_ratio[ratio_feature])  # Word count ratio (gold ref)
        test_predictions_ratio = df_test_with_predicted_ratio["predicted_" + ratio_feature]
        actual_test_ratio = df_test_actual_output_ratio[ratio_feature]  # (test output)

        # Visualize the training data and regression line
        # Set a custom color palette with a light color for the histogram
        custom_palette = ["#FDF3F3",  # Light color for the histogram
                          "#F2C0BD",
                          # Add other colors as needed
                          ]
        sns.set_palette(custom_palette)

        # Create the main figure and axes
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Determine the x-axis range from the line plot data
        x_min, x_max = min(X_test), max(X_test)
        print("Determine the x-axis range from the line plot data: x_min = %s, x_max = %s" % (x_min, x_max))
        bins = np.arange(x_min - 1, x_max + 2)  # +2 to include the upper edge for the last bin
        print("bins for histogram: %s" % (bins))

        # Create a avg line plot
        sns.lineplot(x=X_test, y=test_predictions_ratio, color='red', label='LR Predicted Ratio',
                     err_style="band", errorbar=("sd"), estimator='mean')
        sns.lineplot(x=X_test, y=Y_test, color='gold', label='Gold reference',
                     err_style="band", errorbar=("sd"), estimator='mean')
        sns.lineplot(x=X_test, y=actual_test_ratio, color='blue', label='Obtained Ratio from CTG',
                     err_style="band", errorbar=("sd"), estimator='mean')

        # Create the secondary y-axis
        ax2 = ax1.twinx()

        # Plotting the histogram on ax2 with a label (no log scale for frequencies)
        sns.histplot(X_test, ax=ax2, bins=bins, color='#FDF3F3', kde=False, stat="count",
                     edgecolor='#F2C0BD', label='No. of ' + abs_src_feature)

        # Set the same x-axis range for both plots
        ax1.set_xlim(x_min - 1, x_max + 1)
        ax2.set_xlim(x_min - 1, x_max + 1)

        # Bring the lineplot (ax1) to the front
        ax1.set_zorder(ax2.get_zorder() + 1)
        ax1.patch.set_visible(False)  # Make ax1 background transparent

        # Labels, titles, and legend
        plt.title("%s vs %s" % (ratio_feature, abs_src_feature))
        ax1.set_xlabel(abs_src_feature)
        ax1.set_ylabel(ratio_feature)
        ax2.set_ylabel('No. of ' + abs_src_feature)
        # Combine legends from both axes
        handles, labels = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles + handles2, labels + labels2, loc='upper right')

        fig.savefig(save_plot_path + "/plot_" + ratio_feature + "_vs_" + abs_src_feature, dpi=300, bbox_inches='tight')


def plot_gold_predicted_actual_values_of_test_with_fixed_ratio(test_files_list_path_with_predicted_ratio, test_file_path_actual_output_ratio,
                                              abs_src_feature_list, ratio_feature_list, save_plot_path, fixed_ratio_list):

    for fixed_ratio, test_file_path_with_predicted_ratio, abs_src_feature, ratio_feature in zip(fixed_ratio_list,
            test_files_list_path_with_predicted_ratio, abs_src_feature_list, ratio_feature_list):
        print(f"Reading src_feature: {abs_src_feature}, fixed_ratio: {ratio_feature} and LR-predicted ratio file: {test_file_path_with_predicted_ratio}, ")

        print("Plot results %s vs %s with fixed ratio %s, From: %s,%s, Save:%s" % (ratio_feature_list, abs_src_feature,
                                                                               fixed_ratio,
                                                              test_file_path_with_predicted_ratio,
                                                              test_file_path_actual_output_ratio, save_plot_path))
        # Read test actual output ratios
        # Line,abs_src_Length,abs_src_MaxDepDepth,abs_src_MaxDepLength,abs_src_FreqRank,abs_src_Leven,abs_src_WordCount,abs_tgt_Length,abs_tgt_MaxDepDepth,abs_tgt_MaxDepLength,abs_tgt_FreqRank,abs_tgt_Leven,abs_tgt_WordCount,Length_ratio,MaxDepDepth_ratio,MaxDepLength_ratio,FreqRank_ratio,Leven_ratio,WordCount_ratio
        df_test_actual_output_ratio = pd.read_csv(test_file_path_actual_output_ratio)
        no_of_rows, no_of_columns = df_test_actual_output_ratio.shape
        print("no_of_rows: %s, no_of_columns:%s, file:%s" % (no_of_rows, no_of_columns, test_file_path_actual_output_ratio))
        print(df_test_actual_output_ratio.columns)

        # Read LR predicted ratios
        df_test_with_predicted_ratio = pd.read_csv(test_file_path_with_predicted_ratio, nrows=no_of_rows)
        print(df_test_with_predicted_ratio.columns)

        # Sample data
        X_test = np.array(df_test_actual_output_ratio[abs_src_feature])  # Source sentence word count
        Y_test = np.array(df_test_with_predicted_ratio[ratio_feature])  # gold ref. ratio
        y_fixed_ratio = float(fixed_ratio)
        y_ideal_fixed_ratio = np.round((y_fixed_ratio * X_test), 1) / X_test

        test_lr_predictions_ratio = df_test_with_predicted_ratio["predicted_" + ratio_feature]
        actual_test_ratio = df_test_actual_output_ratio[ratio_feature]

        # Visualize the training data and regression line
        # Set a custom color palette with a light color for the histogram
        custom_palette = ["#FDF3F3",  # Light color for the histogram
                          "#F2C0BD",
                          # Add other colors as needed
                          ]
        sns.set_palette(custom_palette)

        # Create the main figure and axes
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Determine the x-axis range from the line plot data
        x_min, x_max = min(X_test), max(X_test)
        print("Determine the x-axis range from the line plot data: x_min = %s, x_max = %s" % (x_min, x_max))
        bins = np.arange(x_min - 1, x_max + 2)  # +2 to include the upper edge for the last bin
        print("bins for histogram: %s" % (bins))

        # Create a avg line plot
        # sns.lineplot(x=X_test, y=test_predictions_ratio, color='red', label='LR Predicted',
        #              err_style="band", errorbar=("sd"), estimator='mean')
        sns.lineplot(x=X_test, y=y_fixed_ratio, color='red', label='Fixed ratio',
                     err_style="band", errorbar=("sd"), estimator='mean')
        sns.lineplot(x=X_test, y=y_ideal_fixed_ratio, color='green', label='Ideal Fixed ratio',
                     err_style="band", errorbar=("sd"), estimator='mean')
        sns.lineplot(x=X_test, y=Y_test, color='gold', label='Gold reference',
                     err_style="band", errorbar=("sd"), estimator='mean')
        sns.lineplot(x=X_test, y=actual_test_ratio, color='blue', label='Obtained by CTG',
                     err_style="band", errorbar=("sd"), estimator='mean')
        # sns.scatterplot(x=X_test, y=actual_test_ratio, color='blue')

        # Create the secondary y-axis
        ax2 = ax1.twinx()

        # Plotting the histogram on ax2 with a label (no log scale for frequencies)
        sns.histplot(X_test, ax=ax2, bins=bins, color='#FDF3F3', kde=False, stat="count",
                     edgecolor='#F2C0BD', label='No. of ' + abs_src_feature)
        # Annotate each count vertically inside each histogram block
        for p in ax2.patches:
            ax2.annotate(f'{int(p.get_height())}',
                         (p.get_x() + p.get_width() / 2., 1),
                         ha='center', va='center', fontsize=5, color='black', rotation='vertical')

        # Set the same x-axis range for both plots
        ax1.set_xlim(x_min - 1, x_max + 1)
        ax2.set_xlim(x_min - 1, x_max + 1)

        # Bring the lineplot (ax1) to the front
        ax1.set_zorder(ax2.get_zorder() + 1)
        ax1.patch.set_visible(False)  # Make ax1 background transparent

        # Labels, titles, and legend
        plt.title("%s vs %s" % (ratio_feature, abs_src_feature))
        ax1.set_xlabel(abs_src_feature)
        ax1.set_ylabel(ratio_feature)
        ax2.set_ylabel('No. of ' + abs_src_feature)
        # Combine legends from both axes
        handles, labels = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles + handles2, labels + labels2, loc='upper right')

        fig.savefig(save_plot_path + "/plot_" + ratio_feature + "_vs_" + abs_src_feature, dpi=300, bbox_inches='tight')

def plot_all_ratios_vs_abs_src_feature_list(test_files_list_path_with_predicted_ratio, test_files_path_actual_output_ratio,
                                              abs_src_feature_list, ratio_feature_list, save_plot_path, fixed_ratio_list):

    print("Plot results %s vs %s with fixed ratio %s, From: %s,%s, Save:%s" % (ratio_feature_list, abs_src_feature_list,
                                                                               fixed_ratio_list,
                                                           test_files_list_path_with_predicted_ratio,
                                                           test_files_path_actual_output_ratio, save_plot_path))
    # Read test actual output ratios
    # Line,abs_src_Length,abs_src_MaxDepDepth,abs_src_MaxDepLength,abs_src_FreqRank,abs_src_Leven,abs_src_WordCount,abs_tgt_Length,abs_tgt_MaxDepDepth,abs_tgt_MaxDepLength,abs_tgt_FreqRank,abs_tgt_Leven,abs_tgt_WordCount,Length_ratio,MaxDepDepth_ratio,MaxDepLength_ratio,FreqRank_ratio,Leven_ratio,WordCount_ratio
    df_test_actual_output_ratio_lr = pd.read_csv(test_files_path_actual_output_ratio[0])
    df_test_actual_output_ratio_fixed_ratio = pd.read_csv(test_files_path_actual_output_ratio[1])
    df_test_actual_output_ratio_fixed_ratio_calibration = pd.read_csv(test_files_path_actual_output_ratio[2])
    df_test_actual_output_ratio_fixed_ratio_cot = pd.read_csv(test_files_path_actual_output_ratio[3])
    df_test_actual_output_ratio_random_ratio = pd.read_csv(test_files_path_actual_output_ratio[4])
    df_test_actual_output_ratio_random_ratio_2 = pd.read_csv(test_files_path_actual_output_ratio[5])

    df_test_actual_output_ratio_lr_gpt_4_preview = pd.read_csv(test_files_path_actual_output_ratio[6])
    df_test_actual_output_ratio_fixed_ratio_gpt_4_preview = pd.read_csv(test_files_path_actual_output_ratio[7])

    df_test_actual_output_ratio_lr_5_examples = pd.read_csv(test_files_path_actual_output_ratio[8])
    df_test_actual_output_ratio_fixed_ratio_5_examples = pd.read_csv(test_files_path_actual_output_ratio[9])

    no_of_rows, no_of_columns = df_test_actual_output_ratio_lr.shape
    print("no_of_rows: %s, no_of_columns:%s, file:%s" % (no_of_rows, no_of_columns, df_test_actual_output_ratio_lr))
    print(df_test_actual_output_ratio_lr.columns)

    df_test_roberta_predicted_ratios = pd.read_csv(test_files_path_actual_output_ratio[12],nrows=no_of_rows)

    for test_file_path_with_predicted_ratio, abs_src_feature, ratio_feature, fixed_ratio in zip(test_files_list_path_with_predicted_ratio,
                                                                                   abs_src_feature_list, ratio_feature_list,
                                                                                   fixed_ratio_list):
        print(f"Reading src_feature: {abs_src_feature}, ratio: {ratio_feature} and LR-predicted ratio file: {test_file_path_with_predicted_ratio}, ")
        # Read LR predicted ratios
        df_test_with_predicted_ratio = pd.read_csv(test_file_path_with_predicted_ratio, nrows=no_of_rows)
        print(df_test_with_predicted_ratio.columns)
        # Sample data
        X_test = np.array(df_test_with_predicted_ratio[abs_src_feature])  # Source sentence feature value
        Y_test = np.array(df_test_with_predicted_ratio[ratio_feature])  #  (gold ref)
        test_lr_predictions_ratio = df_test_with_predicted_ratio["predicted_" + ratio_feature]

        lr_actual_obtained_test_ratio = df_test_actual_output_ratio_lr[ratio_feature]  # (test lr output)
        fixed_ratio_actual_obtained_test_ratio = df_test_actual_output_ratio_fixed_ratio[ratio_feature]  # (test fr output)
        fixed_ratio_actual_obtained_test_ratio_calibration = df_test_actual_output_ratio_fixed_ratio_calibration[ratio_feature]  # (test fr output)
        fixed_ratio_actual_obtained_test_ratio_cot = df_test_actual_output_ratio_fixed_ratio_cot[ratio_feature]  # (test fr output)
        random_ratio_actual_obtained_test_ratio = df_test_actual_output_ratio_random_ratio[ratio_feature]  # (test fr output)
        random_ratio_2_actual_obtained_test_ratio = df_test_actual_output_ratio_random_ratio_2[ratio_feature]  # (test fr output)

        lr_actual_obtained_test_ratio_gpt_4_preview = df_test_actual_output_ratio_lr_gpt_4_preview[ratio_feature]  # (test lr output)
        fixed_ratio_actual_obtained_test_ratio_gpt_4_preview = df_test_actual_output_ratio_fixed_ratio_gpt_4_preview[ratio_feature]  # (test fr output)

        lr_actual_obtained_test_ratio_5_examples = df_test_actual_output_ratio_lr_5_examples[ratio_feature]  # (test lr output)
        fixed_ratio_actual_obtained_test_ratio_5_examples = df_test_actual_output_ratio_fixed_ratio_5_examples[ratio_feature]  # (test fr output)

        roberta_predicted_abs_tgts = df_test_roberta_predicted_ratios["predicted_" + ratio_feature]

        y_fixed_ratio = float(fixed_ratio)
        y_ideal_fixed_ratio = np.round((y_fixed_ratio * X_test), 1) / X_test

        # Visualize the training data and regression line
        # Set a custom color palette with a light color for the histogram
        custom_palette = ["#FDF3F3",  # Light color for the histogram
                          "#F2C0BD",
                          # Add other colors as needed
                          ]
        sns.set_palette(custom_palette)

        # Create the main figure and axes
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Determine the x-axis range from the line plot data
        x_min, x_max = min(X_test), max(X_test)
        print("Determine the x-axis range from the line plot data: x_min = %s, x_max = %s" % (x_min, x_max))
        bins = np.arange(x_min - 1, x_max + 2)  # +2 to include the upper edge for the last bin
        print("bins for histogram: %s" % (bins))

        # # Create a avg line plot
        # sns.lineplot(x=X_test, y=test_lr_predictions_ratio, color='red', label='LR Predicted Ratio',
        #              err_style="band", errorbar=("sd"), estimator='mean')
        # sns.lineplot(x=X_test, y=y_fixed_ratio, color='brown', label='Fixed Ratio (FR)',
        #              err_style="band", errorbar=("sd"), estimator='mean')
        # sns.lineplot(x=X_test, y=Y_test, color='gold', label='Gold reference',
        #              err_style="band", errorbar=("sd"), estimator='mean')
        # sns.lineplot(x=X_test, y=lr_actual_obtained_test_ratio, color='blue', label='GPT-4 ZS obtained ratio for LR',
        #              err_style="band", errorbar=("sd"), estimator='mean')
        # sns.lineplot(x=X_test, y=fixed_ratio_actual_obtained_test_ratio, color='green', label='GPT-4 ZS obtained ratio for FR',
        #              err_style="band", errorbar=("sd"), estimator='mean')

        sns.lineplot(x=X_test, y=test_lr_predictions_ratio, color='red', label='LR Predicted Ratio',
                            estimator='mean')
        # sns.lineplot(x=X_test, y=lr_actual_obtained_test_ratio, color='blue', label='GPT-4 3-examples-FS - LR Predicted Ratio',
        #              estimator='mean')
        sns.lineplot(x=X_test, y=y_fixed_ratio, color='brown', label='Fixed Ratio (FR)',
                     estimator='mean')
        # sns.lineplot(x=X_test, y=fixed_ratio_actual_obtained_test_ratio, color='green', label='GPT-4 3-examples-FS - FR Ratio',
        #                       estimator='mean')
        # sns.lineplot(x=X_test, y=fixed_ratio_actual_obtained_test_ratio_calibration, color='purple', label='GPT-4 3-examples-FS - FR Ratio-calibration',
        #              estimator='mean')
        # sns.lineplot(x=X_test, y=fixed_ratio_actual_obtained_test_ratio_cot, color='orange', label='GPT-4 FS - FR Ratio-CoT',
        #              estimator='mean')
        # sns.lineplot(x=X_test, y=random_ratio_actual_obtained_test_ratio, color='orange', label='GPT-4 FS - Random ratio-1',
        #              estimator='mean')
        # sns.lineplot(x=X_test, y=random_ratio_2_actual_obtained_test_ratio, color='black', label='GPT-4 FS - Random ratio-2',
        #              estimator='mean')


        # sns.lineplot(x=X_test, y=lr_actual_obtained_test_ratio_5_examples, color='blue', label='GPT-4 5-examples-FS - LR Predicted Ratio',
        #              estimator='mean')
        # sns.lineplot(x=X_test, y=fixed_ratio_actual_obtained_test_ratio_5_examples, color='green', label='GPT-4 5-examples-FS - FR Ratio-calibration',
        #              estimator='mean')

        # sns.lineplot(x=X_test, y=lr_actual_obtained_test_ratio_gpt_4_preview, color='orange',
        #              label='GPT-4-preview 30-FS - LR-ratio',
        #              estimator='mean')
        # sns.lineplot(x=X_test, y=fixed_ratio_actual_obtained_test_ratio_gpt_4_preview, color='black',
        #              label='GPT-4-preview 30-FS - Fixed-ratio',
        #              estimator='mean')

        sns.lineplot(x=X_test, y=roberta_predicted_abs_tgts, color='green',
                             label='Roberta predicted ratios',
                             estimator='mean')
        # sns.lineplot(x=X_test, y=Y_test, color='gold', label='Gold reference',
        #              estimator='mean')


        # Create the secondary y-axis
        ax2 = ax1.twinx()

        # Plotting the histogram on ax2 with a label (no log scale for frequencies)
        sns.histplot(X_test, ax=ax2, bins=bins, color='#FDF3F3', kde=False, stat="count",
                     edgecolor='#F2C0BD', label='No. of occurrences') # abs_src_feature

        # Set the same x-axis range for both plots
        ax1.set_xlim(x_min - 1, x_max + 1)
        ax2.set_xlim(x_min - 1, x_max + 1)

        # Bring the lineplot (ax1) to the front
        ax1.set_zorder(ax2.get_zorder() + 1)
        ax1.patch.set_visible(False)  # Make ax1 background transparent

        # Labels, titles, and legend
        # plt.title("%s vs %s" % (ratio_feature, abs_src_feature))
        ax1.set_xlabel(abs_src_feature)
        ax1.set_ylabel(ratio_feature)
        ax2.set_ylabel('No. of ' + abs_src_feature)
        # Combine legends from both axes
        handles, labels = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles + handles2, labels + labels2, loc='best') #upper right
        # ax1.legend(handles, labels, loc='best') #upper right

        fig.savefig(save_plot_path + "/plot_all_" + ratio_feature + "_vs_" + abs_src_feature, dpi=300, bbox_inches='tight')


def plot_all_outputs_abs_tgt_vs_abs_src_feature_list(test_files_list_path_with_predicted_ratio, test_files_path_actual_output_ratio,
                                              feature_list, save_plot_path, fixed_ratio_list):

    print("Plot absolute tgt vs src for features: %s with fixed ratio %s, From: %s,%s, Save:%s" % (feature_list,
                                                                               fixed_ratio_list,
                                                           test_files_list_path_with_predicted_ratio,
                                                           test_files_path_actual_output_ratio, save_plot_path))
    # Read test actual output ratios
    # Line,abs_src_Length,abs_src_MaxDepDepth,abs_src_MaxDepLength,abs_src_FreqRank,abs_src_Leven,abs_src_WordCount,abs_tgt_Length,abs_tgt_MaxDepDepth,abs_tgt_MaxDepLength,abs_tgt_FreqRank,abs_tgt_Leven,abs_tgt_WordCount,Length_ratio,MaxDepDepth_ratio,MaxDepLength_ratio,FreqRank_ratio,Leven_ratio,WordCount_ratio
    df_test_actual_output_ratio_lr = pd.read_csv(test_files_path_actual_output_ratio[0])
    df_test_actual_output_ratio_fixed_ratio = pd.read_csv(test_files_path_actual_output_ratio[1])
    df_test_actual_output_ratio_fixed_ratio_calibration = pd.read_csv(test_files_path_actual_output_ratio[2])
    df_test_actual_output_ratio_fixed_ratio_cot = pd.read_csv(test_files_path_actual_output_ratio[3])
    df_test_actual_output_ratio_random_ratio = pd.read_csv(test_files_path_actual_output_ratio[4])
    df_test_actual_output_ratio_random_ratio_2 = pd.read_csv(test_files_path_actual_output_ratio[5])

    df_test_actual_output_ratio_lr_gpt_4_preview = pd.read_csv(test_files_path_actual_output_ratio[6])
    df_test_actual_output_ratio_fixed_ratio_gpt_4_preview = pd.read_csv(test_files_path_actual_output_ratio[7])

    df_test_actual_output_ratio_lr_5_examples = pd.read_csv(test_files_path_actual_output_ratio[8])
    df_test_actual_output_ratio_fixed_ratio_5_examples = pd.read_csv(test_files_path_actual_output_ratio[9])
    df_test_actual_output_ratio_fixed_ratio_5_examples_calibration = pd.read_csv(test_files_path_actual_output_ratio[10])
    df_test_actual_output_ratio_fixed_ratio_5_examples_calibration_with_3_Features = pd.read_csv(test_files_path_actual_output_ratio[11])


    no_of_rows, no_of_columns = df_test_actual_output_ratio_lr.shape
    print("no_of_rows: %s, no_of_columns:%s, file:%s" % (no_of_rows, no_of_columns, df_test_actual_output_ratio_lr))
    print(df_test_actual_output_ratio_lr.columns)

    df_test_roberta_predicted_ratios = pd.read_csv(test_files_path_actual_output_ratio[12],nrows=no_of_rows)

    for test_file_path_with_predicted_ratio, feature, fixed_ratio in zip(test_files_list_path_with_predicted_ratio,
                                                                                   feature_list,
                                                                                   fixed_ratio_list):
        print(f"Reading feature: {feature}, and LR-predicted ratio file: {test_file_path_with_predicted_ratio}")
        # Read LR predicted ratios
        df_test_with_predicted_ratio = pd.read_csv(test_file_path_with_predicted_ratio, nrows=no_of_rows)
        print(df_test_with_predicted_ratio.columns)
        # Sample data
        X_test = np.array(df_test_with_predicted_ratio["abs_src_" + feature])  # Source sentence feature value
        Y_test = np.array(df_test_with_predicted_ratio["abs_tgt_" + feature])  #  (gold ref)
        test_lr_predictions_ratio = round(df_test_with_predicted_ratio["predicted_" + feature + "_ratio"] * df_test_with_predicted_ratio["abs_src_" + feature])
        y_fixed_ratio = round( float(fixed_ratio) * df_test_with_predicted_ratio["abs_src_" + feature])

        # y_ideal_fixed_ratio = np.round((y_fixed_ratio * X_test), 1) / X_test

        lr_actual_obtained_test_ratio = df_test_actual_output_ratio_lr["abs_tgt_" + feature]  # (test lr output)
        fixed_ratio_actual_obtained_test_ratio = df_test_actual_output_ratio_fixed_ratio["abs_tgt_" + feature]

        # fixed_ratio_actual_obtained_test_ratio_calibration = df_test_actual_output_ratio_fixed_ratio_calibration["abs_tgt_" + feature]
        # fixed_ratio_actual_obtained_test_ratio_cot = df_test_actual_output_ratio_fixed_ratio_cot["abs_tgt_" + feature]
        # random_ratio_actual_obtained_test_ratio = df_test_actual_output_ratio_random_ratio["abs_tgt_" + feature]
        # random_ratio_2_actual_obtained_test_ratio = df_test_actual_output_ratio_random_ratio_2["abs_tgt_" + feature]

        # lr_actual_obtained_test_ratio_gpt_4_preview = df_test_actual_output_ratio_lr_gpt_4_preview["abs_tgt_" + feature]  # (test lr output)
        # fixed_ratio_actual_obtained_test_ratio_gpt_4_preview = df_test_actual_output_ratio_fixed_ratio_gpt_4_preview["abs_tgt_" + feature]  # (test fr output)

        lr_actual_obtained_test_ratio_5_examples = df_test_actual_output_ratio_lr_5_examples[
            "abs_tgt_" + feature]  # (test lr output)
        fixed_ratio_actual_obtained_test_ratio_5_examples = df_test_actual_output_ratio_fixed_ratio_5_examples[
            "abs_tgt_" + feature]  # (test fr output)
        fixed_ratio_actual_obtained_test_ratio_5_examples_calibration = df_test_actual_output_ratio_fixed_ratio_5_examples_calibration[
            "abs_tgt_" + feature]  # (test fr output)
        fixed_ratio_actual_obtained_test_ratio_5_examples_calibration_with_3_features = df_test_actual_output_ratio_fixed_ratio_5_examples_calibration_with_3_Features[
                    "abs_tgt_" + feature]  # (test fr output)

        roberta_predicted_abs_tgts = round(df_test_roberta_predicted_ratios["predicted_" + feature + "_ratio"] * df_test_with_predicted_ratio["abs_src_" + feature])

        # Visualize the training data and regression line
        # Set a custom color palette with a light color for the histogram
        custom_palette = ["#FDF3F3",  # Light color for the histogram
                          "#F2C0BD",
                          # Add other colors as needed
                          ]
        sns.set_palette(custom_palette)

        # Create the main figure and axes
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Determine the x-axis range from the line plot data
        x_min, x_max = min(X_test), max(X_test)
        print("Determine the x-axis range from the line plot data: x_min = %s, x_max = %s" % (x_min, x_max))
        bins = np.arange(x_min - 1, x_max + 2)  # +2 to include the upper edge for the last bin
        print("bins for histogram: %s" % (bins))

        # # Create a avg line plot
        # sns.lineplot(x=X_test, y=test_lr_predictions_ratio, color='red', label='LR Predicted Ratio',
        #              err_style="band", errorbar=("sd"), estimator='mean')
        # sns.lineplot(x=X_test, y=y_fixed_ratio, color='brown', label='Fixed Ratio (FR)',
        #              err_style="band", errorbar=("sd"), estimator='mean')
        # sns.lineplot(x=X_test, y=Y_test, color='gold', label='Gold reference',
        #              err_style="band", errorbar=("sd"), estimator='mean')
        # sns.lineplot(x=X_test, y=lr_actual_obtained_test_ratio, color='blue', label='GPT-4 ZS obtained ratio for LR',
        #              err_style="band", errorbar=("sd"), estimator='mean')
        # sns.lineplot(x=X_test, y=fixed_ratio_actual_obtained_test_ratio, color='green', label='GPT-4 ZS obtained ratio for FR',
        #              err_style="band", errorbar=("sd"), estimator='mean')


        sns.lineplot(x=X_test, y=test_lr_predictions_ratio, color='red', label='LR Predicted Ratio',
                            estimator='mean')
        # sns.lineplot(x=X_test, y=lr_actual_obtained_test_ratio, color='blue', label='GPT-4 FS - LR Predicted Ratio',
        #              estimator='mean')
        sns.lineplot(x=X_test, y=y_fixed_ratio, color='brown', label='Fixed Ratio (FR)',
                     estimator='mean')
        # sns.lineplot(x=X_test, y=fixed_ratio_actual_obtained_test_ratio, color='green', label='GPT-4 FS - FR Ratio',
        #                       estimator='mean')
        # sns.lineplot(x=X_test, y=fixed_ratio_actual_obtained_test_ratio_calibration, color='purple', label='GPT-4 FS - FR Ratio-calibration',
        #              estimator='mean')
        # sns.lineplot(x=X_test, y=fixed_ratio_actual_obtained_test_ratio_cot, color='orange', label='GPT-4 FS - FR Ratio-CoT',
        #              estimator='mean')
        # sns.lineplot(x=X_test, y=random_ratio_actual_obtained_test_ratio, color='orange', label='GPT-4 FS - Random ratio-1',
        #              estimator='mean')
        # sns.lineplot(x=X_test, y=random_ratio_2_actual_obtained_test_ratio, color='black', label='GPT-4 FS - Random ratio-2',
        #              estimator='mean')

        # sns.lineplot(x=X_test, y=lr_actual_obtained_test_ratio_gpt_4_preview, color='orange',
        #              label='GPT-4-preview 30-FS - LR-ratio',
        #              estimator='mean')
        # sns.lineplot(x=X_test, y=fixed_ratio_actual_obtained_test_ratio_gpt_4_preview, color='black',
        #              label='GPT-4-preview 30-FS - Fixed-ratio',
        #              estimator='mean')

        # sns.lineplot(x=X_test, y=lr_actual_obtained_test_ratio_5_examples, color='blue',
        #              label='GPT-4 5-examples-FS - LR Predicted Ratio',
        #              estimator='mean')
        # sns.lineplot(x=X_test, y=fixed_ratio_actual_obtained_test_ratio_5_examples, color='green', label='GPT-4 5-examples-FS - FR Ratio',
        #              estimator='mean')

        # sns.lineplot(x=X_test, y=fixed_ratio_actual_obtained_test_ratio_5_examples_calibration, color='purple',
        #              label='GPT-4 5-examples-FS - FR Ratio-calibration',
        #              estimator='mean')
        # sns.lineplot(x=X_test, y=fixed_ratio_actual_obtained_test_ratio_5_examples_calibration_with_3_features, color='black',
        #                      label='GPT-4 5-examples-FS - FR Ratio-calibration W/O Diffwords',
        #                      estimator='mean')


        sns.lineplot(x=X_test, y=roberta_predicted_abs_tgts, color='green',
                             label='Roberta predicted ratios',
                             estimator='mean')
        sns.lineplot(x=X_test, y=Y_test, color='gold', label='Gold reference',
                     estimator='mean')


        # Create the secondary y-axis
        ax2 = ax1.twinx()

        # Plotting the histogram on ax2 with a label (no log scale for frequencies)
        sns.histplot(X_test, ax=ax2, bins=bins, color='#FDF3F3', kde=False, stat="count",
                     edgecolor='#F2C0BD', label='No. of occurrences') # abs_src_feature

        # Set the same x-axis range for both plots
        ax1.set_xlim(x_min - 1, x_max + 1)
        ax2.set_xlim(x_min - 1, x_max + 1)

        # Bring the lineplot (ax1) to the front
        ax1.set_zorder(ax2.get_zorder() + 1)
        ax1.patch.set_visible(False)  # Make ax1 background transparent

        # Labels, titles, and legend
        # plt.title("%s vs %s" % (ratio_feature, abs_src_feature))
        ax1.set_xlabel("abs_src_" + feature)
        ax1.set_ylabel("abs_tgt_" + feature)
        ax2.set_ylabel('No. of ' + "abs_src_" + feature)
        # Combine legends from both axes
        handles, labels = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles + handles2, labels + labels2, loc='best') #upper right
        # ax1.legend(handles, labels, loc='best') #upper right

        fig.savefig(save_plot_path + "/plot_all_abs_tgt_" + feature + "_vs_abs_src_" + feature, dpi=300, bbox_inches='tight')





def generate_readability_score_plots(test_file_path_with_predicted_ratio, source_sentence_length,
                                     readability_scores_of_test_src, readability_scores_of_test_tgt_gold,
                                     readability_scores_of_test_output_sents, save_plot_path):

    print("Plot Readability results, From: %s,%s,%s Save:%s" % (readability_scores_of_test_src,
                                                                     readability_scores_of_test_tgt_gold,
                                                                     readability_scores_of_test_output_sents,
                                                                     save_plot_path))

    # FRE, CLI, DCR, ARI
    df_test_output_sents = pd.read_csv(readability_scores_of_test_output_sents)
    no_of_rows, no_of_columns = df_test_output_sents.shape
    print(df_test_output_sents.columns)

    df_test_src = pd.read_csv(readability_scores_of_test_src, nrows=no_of_rows)
    print(df_test_src.columns)

    df_test_tgt_gold = pd.read_csv(readability_scores_of_test_tgt_gold, nrows=no_of_rows)
    print(df_test_tgt_gold.columns)

    df_test_with_predicted_ratio = pd.read_csv(test_file_path_with_predicted_ratio, nrows=no_of_rows)
    X_test = np.array(df_test_with_predicted_ratio[source_sentence_length])  # abs_Source_sentence_word_count


    # Visualize the training data and regression line
    # Set a custom color palette with a light color for the histogram
    custom_palette = ["#FDF3F3",  # Light color for the histogram
                      "#F2C0BD",
                      # Add other colors as needed
                      ]

    for score_name in df_test_src.columns: #['FRE', 'CLI', 'DCR', 'ARI']
        sns.set_palette(custom_palette)

        # Create the main figure and axes
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Determine the x-axis range from the line plot data
        x_min, x_max = min(X_test), max(X_test)
        print("Determine the x-axis range from the line plot data: x_min = %s, x_max = %s" % (x_min, x_max))
        bins = np.arange(x_min - 1, x_max + 2)  # +2 to include the upper edge for the last bin
        print("bins for histogram: %s" % (bins))

        # Create a linear regression plot
        sns.lineplot(x=X_test, y=np.array(df_test_src[score_name]), color='red', label='Input (src)',
                     err_style="band", errorbar=("sd"), estimator='mean')
        sns.lineplot(x=X_test, y=np.array(df_test_tgt_gold[score_name]), color='gold', label='Gold reference (tgt)',
                     err_style="band", errorbar=("sd"), estimator='mean')
        sns.lineplot(x=X_test, y=np.array(df_test_output_sents[score_name]), color='blue', label='Obtained (tgt)',
                     err_style="band", errorbar=("sd"), estimator='mean')

        # Create the secondary y-axis
        ax2 = ax1.twinx()

        # Plotting the histogram on ax2 with a label (no log scale for frequencies)
        sns.histplot(X_test, ax=ax2, bins=bins, color='#FDF3F3', kde=False, stat="count",
                     edgecolor='#F2C0BD', label='No. of ' + source_sentence_length)

        # sns.histplot(X_test, ax=ax2, bins=bins, color='tab:green', kde=False, stat="count",
        #              edgecolor='tab:green', label='No. of ' + source_sentence_length)

        # Set the same x-axis range for both plots
        ax1.set_xlim(x_min - 1, x_max + 1)
        ax2.set_xlim(x_min - 1, x_max + 1)

        # Bring the lineplot (ax1) to the front
        ax1.set_zorder(ax2.get_zorder() + 1)
        ax1.patch.set_visible(False)  # Make ax1 background transparent

        # Labels, titles, and legend
        plt.title("Readability results: %s vs %s" % (score_name, source_sentence_length))
        ax1.set_xlabel(source_sentence_length)
        ax1.set_ylabel(score_name)
        ax2.set_ylabel('No. of ' + source_sentence_length)
        # Combine legends from both axes
        handles, labels = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles + handles2, labels + labels2, loc='upper right')

        fig.savefig(save_plot_path + "/readability_score_" + score_name + "_plot", dpi=300, bbox_inches='tight')


def plot_abs_actual_vs_abs_requested_test(requested_tgt_ideal_path, obtained_tgt_path,
                                              source_sentence_length, length_ratio, save_plot_path):

    # TESTING
    # number_of_examples_found, 0, requested_dependency_depth, -1.0, requested_dependency_length, -1.0, requested_frequency, -1, requested_length, -1.0, requested_levenshtein, -1.0, requested_word_count, 0.6679739680185761, src_feature_value, 28, ideal_tgt_feature_value, 19,
    df_tgt_requested = pd.read_csv(requested_tgt_ideal_path, header=None)
    print(df_tgt_requested.columns)
    print("Plot TEST results %s vs %s, From: %s,%s, Save:%s" % (length_ratio, source_sentence_length,
                                                              requested_tgt_ideal_path,
                                                              obtained_tgt_path, save_plot_path))

    # word_count_source, 28.0, word_count_target, 13.0, word_count_ratio, 0.46
    df_tgt_obtained = pd.read_csv(obtained_tgt_path, header=None) #
    print(df_tgt_obtained.columns)

    # Plot Obtained length vs requested length
    X_test = np.array(df_tgt_requested[17])  # requested ideal target sentence word count
    Y_test = np.array(df_tgt_obtained[3])  # obtained actual output sentence word count

    print(X_test)
    print("!!!")
    print(Y_test)

    # Set a custom color palette with a light color for the histogram
    custom_palette = ["#FDF3F3",  # Light color for the histogram
                      "#F2C0BD",
                      # Add other colors as needed
                      ]
    sns.set_palette(custom_palette)

    # Create the main figure and axes
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Determine the x-axis range from the line plot data
    x_min, x_max = min(X_test), max(X_test)
    print("Determine the x-axis range from the line plot data: x_min = %s, x_max = %s" % (x_min, x_max))
    bins = np.arange(x_min - 1, x_max + 2)  # +2 to include the upper edge for the last bin
    print("bins for histogram: %s" % (bins))

    # Create a linear regression plot
    sns.regplot(x=X_test, y=X_test, ci=None, color='black', label='Ideal (tgt)', line_kws={'linewidth': 1},
                scatter_kws={'s': 10})
    sns.regplot(x=X_test, y=Y_test, ci=None, color='blue', label='Obtained (tgt)', line_kws={'linewidth': 1},
                scatter_kws={'s': 10})
    sns.regplot(x=X_test, y=np.array(df_tgt_obtained[1])  , ci=None, color='green', label='Input (src)', line_kws={'linewidth': 1},
                scatter_kws={'s': 10})

    # Create the secondary y-axis
    ax2 = ax1.twinx()

    # # Plotting the histogram on ax2 with a label (no log scale for frequencies)
    # sns.histplot(X_test, ax=ax2, bins=bins, color='#FDF3F3', kde=False, stat="count",
    #              edgecolor='#F2C0BD', label='No. of ' + source_sentence_length)

    # Set the same x-axis range for both plots
    ax1.set_xlim(x_min - 1, x_max + 1)
    ax2.set_xlim(x_min - 1, x_max + 1)

    # Bring the lineplot (ax1) to the front
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)  # Make ax1 background transparent

    # Labels, titles, and legend
    plt.title("TEST: %s vs requested %s" % (length_ratio, source_sentence_length))
    ax1.set_xlabel("Requested word count")
    ax1.set_ylabel("Obtained word count")
    # ax2.set_ylabel('No. of ' + )
    # Combine legends from both axes
    handles, labels = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles + handles2, labels + labels2, loc='upper right')

    fig.savefig(save_plot_path + "/test_turk_50_obtain_vs_requested", dpi=300, bbox_inches='tight')

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file_path_with_predicted_ratio", required=False, help="")
    parser.add_argument("--fixed_ratio_value", required=False, help="")
    parser.add_argument("--test_file_path_actual_output_ratio", required=False, help="")
    parser.add_argument("--source_sentence_feature", required=False, help="")
    parser.add_argument("--target_ratio_feature", required=False, help="")
    parser.add_argument("--save_plot_path", required=False, help="")

    parser.add_argument("--readability_scores_of_test_src", required=False, help="")
    parser.add_argument("--readability_scores_of_test_tgt_gold", required=False, help="")
    parser.add_argument("--readability_scores_of_test_output_sents", required=False, help="")

    # for requested vs obtained plot:
    parser.add_argument("--requested_feature_details", required=False, help="")
    parser.add_argument("--obtained_feature_details", required=False, help="")
    parser.add_argument("--requested_feature_value", required=False, help="")
    parser.add_argument("--obtained_feature_value", required=False, help="")


    parser.add_argument("--column_number", required=False, help="")

    parser.add_argument("--ratio_file", required=False, help="")
    parser.add_argument("--abs_src_feature_list", required=False, help="")
    parser.add_argument("--ratio_feature_list", required=False, help="")
    parser.add_argument("--feature_list", required=False, help="")
    # parser.add_argument("--save_plot_path", required=False, help="")
    parser.add_argument("--do_filtering", action="store_true", required=False, help="Request to conduct training")
    parser.add_argument("--x_min_value", required=False, help="")
    parser.add_argument("--x_max_value", required=False, help="")
    parser.add_argument("--y_max_value", required=False, help="")


    args = vars(parser.parse_args())
    if args["fixed_ratio_value"] is not None:

        # plot_all_outputs_abs_tgt_vs_abs_src_feature_list(test_files_list_path_with_predicted_ratio=args["test_file_path_with_predicted_ratio"].split(','),
        #                                                  test_files_path_actual_output_ratio=args["test_file_path_actual_output_ratio"].split(','),
        #                                                  feature_list=args["feature_list"].split(','),
        #                                                  save_plot_path=args["save_plot_path"],
        #                                                  fixed_ratio_list=args["fixed_ratio_value"].split(','))


        # plot_all_ratios_vs_abs_src_feature_list(test_files_list_path_with_predicted_ratio=args["test_file_path_with_predicted_ratio"].split(','),
        #                                         test_files_path_actual_output_ratio=args["test_file_path_actual_output_ratio"].split(','),
        #                                         abs_src_feature_list=args["source_sentence_feature"].split(','),
        #                                         ratio_feature_list=args["target_ratio_feature"].split(','),
        #                                         save_plot_path=args["save_plot_path"],
        #                                         fixed_ratio_list=args["fixed_ratio_value"].split(','))

        # # # TESTING with Turk 50 sentences
        plot_gold_predicted_actual_values_of_test_with_fixed_ratio(
            args["test_file_path_with_predicted_ratio"].split(','),
            args["test_file_path_actual_output_ratio"],
            args["abs_src_feature_list"].split(','),
            args["ratio_feature_list"].split(','),
            args["save_plot_path"],
            args["fixed_ratio_value"].split(',')
            )
    else:
        if args["test_file_path_with_predicted_ratio"] is not None:
            # # for single feature
            # plot_gold_predicted_actual_values_of_test(
            #     args["test_file_path_with_predicted_ratio"],
            #     args["test_file_path_actual_output_ratio"],
            #     args["source_sentence_feature"],
            #     args["target_ratio_feature"],
            #     args["save_plot_path"]
            #     )

            # for multiple feature
            plot_gold_predicted_actual_values_of_test_with_feature_list(args["test_file_path_with_predicted_ratio"].split(','),
                                                                    args["test_file_path_actual_output_ratio"],
                                                                    args["abs_src_feature_list"].split(','),
                                                                    args["ratio_feature_list"].split(','),
                                                                    args["save_plot_path"])

            # generate_readability_score_plots(
            #     args["test_file_path_with_predicted_ratio"],
            #     args["source_sentence_feature"],
            #     args["readability_scores_of_test_src"],
            #     args["readability_scores_of_test_tgt_gold"],
            #     args["readability_scores_of_test_output_sents"],
            #     args["save_plot_path"]
            # )

    # ************************************************************************************************************
   # Filter ratio and generate plots and csv files.
   #  parser = argparse.ArgumentParser()
   #  parser.add_argument("--ratio_file", required=False, help="")
   #  parser.add_argument("--abs_src_feature_list", required=False, help="")
   #  parser.add_argument("--ratio_feature_list", required=False, help="")
   #  parser.add_argument("--save_plot_path", required=False, help="")
   #  parser.add_argument("--do_filtering", action="store_true", required=False, help="Request to conduct training")
   #  parser.add_argument("--x_min_value", required=False, help="")
   #  parser.add_argument("--x_max_value", required=False, help="")
   #  parser.add_argument("--y_max_value", required=False, help="")
   #  args = vars(parser.parse_args())

    if args["ratio_file"] is not None:
        x_min_values=[]
        x_max_values=[]
        y_max_value = None
        if args["do_filtering"]:
            # Split the string into a list and then convert each element to an integer
            x_min_values = [int(x) for x in args["x_min_value"].split(',')]
            # Do the same for x_max_value if necessary
            x_max_values = [int(x) for x in args["x_max_value"].split(',')]
            y_max_value = int(args["y_max_value"])

        generate_lineplot_and_histogram(args["ratio_file"],
                                        args["abs_src_feature_list"].split(','),
                                        args["ratio_feature_list"].split(','),
                                        args["save_plot_path"], args["do_filtering"],
                                        x_min_values,
                                        x_max_values,
                                        y_max_value
                                        )

    # # Filter ratio WikiLarge log-scale plots
    # abs_src_feature_list = ["abs_src_MaxDepDepth", "abs_src_MaxDepLength", "abs_src_FreqRank", "abs_src_WordCount"]
    # ratio_feature_list = ["MaxDepDepth_ratio", "MaxDepLength_ratio", "FreqRank_ratio", "WordCount_ratio"]
    # # Order: maxdepDepth, maxdepthLength, freqrank, wordcount
    # x_min_value = [2, 2, 7, 6]
    # x_max_value = [11, 20, 12, 40]

    # generate_lineplot_and_histogram("data_auxiliary/en/feature_distribution_analyse/WikiLarge_V2/wikilarge_all_v2/tgt/ratio_stats.csv",
    #                                 abs_src_feature_list, ratio_feature_list,
    #                                 "data_auxiliary/en/feature_distribution_analyse/WikiLarge_V2/wikilarge_all_v2/tgt/3_filter_ratio_and_abs_src_values",
    #                                 is_filtering=True, x_min_value_list=x_min_value, x_max_value_list=x_max_value )

    # generate_lineplot_and_histogram("data_auxiliary/en/feature_distribution_analyse/WikiLarge_V2/wikilarge_all_v2/tgt/ratio_stats.csv",
    #                                 abs_src_feature_list, ratio_feature_list,
    #                                 "data_auxiliary/en/feature_distribution_analyse/WikiLarge_V2/wikilarge_all_v2/tgt/2_filter_ratio_above_2",
    #                                 is_filtering=True)

    # generate_lineplot_and_histogram("data_auxiliary/en/feature_distribution_analyse/WikiLarge_V2/wikilarge_all_v2/tgt/ratio_stats.csv",
    #                                 abs_src_feature_list, ratio_feature_list,
    #                                 "data_auxiliary/en/feature_distribution_analyse/WikiLarge_V2/wikilarge_all_v2/tgt/1_ratio_vs_abs",
    #                                 is_filtering=False)




