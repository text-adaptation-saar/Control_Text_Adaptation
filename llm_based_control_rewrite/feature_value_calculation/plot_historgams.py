import argparse
import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# pip install seaborn matplotlib
# pip install statsmodels

def generate_lineplot_and_histogram(file_path, abs_src_feature_list, abs_tgt_feature_list, ratio_feature_list, save_plot_path):

    print(file_path)

    if not os.path.exists(save_plot_path):
        os.makedirs(save_plot_path)

    df = pd.read_csv(file_path)
    print("generate_lineplot_and_histogram: %s vs %s,%s, From: %s, Save:%s\n" % (ratio_feature_list,
                                                                                 abs_src_feature_list,abs_tgt_feature_list,
                                                                                 file_path, save_plot_path))

    generate_plots(df, abs_src_feature_list, abs_tgt_feature_list, ratio_feature_list, save_plot_path)


def generate_plots(df, abs_src_feature_list, abs_tgt_feature_list, ratio_feature_list, save_plot_path):

    print("\nGenerate histogram plots! Save:%s" % (save_plot_path))

    for abs_src_feature,abs_tgt_feature, ratio_feature in zip(abs_src_feature_list, abs_tgt_feature_list, ratio_feature_list):
        print("abs_src_feature:%s\tabs_tgt_feature:%s\t ratio_feature:%s " % (abs_src_feature, abs_tgt_feature, ratio_feature))
        # Create the main figure and axes
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Determine the x-axis range from the line plot data
        x_min, x_max = min(df[abs_src_feature].round(0)), max(df[abs_src_feature].round(0))
        print("Determine the x-axis range from the line plot data: x_min = %s, x_max = %s" % (x_min, x_max))
        bins = np.arange(x_min - 1, x_max + 2)  # +2 to include the upper edge for the last bin
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
        ax1.set_xlabel(abs_src_feature.replace("src_", ""))
        ax1.set_ylabel(ratio_feature)
        ax2.set_ylabel('No. of ' + abs_src_feature.replace("src_", ""))
        # Combine legends from both axes
        handles, labels = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles + handles2, labels + labels2, loc='upper right')

        fig.savefig(save_plot_path + "/" + ratio_feature + "_vs_" + abs_src_feature, dpi=300, bbox_inches='tight')


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratio_file", required=False, help="")
    parser.add_argument("--abs_src_feature_list", required=False, help="")
    parser.add_argument("--abs_tgt_feature_list", required=False, help="")
    parser.add_argument("--ratio_feature_list", required=False, help="")
    parser.add_argument("--save_plot_path", required=False, help="")
    args = vars(parser.parse_args())

    if args["ratio_file"] is not None:
        generate_lineplot_and_histogram(args["ratio_file"],
                                        args["abs_src_feature_list"].split(','),
                                        args["abs_tgt_feature_list"].split(','),
                                        args["ratio_feature_list"].split(','),
                                        args["save_plot_path"]
                                        )



