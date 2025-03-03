#
#
# We didn't use this code, we train Catboost Regression directly from: "ControlTS_T5" repo  (paper: Controlling Pre-trained Language Models for Grade-Specific Text Simplification.)
#
#
import argparse
import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score
from catboost import CatBoostRegressor
# conda install catboost

# **********************************************************************************************************************
def generate_catboost_regression_train(train_file_path, eval_file, abs_src_feature, ratio_feature, save_path):

    df = pd.read_csv(train_file_path)
    df_eval = pd.read_csv(eval_file)
    print(df.columns)
    print("%s vs %s, From: %s, Save:%s" % (ratio_feature, abs_src_feature, train_file_path, save_path))

    # Sample data
    X = np.array(df[abs_src_feature])  # Source sentence word count
    Y = np.array(df[ratio_feature])  # Word count ratio
    print(f"X.shape: {X.shape}, Y.shape: {Y.shape}")

    # Reshape X to a 2D array (required for scikit-learn)
    X = X.reshape(-1, 1)
    print(f"X.shape: {X.shape}, Y.shape: {Y.shape}")

    model = CatBoostRegressor(iterations=1000, depth=6, learning_rate=0.1)

    # Making predictions and evaluating the model
    x_eval = np.array(df_eval[abs_src_feature])
    x_eval = x_eval.reshape(-1, 1)
    y_eval = np.array(df_eval[ratio_feature])
    print(f"x_eval.shape: {x_eval.shape}, y_eval.shape: {y_eval.shape}")
    # Fit the model to the training data
    model.fit(X, Y,eval_set=(x_eval, y_eval), verbose=50)

    # Coefficients
    dict = {}
    #Evaluation metrics (MSE and R²)
    y_pred = model.predict(x_eval)
    # .to_numpy().reshape(-1, 1)
    print(f"x_eval.shape: {x_eval.shape}, y_eval.shape: {y_eval.shape}, y_pred.shape: {y_pred.shape}")
    dict["mse"] = mean_squared_error(y_eval, y_pred)
    dict["r_squared"] = r2_score(y_eval, y_pred)

    # Save the model using joblib
    model_filename = f"{save_path}/catboost_regressor_{ratio_feature}.cbm"
    model.save_model(model_filename)

    # save_plot_path = save_path + "/train_catboost_regressor_with_" + ratio_feature + "_vs_" + abs_src_feature
    # generate_plots(df, abs_src_feature, ratio_feature, model.predict(X), save_plot_path, dict )
    # save_plot_path = save_path + "/eval_catboost_regressor_with_" + ratio_feature + "_vs_" + abs_src_feature
    # generate_plots(df_eval, abs_src_feature, ratio_feature, y_pred, save_plot_path, dict)

    # Print model information and evaluation metrics
    print(f'Mean Squared Error (Eval): {dict["mse"]:.2f}')
    print(f'R-squared (Eval): {dict["r_squared"]:.2f}')


def generate_plots(df, abs_src_feature, ratio_feature, model_predicted , save_plot_path,
                   dict_of_regression_model_variables, x_sorted_values=None):
    import matplotlib as mpl
    mpl.rcParams['agg.path.chunksize'] = 10000  # You can adjust this number as needed

    print("abs_src_feature:%s and ratio_feature:%s " % (abs_src_feature, ratio_feature))
    # Create the main figure and axes
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Determine the x-axis range from the line plot data
    x_min, x_max = min(df[abs_src_feature].round(0)), max(df[abs_src_feature].round(0))
    print("Determine the x-axis range from the line plot data: x_min = %s, x_max = %s" % (x_min, x_max))
    bins = np.arange(x_min - 1, x_max + 2)  # +2 to include the upper edge for the last bin
    print("bins for histogram: %s" % (bins))

    # Plotting the line graph on ax1 with a label
    sns.lineplot(x=df[abs_src_feature], y=df[ratio_feature], ax=ax1, color='tab:blue', linewidth=1.0, label="Ratios on training data") #ratio_feature
    # sns.regplot(x=df[abs_src_feature], y=df[ratio_feature], scatter=False, ci=None, color='tab:red', order=2)

    # plot LR model line
    plt.plot(df[abs_src_feature], model_predicted, color='red', linewidth=2,
         label='CatBoost  Regression')
    # Create the secondary y-axis
    ax2 = ax1.twinx()

    # Plotting the histogram on ax2 with a label (no log scale for frequencies)
    sns.histplot(df[abs_src_feature], ax=ax2, bins=bins, color='#FDF3F3', kde=False, stat="count",
                 edgecolor='#F2C0BD', label='No. of occurrences') # + abs_src_feature

    # # Annotate each count vertically inside each histogram block
    # for p in ax2.patches:
    #     ax2.annotate(f'{int(p.get_height())}',
    #                  (p.get_x() + p.get_width() / 2., p.get_height() / 2),
    #                  ha='center', va='center', fontsize=5, color='black', rotation='vertical')

    # Set the same x-axis range for both plots
    ax1.set_xlim(x_min - 1, x_max + 1)
    ax2.set_xlim(x_min -1, x_max + 1)

    # Bring the lineplot (ax1) to the front
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)  # Make ax1 background transparent

    # Labels, titles, and legend
    ax1.set_title("Regression of %s vs %s" % (ratio_feature, abs_src_feature))
    ax1.set_xlabel(abs_src_feature)
    ax1.set_ylabel(ratio_feature)
    ax2.set_ylabel('No. of ' + abs_src_feature)
    # Combine legends from both axes
    handles, labels = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles + handles2, labels + labels2, loc='upper right')

    fig.savefig(save_plot_path, dpi=300, bbox_inches='tight')


def catboost_regression_test(model_filename, test_file_path, abs_src_feature, ratio_feature, save_plot_path, test_or_valid):

    # TESTING
    df_test = pd.read_csv(test_file_path)
    print(df_test.columns)
    print(
        "%s - %s vs %s, From: %s, Save:%s" % (test_or_valid, ratio_feature, abs_src_feature, test_file_path, save_plot_path))

    # Sample data
    X_test = np.array(df_test[abs_src_feature])  # Source sentence word count
    print(X_test)
    Y_test = np.array(df_test[ratio_feature])  # gold Word count ratio

    # Reshape X to a 2D array (required for scikit-learn)
    X_test = X_test.reshape(-1, 1)

    # Load the model from the saved file
    loaded_model = CatBoostRegressor()
    loaded_model.load_model(model_filename)

    # Test the loaded model on the test data
    test_predictions = loaded_model.predict(X_test)
    # rounded_test_predictions = np.round(test_predictions, 2) #ratio are given in 2 decimal values.

    # Reshape to default
    X_test = X_test.flatten()

    # Calculate evaluation metrics
    mse_test = mean_squared_error(Y_test, test_predictions)
    rmse_test = np.sqrt(mse_test)  # Calculate RMSE
    r_squared_test = r2_score(Y_test, test_predictions)

    # Calculate Correlation
    # correlation_matrix = np.corrcoef(Y_test, test_predictions)
    # correlation = correlation_matrix[0, 1]  # Pearson's correlation
    correlation, p_value = pearsonr(Y_test, test_predictions)

    # Coefficients
    # slope = loaded_model.coef_[0]
    # intercept = loaded_model.intercept_

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
    sns.lineplot(x=X_test, y=test_predictions, color='red', label='Predicted',
                 err_style="band", errorbar=("sd"), estimator='mean')
    sns.lineplot(x=X_test, y=Y_test, color='gold', label='Gold reference',
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
    plt.title("%s: CatBoostRegressor of %s vs %s" % (test_or_valid, ratio_feature, abs_src_feature))
    ax1.set_xlabel(abs_src_feature)
    ax1.set_ylabel(ratio_feature)
    ax2.set_ylabel('No. of ' + abs_src_feature)
    # Combine legends from both axes
    handles, labels = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles + handles2, labels + labels2, loc='upper right')

    ax1.text(max(X_test) + 3, max(Y_test) - 0.1,
             f'MSE: {mse_test:.3f}\nRMSE: {rmse_test:.3f}\nR-squared: {r_squared_test:.3f}\n'
             f'Pearson Correlation: {correlation:.3f}')
    fig.savefig(save_plot_path + "/LR_predicted_ratio_vs_abs_plot", dpi=300, bbox_inches='tight')

    # Print evaluation metrics for the test data
    print(f'Mean Squared Error ({test_or_valid}): {mse_test:.3f}')
    print(f'Root Mean Squared Error ({test_or_valid}): {rmse_test:.3f}')
    print(f'R-squared ({test_or_valid}): {r_squared_test:.3f}')
    print(f'Pearson Correlation ({test_or_valid}): {correlation:.3f}')
    print(f'Pearson p_value ({test_or_valid}): {p_value:.3f}')

    # Save predicted ratios.
    new_column_name = "predicted_" + ratio_feature
    df_test[new_column_name] = test_predictions
    # calculate expected/ideal abs_tgt value and Save it.
    feature_name = ratio_feature.replace("_ratio", "")
    new_column_name_abs_tgt = "predicted_abs_tgt_" + feature_name
    predicted_abs_tgt = np.round(test_predictions * X_test)
    df_test[new_column_name_abs_tgt] = predicted_abs_tgt
    csv_file = save_plot_path + "/ratio_stats_with_LR_predicted_ratio.csv"
    df_test.to_csv(csv_file, index=False)  # Set index=False to exclude the index column in the CSV file

    # Create a DataFrame with the scores
    scores_df = pd.DataFrame([{
        'MSE': mse_test,
        'RMSE': rmse_test,
        'R-squared': r_squared_test,
        'Pearson Correlation': correlation,
        'Pearson p_value': p_value,
        'feature_name': ratio_feature,
        'predicted_ratio_file': csv_file
    }])

    # Extract the directory path from the file path
    directory_path = os.path.dirname(csv_file)
    scores_file_path = os.path.join(directory_path, 'evaluation_metrics.csv')
    scores_df.to_csv(scores_file_path, index=False)


def read_text_file(file_path):
    # Read the text file and create a DataFrame
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        # Stripping newline characters from each line
        lines = [line.strip() for line in lines]

    return pd.DataFrame(lines, columns=['text'])

def generate_catboost_regression_train_multi(train_file_path, eval_file, abs_src_feature_list, ratio_feature_list, save_path):

    df = pd.read_csv(train_file_path)
    df_eval = pd.read_csv(eval_file)
    print(df.columns)
    print("%s vs %s, From: %s, Save:%s" % (ratio_feature_list, abs_src_feature_list, train_file_path, save_path))

    # Sample data
    X = df[abs_src_feature_list]  # Source sentence word count
    Y = df[ratio_feature_list]  # Word count ratio
    print(f"X.shape: {X.shape}, Y.shape: {Y.shape}")

    # evaluating
    x_eval = df_eval[abs_src_feature_list]
    y_eval = df_eval[ratio_feature_list]
    print(f"x_eval.shape: {x_eval.shape}, y_eval.shape: {y_eval.shape}")

    # Initialize and train the multi-output model
    multi_model = CatBoostRegressor(iterations=1000, depth=6, learning_rate=0.1, loss_function='MultiRMSE')
    multi_model.fit(X, Y, eval_set=(x_eval, y_eval), verbose=50)

    # Coefficients
    dict = {}
    # Evaluation metrics (MSE and R²)
    y_pred = multi_model.predict(x_eval)
    # .to_numpy().reshape(-1, 1)
    print(f"x_eval.shape: {x_eval.shape}, y_eval.shape: {y_eval.shape}, y_pred.shape: {y_pred.shape}")
    dict["mse"] = mean_squared_error(y_eval, y_pred)
    dict["r_squared"] = r2_score(y_eval, y_pred)

    # Save the model using joblib
    model_filename = f"{save_path}/catboost_regressor_{ratio_feature_list}.cbm"
    multi_model.save_model(model_filename)

    # save_plot_path = save_path + "/train_catboost_regressor_with"
    # generate_plots_multi(df, abs_src_feature_list, ratio_feature_list, multi_model.predict(X), save_plot_path, dict)
    # save_plot_path = save_path + "/eval_catboost_regressor_with"
    # generate_plots_multi(df_eval, abs_src_feature_list, ratio_feature_list, y_pred, save_plot_path, dict)

    # Print model information and evaluation metrics
    print(f'Mean Squared Error (Eval): {dict["mse"]:.2f}')
    print(f'R-squared (Eval): {dict["r_squared"]:.2f}')

def catboost_regression_test_multi(model_filename, test_file_path, abs_src_feature_list, ratio_feature_list, save_plot_path, test_or_valid,
                                   grade_level=None):
    # TESTING
    df_test = pd.read_csv(test_file_path)
    print(df_test.columns)
    print(
        "%s - %s vs %s, From: %s, Save:%s" % (
        test_or_valid, ratio_feature_list, abs_src_feature_list, test_file_path, save_plot_path))

    # Sample data
    X_test = df_test[abs_src_feature_list]  # Source sentence word count
    if grade_level is not None:
        X_test['tgt_fkgl'] = grade_level  # Adds a new column 'grade_level' with the given value across all rows
    print(X_test)
    Y_test = df_test[ratio_feature_list]  # gold Word count ratio
    print(f"X_test.shape: {X_test.shape}, Y_test.shape: {Y_test.shape}")

    # Load the model from the saved file
    loaded_model = CatBoostRegressor()
    loaded_model.load_model(model_filename)

    # Test the loaded model on the test data
    test_predictions = loaded_model.predict(X_test)
    test_predictions_df = pd.DataFrame(test_predictions, columns=Y_test.columns)
    print(f"X_test.shape: {X_test.shape}, Y_test.shape: {Y_test.shape}, test_predictions.shape: {test_predictions.shape}")
    # rounded_test_predictions = np.round(test_predictions, 2) #ratio are given in 2 decimal values.

    # # Calculate evaluation metrics
    # multi_mse_test = mean_squared_error(Y_test, test_predictions)
    # multi_rmse_test = np.sqrt(multi_mse_test)  # Calculate RMSE
    # multi_r_squared_test = r2_score(Y_test, test_predictions)
    # multi_correlation, multi_p_value = pearsonr(Y_test, test_predictions)

    generate_plots_for_testing(X_test, Y_test, abs_src_feature_list, df_test, ratio_feature_list,
                               save_plot_path, test_or_valid, test_predictions_df)


def generate_plots_for_testing(X_test, Y_test, abs_src_feature_list, df_test, ratio_feature_list,
                               save_plot_path, test_or_valid, test_predictions_df):

    scores_df = pd.DataFrame(columns=['MSE', 'RMSE', 'R-squared', 'Pearson Correlation', 'Pearson Correlation p_value',
                                      'Feature Name', 'predicted_ratio_file'])
    csv_file = save_plot_path + "/ratio_stats_with_LR_predicted_ratio.csv"

    for abs_src_feature, ratio_feature in zip(abs_src_feature_list, ratio_feature_list):
        # Calculate evaluation metrics
        print(f"Single: Y_test[ratio_feature].shape: {(Y_test[ratio_feature]).shape}, "
              f"test_predictions_df[ratio_feature].shape: {test_predictions_df[ratio_feature].shape} ")
        mse_test = mean_squared_error(Y_test[ratio_feature], test_predictions_df[ratio_feature])
        rmse_test = np.sqrt(mse_test)  # Calculate RMSE
        r_squared_test = r2_score(Y_test[ratio_feature], test_predictions_df[ratio_feature])
        correlation, p_value = pearsonr(Y_test[ratio_feature], test_predictions_df[ratio_feature])

        # Set a custom color palette with a light color for the histogram
        custom_palette = ["#FDF3F3",  # Light color for the histogram
                          "#F2C0BD",
                          # Add other colors as needed
                          ]
        sns.set_palette(custom_palette)

        # Create the main figure and axes
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Determine the x-axis range from the line plot data
        x_min, x_max = min(X_test[abs_src_feature]), max(X_test[abs_src_feature])
        print("Determine the x-axis range from the line plot data: x_min = %s, x_max = %s" % (x_min, x_max))
        bins = np.arange(x_min - 1, x_max + 2)  # +2 to include the upper edge for the last bin
        print("bins for histogram: %s" % (bins))

        # Create a linear regression plot
        sns.lineplot(x=X_test[abs_src_feature], y=test_predictions_df[ratio_feature], color='red', label='Predicted',
                     err_style="band", errorbar=("sd"), estimator='mean')
        sns.lineplot(x=X_test[abs_src_feature], y=Y_test[ratio_feature], color='gold', label='Gold reference',
                     err_style="band", errorbar=("sd"), estimator='mean')

        # Create the secondary y-axis
        ax2 = ax1.twinx()

        # Plotting the histogram on ax2 with a label (no log scale for frequencies)
        sns.histplot(X_test[abs_src_feature], ax=ax2, bins=bins, color='#FDF3F3', kde=False, stat="count",
                     edgecolor='#F2C0BD', label='No. of ' + abs_src_feature)

        # Set the same x-axis range for both plots
        ax1.set_xlim(x_min - 1, x_max + 1)
        ax2.set_xlim(x_min - 1, x_max + 1)

        # Bring the lineplot (ax1) to the front
        ax1.set_zorder(ax2.get_zorder() + 1)
        ax1.patch.set_visible(False)  # Make ax1 background transparent

        # Labels, titles, and legend
        plt.title("%s: CatBoostRegressor of %s vs %s" % (test_or_valid, ratio_feature, abs_src_feature))
        ax1.set_xlabel(abs_src_feature)
        ax1.set_ylabel(ratio_feature)
        ax2.set_ylabel('No. of ' + abs_src_feature)
        # Combine legends from both axes
        handles, labels = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles + handles2, labels + labels2, loc='upper right')

        ax1.text(max(X_test[abs_src_feature]) + 3, max(Y_test[ratio_feature]) - 0.1,
                 f'MSE: {mse_test:.3f}\nRMSE: {rmse_test:.3f}\nR-squared: {r_squared_test:.3f}\n'
                 f'Pearson Correlation: {correlation:.3f}')
        fig.savefig(f"{save_plot_path}/LR_predicted_{ratio_feature}_vs_{abs_src_feature}", dpi=300, bbox_inches='tight')

        # Print evaluation metrics for the test data
        print(f'Mean Squared Error ({test_or_valid}): {mse_test:.3f}')
        print(f'Root Mean Squared Error ({test_or_valid}): {rmse_test:.3f}')
        print(f'R-squared ({test_or_valid}): {r_squared_test:.3f}')
        print(f'Pearson Correlation ({test_or_valid}): {correlation:.3f}')
        print(f'Pearson p_value ({test_or_valid}): {p_value:.3f}')

        # Save predicted ratios.
        new_column_name = "predicted_" + ratio_feature
        df_test[new_column_name] = test_predictions_df[ratio_feature]
        # calculate expected/ideal abs_tgt value and Save it.
        feature_name = ratio_feature.replace("_ratio", "")
        new_column_name_abs_tgt = "predicted_abs_tgt_" + feature_name
        predicted_abs_tgt = np.round(test_predictions_df[ratio_feature] * X_test[abs_src_feature])
        df_test[new_column_name_abs_tgt] = predicted_abs_tgt

        new_row = pd.DataFrame({
            'MSE': [mse_test],
            'RMSE': [rmse_test],
            'R-squared': [r_squared_test],
            'Pearson Correlation': [correlation],
            'Pearson Correlation p_value': [p_value],
            'Feature Name': [ratio_feature],
            'predicted_ratio_file': [csv_file]
        })

        scores_df = pd.concat([scores_df, new_row], ignore_index=True)


    df_test.to_csv(csv_file, index=False)  # Set index=False to exclude the index column in the CSV file
    # Extract the directory path from the file path
    directory_path = os.path.dirname(csv_file)
    scores_file_path = os.path.join(directory_path, 'evaluation_metrics.csv')
    scores_df.to_csv(scores_file_path, index=False)

def generate_plots_multi(df, abs_src_feature_list, ratio_feature_list, model_predicted , save_plot_path,
                   dict_of_regression_model_variables, x_sorted_values=None):
    import matplotlib as mpl

    for abs_src_feature, ratio_feature in zip(abs_src_feature_list,ratio_feature_list):
        mpl.rcParams['agg.path.chunksize'] = 10000  # You can adjust this number as needed

        print("abs_src_feature:%s and ratio_feature:%s " % (abs_src_feature, ratio_feature))
        # Create the main figure and axes
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Determine the x-axis range from the line plot data
        x_min, x_max = min(df[abs_src_feature].round(0)), max(df[abs_src_feature].round(0))
        print("Determine the x-axis range from the line plot data: x_min = %s, x_max = %s" % (x_min, x_max))
        bins = np.arange(x_min - 1, x_max + 2)  # +2 to include the upper edge for the last bin
        print("bins for histogram: %s" % (bins))

        # Plotting the line graph on ax1 with a label
        sns.lineplot(x=df[abs_src_feature], y=df[ratio_feature], ax=ax1, color='tab:blue', linewidth=1.0, label="Ratios on training data") #ratio_feature
        # sns.regplot(x=df[abs_src_feature], y=df[ratio_feature], scatter=False, ci=None, color='tab:red', order=2)

        # plot LR model line
        plt.plot(df[abs_src_feature], model_predicted, color='red', linewidth=2,
             label='Regressor') #CatBoost
        # sns.lineplot(x=df[abs_src_feature], y=model_predicted, ax=ax1, color='tab:red', linewidth=1.0, label="CatBoost  Regression") #ratio_feature

        # Create the secondary y-axis
        ax2 = ax1.twinx()

        # Plotting the histogram on ax2 with a label (no log scale for frequencies)
        sns.histplot(df[abs_src_feature], ax=ax2, bins=bins, color='#FDF3F3', kde=False, stat="count",
                     edgecolor='#F2C0BD', label='No. of occurrences') # + abs_src_feature

        # # Annotate each count vertically inside each histogram block
        # for p in ax2.patches:
        #     ax2.annotate(f'{int(p.get_height())}',
        #                  (p.get_x() + p.get_width() / 2., p.get_height() / 2),
        #                  ha='center', va='center', fontsize=5, color='black', rotation='vertical')

        # Set the same x-axis range for both plots
        ax1.set_xlim(x_min - 1, x_max + 1)
        ax2.set_xlim(x_min -1, x_max + 1)

        # Bring the lineplot (ax1) to the front
        ax1.set_zorder(ax2.get_zorder() + 1)
        ax1.patch.set_visible(False)  # Make ax1 background transparent

        # Labels, titles, and legend
        ax1.set_title("Regression of %s vs %s" % (ratio_feature, abs_src_feature))
        ax1.set_xlabel(abs_src_feature)
        ax1.set_ylabel(ratio_feature)
        ax2.set_ylabel('No. of ' + abs_src_feature)
        # Combine legends from both axes
        handles, labels = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles + handles2, labels + labels2, loc='upper right')

        fig.savefig(f"{save_plot_path}_{ratio_feature}_vs_{abs_src_feature}", dpi=300, bbox_inches='tight')

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file_path", required=False, help="")
    parser.add_argument("--train_src_file_path", required=False, help="")
    parser.add_argument("--eval_file", required=False, help="")
    parser.add_argument("--eval_src_file", required=False, help="")
    parser.add_argument("--abs_src_feature", required=False, help="")
    parser.add_argument("--ratio_feature", required=False, help="")
    parser.add_argument("--save_path", required=False, help="")
    parser.add_argument("--model_filename", required=False, help="")
    parser.add_argument("--test_file_path", required=False, help="")
    parser.add_argument("--test_or_valid", required=False, help="")
    parser.add_argument("--do_train_single", action="store_true", required=False, help="Request to conduct training")
    parser.add_argument("--do_train_multi", action="store_true", required=False, help="Request to conduct training")
    parser.add_argument("--do_eval_single", action="store_true", required=False, help="Request to conduct training")
    parser.add_argument("--do_eval_multi", action="store_true", required=False, help="Request to conduct training")
    parser.add_argument("--grade_level", required=False, help="")
    args = vars(parser.parse_args())

    if args["do_train_single"]:
        generate_catboost_regression_train(args["train_file_path"], args["eval_file"],
                                         args["abs_src_feature"], args["ratio_feature"],
                                         args["save_path"])

    if args["do_eval_single"]:
        catboost_regression_test(args["model_filename"], args["test_file_path"],
                               args["abs_src_feature"], args["ratio_feature"],
                               args["save_path"], args["test_or_valid"])

    if args["do_train_multi"]:
        generate_catboost_regression_train_multi(args["train_file_path"], args["eval_file"],
                                         args["abs_src_feature"].split(','), args["ratio_feature"].split(','),
                                         args["save_path"])

    if args["do_eval_multi"]:
        catboost_regression_test_multi(args["model_filename"], args["test_file_path"],
                               args["abs_src_feature"].split(','), args["ratio_feature"].split(','),
                               args["save_path"], args["test_or_valid"], args["grade_level"])




