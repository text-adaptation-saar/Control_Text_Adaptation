import argparse
import pandas as pd
import numpy as np
import os
from llm_based_control_rewrite.utils.helpers import write_success_rate_test_stats
from scipy.stats import pearsonr

def calculate_RatioSuccess_rate(predicted_ratio_files, obtained_ratio_file, feature_names, success_rate_type,
                           default_input_src, tested_input_src, output_generation_path, default_ref_tgt, tested_ref_tgt):

    final_string_to_return = ""
    for feature, predicted_ratio_file in zip(feature_names.split(','), predicted_ratio_files.split(',')):

        check_identical, reason = compare_two_test_files(default_input_src, tested_input_src)
        if check_identical:
            print(f"default_input_src and tested_input_src are identical. default_input_src:{default_input_src},\ttested_input_src: {tested_input_src}")
            print(f"calculate_ratio_success_rate for {feature} : predicted_ratio_file={predicted_ratio_file}, obtained_ratio_file={obtained_ratio_file}, "
                f"success_rate_type: {success_rate_type} ")
            df1_predicted = pd.read_csv(predicted_ratio_file.strip())
        else:
            print(f"default_input_src and tested_input_src are not identical: {reason}. default_input_src:{default_input_src},\ttested_input_src: {tested_input_src}")
            filtered_predicted_ratio_file_path = os.path.join(output_generation_path,
                                                              f"copied_filtered_ratio_stats_with_LR_predicted_ratio_for_{feature}.csv")

            # if not os.path.exists(filtered_predicted_ratio_file_path):
            #     print(f"filtered_predicted_ratio_file_path is not exits, so going to filter predicted_ratios according to tested_input_src: {tested_input_src}")
            filter_ratios_for_only_considered_input_sentences(default_input_src, predicted_ratio_file,
                                                              tested_input_src, filtered_predicted_ratio_file_path,
                                                              default_ref_tgt, tested_ref_tgt)
            df1_predicted = pd.read_csv(filtered_predicted_ratio_file_path.strip())
            print(f"calculate_ratio_success_rate for {feature} : filtered_predicted_ratio_file_path={filtered_predicted_ratio_file_path}, obtained_ratio_file={obtained_ratio_file}, "
                f"success_rate_type: {success_rate_type} ")

        df2_obtained = pd.read_csv(obtained_ratio_file)
        # Check if the second DataFrame has at least 200 rows
        if len(df1_predicted) != len(df2_obtained):
            raise ValueError("The predicted_ratio_file file does not have same rows compared to obtained_ratio_file.")

        # Count successes for the first 200 rows
        success_count = 0
        squared_errors = []  # List to store squared errors for MSE and RMSE calculation
        actual_vs_predicted = []  # For correlation calculation
        for i in range(len(df2_obtained)):  # Limiting the loop to the first 200 rows
            # In ratio match success rate, ratio round to one decimal point.
            predicted = round(df1_predicted.loc[i, "predicted_" + feature + "_ratio"],1)
            actual = round(df2_obtained.loc[i, feature + "_ratio"], 1)
            if feature == "FreqRank" or feature == "Leven":
                predicted = round(df1_predicted.loc[i, "predicted_" + feature + "_ratio"], 2)
                actual = round(df2_obtained.loc[i, feature + "_ratio"], 2)
            print(f"requested predicted_ratio: {predicted} \t actual_ratio: {actual}")
            squared_errors.append((df1_predicted.loc[i, "predicted_" + feature + "_ratio"] - df2_obtained.loc[i, feature + "_ratio"]) ** 2)  # Calculate squared error for current row and add it to the list.
            actual_vs_predicted.append((df2_obtained.loc[i, feature + "_ratio"], df1_predicted.loc[i, "predicted_" + feature + "_ratio"]))

            if "exact_match" == success_rate_type and predicted == actual:
                success_count += 1
                print(f"exact_match LR_predicted: index:{i}, {predicted} and actual:{actual}, COUNT={success_count}")
            elif "equal_or_lessthan" == success_rate_type and actual <= predicted:
                success_count += 1
                print(f"equal_or_lessthan: index:{i}, LR_predicted: {predicted} and actual:{actual}, COUNT={success_count}")

        final_string_to_return = final_RatioSuccessRate_cal(actual_vs_predicted, df2_obtained, feature,
                                                              final_string_to_return, obtained_ratio_file,
                                                              squared_errors, success_count, success_rate_type, sufix="lr")

    return final_string_to_return.strip(',')


def final_RatioSuccessRate_cal(actual_vs_predicted, df2_obtained, feature, final_string_to_return,
                                 obtained_ratio_file, squared_errors, success_count, success_rate_type, sufix):
    # Calculate success rate
    success_rate = (success_count / len(df2_obtained)) * 100
    print(f"Ratiosuccess_rate for {feature} {success_rate_type}: {success_rate}% on {len(df2_obtained)} sentences from: {obtained_ratio_file}.")
    final_string_to_return += f"{sufix}_RatioSuccess_rate_for_" + feature + "_" + success_rate_type + "," + str(
        success_rate) + ","
    if "exact_match" == success_rate_type:
        # Calculate Mean Squared Error (MSE)
        mse = np.mean(squared_errors)
        rmse = np.sqrt(mse)
        actual, predicted = zip(*actual_vs_predicted)
        # correlation_matrix = np.corrcoef(predicted, actual)
        # correlation = correlation_matrix[0, 1]  # Correlation between actual and predicted
        correlation, p_value = pearsonr(predicted, actual)
        print(f"Ratio of feature: {feature}\tRatioMSE: {mse}\tRatioRMSE:{rmse}")
        final_string_to_return += f"{sufix}_RatioMSE_for_{feature}, {mse},"
        final_string_to_return += f"{sufix}_RatioRMSE_for_{feature}, {rmse},"
        final_string_to_return += f"{sufix}_RatioCorrelation_for_{feature}, {correlation},"
        final_string_to_return += f"{sufix}_RatioCorrelation_p_value_for_{feature}, {p_value},"
    return final_string_to_return

def filter_ratios_for_only_considered_input_sentences(default_input_src, default_predicted_ratio_file, tested_input_src, filtered_predicted_ratio_file_path,
                                                      default_ref_tgt, tested_ref_tgt):

        # print("Filtering ratio for only considered input sentences:")
        # print(f"Reading default_input_src: {default_input_src}, default_ref_tgt:{default_ref_tgt}")
        # # Read and combine input and reference sentences for both default and tested sets
        # with open(default_input_src, 'r') as file1, open(default_ref_tgt, 'r') as file2:
        #     default_combined = [f"{src.strip()} - {tgt.strip()}" for src, tgt in zip(file1, file2)]
        #
        # print(f"Reading tested_input_src: {tested_input_src}, tested_ref_tgt:{tested_ref_tgt}")
        # with open(tested_input_src, 'r') as file1, open(tested_ref_tgt, 'r') as file2:
        #     tested_combined = [f"{src.strip()} - {tgt.strip()}" for src, tgt in zip(file1, file2)]
        #
        # # Create a map from combined sentence to index for default set
        # sentence_to_index = {sentence: i for i, sentence in enumerate(default_combined)}
        #
        # # Determine indexes of tested sentences within the default set
        # indexes_to_keep = [sentence_to_index[sentence] for sentence in tested_combined if sentence in sentence_to_index]
        #
        # # Read the predicted ratios CSV file
        # print(f"Reading default_predicted_ratio_file: {default_predicted_ratio_file}")
        # df_predicted_ratios = pd.read_csv(default_predicted_ratio_file)
        #
        # # Filter the DataFrame based on these indexes
        # filtered_df = df_predicted_ratios.iloc[indexes_to_keep]
        #
        # # Save the filtered DataFrame to a new CSV file
        # filtered_df.to_csv(filtered_predicted_ratio_file_path, index=False)
        #
        # print(f"Filtered predicted ratio file saved to {filtered_predicted_ratio_file_path}")


        print(f"filter ratio for only considered input sentences:")
        print(f"Reading default_input_src: {default_input_src}, default_ref_tgt:{default_ref_tgt}")
        # Read the full list of input sentences
        with open(default_input_src, 'r') as file, open(default_ref_tgt, 'r') as file2:
            default_input_sentences = file.readlines()
            default_ref_sentences = file2.readlines()
        default_input_vs_ref_sentences = [line_src.strip() + "-" + line_ref.strip() for line_src, line_ref in zip(default_input_sentences,default_ref_sentences)]

        # Read the predicted ratios CSV file
        print(f"Reading default_predicted_ratio_file: {default_predicted_ratio_file}")
        df_predicted_ratios = pd.read_csv(default_predicted_ratio_file)

        # Read the selected input sentences
        print(f"Reading tested_input_src: {tested_input_src}, tested_ref_tgt:{tested_ref_tgt}")
        with open(tested_input_src, 'r') as file, open(tested_ref_tgt, 'r') as file2:
            tested_input_sentences = file.readlines()
            tested_ref_sentences = file2.readlines()
        tested_input_vs_ref_sentences = [line_src.strip() + "-" + line_ref.strip() for line_src, line_ref in zip(tested_input_sentences,tested_ref_sentences)]

        # Filter to get the indexes of the selected sentences in the full list
        indexes_to_keep = [default_input_vs_ref_sentences.index(sentence) for sentence in tested_input_vs_ref_sentences if
                           sentence in default_input_vs_ref_sentences]

        # Filter the DataFrame based on these indexes
        filtered_df = df_predicted_ratios.iloc[indexes_to_keep]

        # Save the filtered DataFrame to a new CSV file
        filtered_df.to_csv(filtered_predicted_ratio_file_path, index=False)

        print(f"Filtered predicted ratio file saved to {filtered_predicted_ratio_file_path}")


def compare_two_test_files(file_path_1, file_path_2):
    # Read the content of both files
    with open(file_path_1, 'r') as file1:
        file1_lines = file1.readlines()
    with open(file_path_2, 'r') as file2:
        file2_lines = file2.readlines()

    # Check if both files have the same number of lines
    if len(file1_lines) != len(file2_lines):
        return False, f"The files have a different number of lines: {len(file1_lines)} vs {len(file2_lines)}"

    # Compare each line
    for line1, line2 in zip(file1_lines, file2_lines):
        if line1.strip() != line2.strip():
            return False, "The files are not identical."

    # If all checks pass
    return True, "The files have the same number of lines and are identical."

def calculate_abs_tgt_value_metrics_for_lr(predicted_ratio_files, obtained_ratio_file, feature_names, success_rate_type, feature_range,
                             default_input_src, tested_input_src, default_ref_tgt, tested_ref_tgt, output_generation_path,
                                           default_prefix_for_predicted_values="predicted_abs_tgt",
                                           default_prefix_for_gen_output_values="abs_tgt"):
    final_string_to_return = ""
    for feature, predicted_ratio_file, feature_range_diff in zip(feature_names.split(','), predicted_ratio_files.split(','), feature_range):

        check_identical, reason = compare_two_test_files(default_input_src,tested_input_src)
        if check_identical:
            print(f"default_input_src and tested_input_src are identical. default_input_src:{default_input_src},\ttested_input_src: {tested_input_src}")
            print(f"calculate_abs_tgt_success_rate for {feature} : predicted_ratio_file={predicted_ratio_file}, obtained_ratio_file={obtained_ratio_file}, "
                f"success_rate_type: {success_rate_type} ")
            df1_predicted = pd.read_csv(predicted_ratio_file.strip())
        else:
            print(f"default_input_src and tested_input_src are not identical: Reason: {reason}. default_input_src:{default_input_src},\ttested_input_src: {tested_input_src}")
            filtered_predicted_ratio_file_path = os.path.join(output_generation_path,
                                                              f"copied_filtered_ratio_stats_with_LR_predicted_ratio_for_{feature}.csv")

            # if not os.path.exists(filtered_predicted_ratio_file_path):
            #     print(f"filtered_predicted_ratio_file_path is not exits, so going to filter predicted_ratios according to tested_input_src: {tested_input_src}")
            filter_ratios_for_only_considered_input_sentences(default_input_src, predicted_ratio_file,
                                                          tested_input_src, filtered_predicted_ratio_file_path,
                                                              default_ref_tgt, tested_ref_tgt)
            df1_predicted = pd.read_csv(filtered_predicted_ratio_file_path.strip())
            print(f"calculate_abs_tgt_success_rate for {feature} : filtered_predicted_ratio_file_path={filtered_predicted_ratio_file_path}, obtained_ratio_file={obtained_ratio_file}, "
                f"success_rate_type: {success_rate_type} ")

        df2_obtained = pd.read_csv(obtained_ratio_file.strip())
        if len(df1_predicted) != len(df2_obtained):
            raise ValueError("The predicted_ratio_file file does not have same rows compared to obtained_ratio_file.")

        success_count = 0
        squared_errors = []  # List to store squared errors for MSE and RMSE calculation
        fuzzy_success_count = 0
        fuzzy_squared_errors = []  # List to store squared errors for MSE and RMSE calculation
        actual_vs_predicted = []  # For correlation calculation

        for i in range(len(df2_obtained)):
            # predicted_ratio = df1_predicted.loc[i, f"predicted_{feature}_ratio"]
            # ideal_tgt = round(predicted_ratio * df2_obtained.loc[i, f"abs_src_{feature}"])
            ideal_tgt = round(df1_predicted.loc[i, f"{default_prefix_for_predicted_values}_{feature}"])
            actual_tgt = round(df2_obtained.loc[i, f"{default_prefix_for_gen_output_values}_{feature}"]) # bcz wordcount might be in decimal.
            squared_errors.append((ideal_tgt - actual_tgt) ** 2) # Calculate squared error for current row and add it to the list.
            actual_vs_predicted.append((actual_tgt, ideal_tgt))
            print(f"for line: {i} requested ideal_tgt: {ideal_tgt} \t actual_tgt: {actual_tgt}")

            fuzzy_success_count = add_fuzzy_success_rate_and_mse(actual_tgt, feature_range_diff, fuzzy_squared_errors,
                                                                 fuzzy_success_count, i, ideal_tgt, success_rate_type)

            # calcaute exact match success rate and mse
            if "exact_match" == success_rate_type and ideal_tgt == actual_tgt:
                success_count += 1
            elif "equal_or_lessthan" == success_rate_type and actual_tgt <= ideal_tgt:
                success_count += 1

        final_string_to_return = generate_final_values(actual_vs_predicted, df2_obtained, feature,
                                                       final_string_to_return, fuzzy_squared_errors,
                                                       fuzzy_success_count, obtained_ratio_file, squared_errors,
                                                       success_count, success_rate_type, sufix="lr")

    return final_string_to_return.strip(',')


def generate_final_values(actual_vs_predicted, df2_obtained, feature, final_string_to_return, fuzzy_squared_errors,
                          fuzzy_success_count, obtained_ratio_file, squared_errors, success_count, success_rate_type,
                          sufix):
    # Calculate success rate
    success_rate = (success_count / len(df2_obtained)) * 100
    fuzzy_success_rate = (fuzzy_success_count / len(df2_obtained)) * 100
    print(
        f"EXACT: abs_tgt_success_rate for {feature} {success_rate_type}: {success_rate}% on {len(df2_obtained)} sentences from: {obtained_ratio_file}.")
    print(
        f"FUZZY: abs_tgt_success_rate for {feature} {success_rate_type}: {fuzzy_success_rate}% on {len(df2_obtained)} sentences from: {obtained_ratio_file}.")
    final_string_to_return += f"{sufix}_abs_tgt_EXACT_success_rate_for_{feature}_{success_rate_type}, {success_rate},"
    final_string_to_return += f"{sufix}_abs_tgt_FUZZY_success_rate_for_{feature}_{success_rate_type}, {fuzzy_success_rate},"
    if "exact_match" == success_rate_type:
        # Calculate Mean Squared Error (MSE)
        print(f"Length of squared_errors list: {len(squared_errors)}\t squared_errors: {squared_errors}")
        mse = np.mean(squared_errors)
        rmse = np.sqrt(mse)
        fuzzy_mse = np.mean(fuzzy_squared_errors)
        fuzzy_rmse = np.sqrt(fuzzy_mse)
        actual, predicted = zip(*actual_vs_predicted)
        # correlation_matrix = np.corrcoef(predicted, actual)
        # correlation = correlation_matrix[0, 1]  # Correlation between actual and predicted
        correlation, p_value = pearsonr(predicted, actual) # order doesnt matter

        print(f"EXACT: feature: {feature}\tMSE: {mse}\tRMSE:{rmse}")
        print(f"FUZZY: feature: {feature}\tMSE: {fuzzy_mse}\tRMSE:{fuzzy_rmse}")
        final_string_to_return += f"{sufix}_EXACT_mse_for_{feature}, {mse},"
        final_string_to_return += f"{sufix}_FUZZY_mse_for_{feature}, {fuzzy_mse},"

        final_string_to_return += f"{sufix}_EXACT_rmse_for_{feature}, {rmse},"
        final_string_to_return += f"{sufix}_FUZZY_rmse_for_{feature}, {fuzzy_rmse},"

        final_string_to_return += f"{sufix}_EXACT_correlation_for_{feature}, {correlation},"
        final_string_to_return += f"{sufix}_EXACT_correlation_p_value_for_{feature}, {p_value},"
    return final_string_to_return


def add_fuzzy_success_rate_and_mse(actual_tgt, feature_range_diff, fuzzy_squared_errors, fuzzy_success_count, i,
                                   ideal_tgt, success_rate_type):
    # if fuzzy range give, evalute agsinst it.
    if int(feature_range_diff) > 0:
        tgt_min = ideal_tgt - int(feature_range_diff)
        tgt_max = ideal_tgt + int(feature_range_diff)
        print(f"for line: {i} feature range is given. tgt_min: {tgt_min}, tgt_max:{tgt_max}")

        if "exact_match" == success_rate_type and tgt_min <= actual_tgt and actual_tgt <= tgt_max:
            fuzzy_success_count += 1
        elif "equal_or_lessthan" == success_rate_type and actual_tgt <= tgt_max:
            fuzzy_success_count += 1

        if tgt_min <= actual_tgt and actual_tgt <= tgt_max:
            fuzzy_squared_errors.append(0 ** 2)  # //skip
        else:
            diff = min(tgt_max - actual_tgt, actual_tgt - tgt_min)
            fuzzy_squared_errors.append(diff ** 2)
    return fuzzy_success_count





def calculate_abs_tgt_metrics_for_gold_ref(gold_ref_file, obtained_ratio_file, feature_names, success_rate_type, feature_range,
                             default_input_src, tested_input_src, default_ref_tgt, tested_ref_tgt, output_generation_path):
    final_string_to_return = ""
    df2_obtained = pd.read_csv(obtained_ratio_file.strip())
    check_identical, reason = compare_two_test_files(default_input_src, tested_input_src)
    if check_identical:
        print(f"default_input_src and tested_input_src are identical. default_input_src:{default_input_src},\ttested_input_src: {tested_input_src}")
        print(f"calculate_abs_tgt_success_rate for {feature_names} : GOLD gold_ref_file={gold_ref_file}, obtained_ratio_file={obtained_ratio_file}, "
            f"success_rate_type: {success_rate_type} ")
        df1_predicted = pd.read_csv(gold_ref_file.strip())
    else:
        print(f"default_input_src and tested_input_src are not identical: Reason: {reason}. default_input_src:{default_input_src},\ttested_input_src: {tested_input_src}")
        filtered_gold_ratio_file_path = os.path.join(output_generation_path, f"copied_filtered_gold_ratio_stats.csv")

        # if not os.path.exists(filtered_gold_ratio_file_path):
        #     print(f"filtered_gold_ratio_file_path is not exits, so going to filter GOLD_ratios according to tested_input_src: {tested_input_src}")
        filter_ratios_for_only_considered_input_sentences(default_input_src, gold_ref_file,
                                                          tested_input_src, filtered_gold_ratio_file_path,
                                                          default_ref_tgt, tested_ref_tgt)
        df1_predicted = pd.read_csv(filtered_gold_ratio_file_path.strip())
        print(f"calculate_abs_tgt_success_rate for {feature_names}: filtered_gold_ratio_file_path={filtered_gold_ratio_file_path}, obtained_ratio_file={obtained_ratio_file}, "
            f"success_rate_type: {success_rate_type} ")

    if len(df1_predicted) != len(df2_obtained):
        raise ValueError("The gold_ref_file file does not have same rows compared to obtained_ratio_file.")

    for feature, feature_range_diff in zip(feature_names.split(','), feature_range):
        print(f"Calcuate abs_tgt_success rate for feature: {feature}")
        success_count = 0
        squared_errors = []  # List to store squared errors for MSE and RMSE calculation
        fuzzy_success_count = 0
        fuzzy_squared_errors = []  # List to store squared errors for MSE and RMSE calculation
        actual_vs_predicted = []  # For correlation calculation

        for i in range(len(df2_obtained)):
            if feature == "Grade":
                ideal_tgt = round(df1_predicted.loc[i, f"abs_tgt_FKGL_{feature}"]) # from gold ratio file.
            else:
                ideal_tgt = round(df1_predicted.loc[i, f"abs_tgt_{feature}"]) # from gold ratio file.
            actual_tgt = round(df2_obtained.loc[i, f"abs_tgt_{feature}"]) # bcz wordcount might be in decimal.
            squared_errors.append((ideal_tgt - actual_tgt) ** 2) # Calculate squared error for current row and add it to the list.
            actual_vs_predicted.append((actual_tgt, ideal_tgt))
            print(f"for line: {i} requested ideal_tgt: {ideal_tgt} \t actual_tgt: {actual_tgt}")

            fuzzy_success_count = add_fuzzy_success_rate_and_mse(actual_tgt, feature_range_diff, fuzzy_squared_errors,
                                                                 fuzzy_success_count, i, ideal_tgt, success_rate_type)

            # calcaute exact match success rate and mse
            if "exact_match" == success_rate_type and ideal_tgt == actual_tgt:
                success_count += 1
            elif "equal_or_lessthan" == success_rate_type and actual_tgt <= ideal_tgt:
                success_count += 1

        final_string_to_return = generate_final_values(actual_vs_predicted, df2_obtained, feature,
                                                       final_string_to_return, fuzzy_squared_errors,
                                                       fuzzy_success_count, obtained_ratio_file, squared_errors,
                                                       success_count, success_rate_type, sufix="gold_ref")

    return final_string_to_return.strip(',')


def calculate_RatioSuccess_rate_gold_ref(gold_ref_file, obtained_ratio_file, feature_names, success_rate_type,
                           default_input_src, tested_input_src, output_generation_path, default_ref_tgt, tested_ref_tgt):
    final_string_to_return = ""
    df2_obtained = pd.read_csv(obtained_ratio_file.strip())
    check_identical, reason = compare_two_test_files(default_input_src, tested_input_src)
    if check_identical:
        print(f"default_input_src and tested_input_src are identical. default_input_src:{default_input_src},\ttested_input_src: {tested_input_src}")
        print(f"calculate_Ratiosuccess_rate for {feature_names} : GOLD gold_ref_file={gold_ref_file}, obtained_ratio_file={obtained_ratio_file}, "
            f"success_rate_type: {success_rate_type} ")
        df1_predicted = pd.read_csv(gold_ref_file.strip())
    else:
        print(f"default_input_src and tested_input_src are not identical: Reason: {reason}. default_input_src:{default_input_src},\ttested_input_src: {tested_input_src}")
        filtered_gold_ratio_file_path = os.path.join(output_generation_path, f"copied_filtered_gold_ratio_stats.csv")

        # if not os.path.exists(filtered_gold_ratio_file_path):
        #     print(f"filtered_gold_ratio_file_path is not exits, so going to filter GOLD_ratios according to tested_input_src: {tested_input_src}")
        filter_ratios_for_only_considered_input_sentences(default_input_src, gold_ref_file,
                                                              tested_input_src, filtered_gold_ratio_file_path,
                                                              default_ref_tgt, tested_ref_tgt)
        df1_predicted = pd.read_csv(filtered_gold_ratio_file_path.strip())
        print(f"calculate_Ratiosuccess_rate: filtered_gold_ratio_file_path={filtered_gold_ratio_file_path}, obtained_ratio_file={obtained_ratio_file}, "
            f"success_rate_type: {success_rate_type} ")

    # Check if the second DataFrame has at least 200 rows
    if len(df1_predicted) != len(df2_obtained):
        raise ValueError("The gold_ref_file file does not have same rows compared to obtained_ratio_file.")

    for feature in feature_names.split(','):
        print(f"Calcuate abs_tgt_success rate for feature: {feature}")
        # Count successes for the first 200 rows
        success_count = 0
        squared_errors = []  # List to store squared errors for MSE and RMSE calculation
        actual_vs_predicted = []  # For correlation calculation
        for i in range(len(df2_obtained)):  # Limiting the loop to the first 200 rows
            # In ratio match success rate, ratio round to one decimal point.
            predicted = round(df1_predicted.loc[i, feature + "_ratio"],1) # bcz we read from gold ratio file.
            actual = round(df2_obtained.loc[i, feature + "_ratio"], 1)
            squared_errors.append((df1_predicted.loc[i, feature + "_ratio"] - df2_obtained.loc[i, feature + "_ratio"]) ** 2)  # Calculate squared error for current row and add it to the list.
            actual_vs_predicted.append((df2_obtained.loc[i, feature + "_ratio"], df1_predicted.loc[i, feature + "_ratio"]))
            print(f"requested gold_ref_ratio: {predicted} \t actual_ratio: {actual}")

            if "exact_match" == success_rate_type and predicted == actual:
                success_count += 1
                print(f"exact_match gold_ref_ratio: index:{i}, {predicted} and actual:{actual}, COUNT={success_count}")
            elif "equal_or_lessthan" == success_rate_type and actual <= predicted:
                success_count += 1
                print(f"equal_or_lessthan: index:{i}, gold_ref_ratio: {predicted} and actual:{actual}, COUNT={success_count}")

        final_string_to_return = final_RatioSuccessRate_cal(actual_vs_predicted, df2_obtained, feature,
                                                              final_string_to_return, obtained_ratio_file,
                                                              squared_errors, success_count, success_rate_type,
                                                              sufix="gold_ref")

    return final_string_to_return.strip(',')




def calculate_RatioSuccess_rate_fixed_ratio(predicted_fixed_ratio_dict, obtained_ratio_file, feature_names, success_rate_type):
    final_string_to_return = ""
    for feature in feature_names.split(','):
        df2_obtained = pd.read_csv(obtained_ratio_file)
        print(f"calculate_success_rate_fixed_ratio for {feature} : predicted_fixed_ratio={predicted_fixed_ratio_dict}, "
              f"obtained_ratio_file: {obtained_ratio_file}, success_rate_type: {success_rate_type} ")

        # Count successes for the first 200 rows
        success_count = 0
        squared_errors = []  # List to store squared errors for MSE and RMSE calculation
        actual_vs_predicted = []  # For correlation calculation
        for i in range(len(df2_obtained)):  # Limiting the loop to the first 200 rows
            # In ratio match success rate, ratio round to one decimal point.
            predicted = round(predicted_fixed_ratio_dict[feature], 1)
            actual = round(df2_obtained.loc[i, feature + "_ratio"], 1)
            squared_errors.append((predicted_fixed_ratio_dict[feature] - df2_obtained.loc[i, feature + "_ratio"]) ** 2)  # Calculate squared error for current row and add it to the list.
            actual_vs_predicted.append((df2_obtained.loc[i, feature + "_ratio"], predicted_fixed_ratio_dict[feature]))
            if "exact_match" == success_rate_type and predicted == actual:
                success_count += 1
                print(f"exact_match fixed_ratio: index:{i}, {predicted} and actual:{actual}, COUNT={success_count}")
            elif "equal_or_lessthan" == success_rate_type and actual <= predicted :
                success_count += 1
                print(f"equal_or_lessthan fixed_ratio: index:{i}, {predicted} and actual:{actual}, COUNT={success_count}")

        final_string_to_return = final_RatioSuccessRate_cal(actual_vs_predicted, df2_obtained, feature,
                                                              final_string_to_return, obtained_ratio_file,
                                                              squared_errors, success_count, success_rate_type,
                                                              sufix="fr")

    return final_string_to_return.strip(',')


def calculate_abs_tgt_value_metrics_for_fr(predicted_fixed_ratio_dict, obtained_ratio_file, feature_names, success_rate_type,
                                                           feature_range):
    final_string_to_return = ""
    for feature, feature_range_diff in zip(feature_names.split(','), feature_range):
        df2_obtained = pd.read_csv(obtained_ratio_file.strip())
        print(f"calculate_abs_tgt_success_rate_fixed_ratio for {feature} : predicted_fixed_ratio={predicted_fixed_ratio_dict}, "
              f"obtained_ratio_file: {obtained_ratio_file}, success_rate_type: {success_rate_type} ")
        success_count = 0
        squared_errors = []  # List to store squared errors for MSE and RMSE calculation
        fuzzy_success_count = 0
        fuzzy_squared_errors = []  # List to store squared errors for MSE and RMSE calculation
        actual_vs_predicted = []  # For correlation calculation

        for i in range(len(df2_obtained)):
            ideal_tgt = round(predicted_fixed_ratio_dict[feature] * df2_obtained.loc[i, f"abs_src_{feature}"])
            actual_tgt = round(df2_obtained.loc[i, f"abs_tgt_{feature}"]) # bcz wordcount might be in decimal.
            squared_errors.append((ideal_tgt - actual_tgt) ** 2)  # Calculate squared error for current row and add it to the list.
            actual_vs_predicted.append((actual_tgt, ideal_tgt))

            fuzzy_success_count = add_fuzzy_success_rate_and_mse(actual_tgt, feature_range_diff, fuzzy_squared_errors,
                                                                 fuzzy_success_count, i, ideal_tgt, success_rate_type)

            # calcaute exact match success rate and mse
            if "exact_match" == success_rate_type and ideal_tgt == actual_tgt:
                success_count += 1
            elif "equal_or_lessthan" == success_rate_type and actual_tgt <= ideal_tgt:
                success_count += 1

        final_string_to_return = generate_final_values(actual_vs_predicted, df2_obtained, feature,
                              final_string_to_return, fuzzy_squared_errors,
                              fuzzy_success_count, obtained_ratio_file, squared_errors,
                              success_count, success_rate_type, sufix="fr")

    return final_string_to_return.strip(',')

def calculate_abs_tgt_value_metrics_for_grade(predicted_fixed_ratio_dict, obtained_ratio_file, feature_names, success_rate_type,
                                                           feature_range):
    final_string_to_return = ""
    for feature, feature_range_diff in zip(feature_names.split(','), feature_range):
        df2_obtained = pd.read_csv(obtained_ratio_file.strip())
        print(f"calculate_abs_tgt_success_rate_for_Grade for {feature} : predicted_Grade={predicted_fixed_ratio_dict}, "
              f"obtained_ratio_file: {obtained_ratio_file}, success_rate_type: {success_rate_type} ")
        success_count = 0
        squared_errors = []  # List to store squared errors for MSE and RMSE calculation
        fuzzy_success_count = 0
        fuzzy_squared_errors = []  # List to store squared errors for MSE and RMSE calculation
        actual_vs_predicted = []  # For correlation calculation

        for i in range(len(df2_obtained)):
            ideal_tgt = round(predicted_fixed_ratio_dict[feature]) #grade is given in abs value. no ratios.
            actual_tgt = round(df2_obtained.loc[i, f"abs_tgt_{feature}"]) # bcz wordcount might be in decimal.
            squared_errors.append((ideal_tgt - actual_tgt) ** 2)  # Calculate squared error for current row and add it to the list.
            actual_vs_predicted.append((actual_tgt, ideal_tgt))

            fuzzy_success_count = add_fuzzy_success_rate_and_mse(actual_tgt, feature_range_diff, fuzzy_squared_errors,
                                                                 fuzzy_success_count, i, ideal_tgt, success_rate_type)

            # calcaute exact match success rate and mse
            if "exact_match" == success_rate_type and ideal_tgt == actual_tgt:
                success_count += 1
            elif "equal_or_lessthan" == success_rate_type and actual_tgt <= ideal_tgt:
                success_count += 1

        final_string_to_return = generate_final_values(actual_vs_predicted, df2_obtained, feature,
                              final_string_to_return, fuzzy_squared_errors,
                              fuzzy_success_count, obtained_ratio_file, squared_errors,
                              success_count, success_rate_type, sufix="tg")

    return final_string_to_return.strip(',')

def calculate_RatioSuccess_rate_for_Grade(predicted_fixed_ratio_dict, obtained_ratio_file, feature_names, success_rate_type):
    final_string_to_return = ""
    for feature in feature_names.split(','):
        df2_obtained = pd.read_csv(obtained_ratio_file)
        print(f"calculate_success_rate_for_Grade for {feature} : predicted_Grade={predicted_fixed_ratio_dict}, "
              f"obtained_ratio_file: {obtained_ratio_file}, success_rate_type: {success_rate_type} ")

        # Count successes for the first 200 rows
        success_count = 0
        squared_errors = []  # List to store squared errors for MSE and RMSE calculation
        actual_vs_predicted = []  # For correlation calculation
        for i in range(len(df2_obtained)):  # Limiting the loop to the first 200 rows
            # In ratio match success rate, ratio round to one decimal point.
            predicted = round((predicted_fixed_ratio_dict[feature]/df2_obtained.loc[i, f"abs_src_{feature}"]), 1) #grade is given in abs value. no ratios.
            actual = round(df2_obtained.loc[i, feature + "_ratio"], 1)
            squared_errors.append((predicted_fixed_ratio_dict[feature] - df2_obtained.loc[i, feature + "_ratio"]) ** 2)  # Calculate squared error for current row and add it to the list.
            actual_vs_predicted.append((df2_obtained.loc[i, feature + "_ratio"], predicted_fixed_ratio_dict[feature]))
            if "exact_match" == success_rate_type and predicted == actual:
                success_count += 1
                print(f"exact_match fixed_ratio: index:{i}, {predicted} and actual:{actual}, COUNT={success_count}")
            elif "equal_or_lessthan" == success_rate_type and actual <= predicted :
                success_count += 1
                print(f"equal_or_lessthan fixed_ratio: index:{i}, {predicted} and actual:{actual}, COUNT={success_count}")

        final_string_to_return = final_RatioSuccessRate_cal(actual_vs_predicted, df2_obtained, feature,
                                                              final_string_to_return, obtained_ratio_file,
                                                              squared_errors, success_count, success_rate_type,
                                                              sufix="tg")

    return final_string_to_return.strip(',')

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predicted_ratio_files", required=False, help="yaml config file for feature based evaluation")
    parser.add_argument("--fixed_ratio", required=False, help="yaml config file for feature based evaluation")
    parser.add_argument("--obtained_ratio_file", required=True, help="yaml config file for feature based evaluation")
    parser.add_argument("--feature_names", required=True, help="yaml config file for feature based evaluation")
    parser.add_argument("--feature_range", required=True, help="yaml config file for feature based evaluation")
    parser.add_argument("--default_input_src", required=False, help="yaml config file for feature based evaluation")
    parser.add_argument("--tested_input_src", required=False, help="yaml config file for feature based evaluation")
    parser.add_argument("--default_ref_tgt", required=False, help="yaml config file for feature based evaluation")
    parser.add_argument("--tested_ref_tgt", required=False, help="yaml config file for feature based evaluation")
    parser.add_argument("--output_generation_path", required=True, help="yaml config file for feature based evaluation")

    # parser.add_argument("--do_score_cal_for_fr", action="store_true", required=False, help="Request to conduct training")

    args = vars(parser.parse_args())

    if args["fixed_ratio"] is not None:
        feature_names_list=args['feature_names'].split(",")
        fixed_ratio_list=args['fixed_ratio'].split(",")
        # give single feature and it's fr.
        predicted_fixed_ratio_dict = {
            feature_names_list[0].strip(): float(fixed_ratio_list[0]),
            feature_names_list[1].strip(): float(fixed_ratio_list[1]),
            feature_names_list[2].strip(): float(fixed_ratio_list[2]),
            feature_names_list[3].strip(): float(fixed_ratio_list[3])
        }

        abs_tgt_success_rate_exact_match_and_mse = calculate_abs_tgt_value_metrics_for_fr(
            predicted_fixed_ratio_dict=predicted_fixed_ratio_dict,
            obtained_ratio_file=args["obtained_ratio_file"],
            feature_names=args["feature_names"],
            success_rate_type="exact_match",
            feature_range=args["feature_range"].split(','))

        abs_tgt_success_rate_equal_or_lessthan = calculate_abs_tgt_value_metrics_for_fr(
            predicted_fixed_ratio_dict=predicted_fixed_ratio_dict,
            obtained_ratio_file=args["obtained_ratio_file"],
            feature_names=args["feature_names"],
            success_rate_type="equal_or_lessthan",
            feature_range=args["feature_range"].split(','))

        #  ratio_success_rate
        success_rate_exact_match = calculate_RatioSuccess_rate_fixed_ratio(
            predicted_fixed_ratio_dict=predicted_fixed_ratio_dict,
            obtained_ratio_file=args["obtained_ratio_file"],
            feature_names=args["feature_names"],
            success_rate_type="exact_match")
        success_rate_equal_or_lessthan = calculate_RatioSuccess_rate_fixed_ratio(
            predicted_fixed_ratio_dict=predicted_fixed_ratio_dict,
            obtained_ratio_file=args["obtained_ratio_file"],
            feature_names=args["feature_names"],
            success_rate_type="equal_or_lessthan")

        path_parts = args["output_generation_path"].split('/')
        subfolder_name = path_parts[-1]
        write_success_rate_test_stats(f"{args['output_generation_path']}/success_rate_{subfolder_name}.csv",
                                      abs_tgt_success_rate_exact_match_and_mse, abs_tgt_success_rate_equal_or_lessthan,
                                      success_rate_exact_match, success_rate_equal_or_lessthan,
                                      f'obtained_ratio_file,{args["obtained_ratio_file"]}, fr_for_{args["feature_names"].strip()},{args["fixed_ratio"].strip()}')
    else:
        abs_tgt_success_rate_exact_match_and_mse = calculate_abs_tgt_value_metrics_for_lr(
            predicted_ratio_files=args["predicted_ratio_files"],
            obtained_ratio_file=args["obtained_ratio_file"],
            feature_names=args["feature_names"],
            success_rate_type="exact_match",
            feature_range=args["feature_range"].split(','),
            default_input_src=args["default_input_src"],
            tested_input_src=args["tested_input_src"],
            default_ref_tgt=args["default_ref_tgt"],
            tested_ref_tgt=args["default_ref_tgt"],
            output_generation_path=args["output_generation_path"])

        abs_tgt_success_rate_equal_or_lessthan = calculate_abs_tgt_value_metrics_for_lr(
            predicted_ratio_files=args["predicted_ratio_files"],
            obtained_ratio_file=args["obtained_ratio_file"],
            feature_names=args["feature_names"],
            success_rate_type="equal_or_lessthan",
            feature_range=args["feature_range"].split(','),
            default_input_src=args["default_input_src"],
            tested_input_src=args["tested_input_src"],
            default_ref_tgt=args["default_ref_tgt"],
            tested_ref_tgt=args["default_ref_tgt"],
            output_generation_path=args["output_generation_path"])

        #  ratio_success_rate
        success_rate_exact_match = calculate_RatioSuccess_rate(
            predicted_ratio_files=args["predicted_ratio_files"],
            obtained_ratio_file=args["obtained_ratio_file"],
            feature_names=args["feature_names"],
            success_rate_type="exact_match",
            default_input_src=args["default_input_src"],
            tested_input_src=args["tested_input_src"],
            default_ref_tgt=args["default_ref_tgt"],
            tested_ref_tgt=args["default_ref_tgt"],
            output_generation_path=args["output_generation_path"])
        success_rate_equal_or_lessthan = calculate_RatioSuccess_rate(
            predicted_ratio_files=args["predicted_ratio_files"],
            obtained_ratio_file=args["obtained_ratio_file"],
            feature_names=args["feature_names"],
            success_rate_type="equal_or_lessthan",
            default_input_src=args["default_input_src"],
            tested_input_src=args["tested_input_src"],
            default_ref_tgt=args["default_ref_tgt"],
            tested_ref_tgt=args["default_ref_tgt"],
            output_generation_path=args["output_generation_path"])

        path_parts = args["predicted_ratio_files"].split('/')
        subfolder_name = path_parts[-2]
        write_success_rate_test_stats(f"{args['output_generation_path']}/success_rate_{subfolder_name}.csv",
                                      abs_tgt_success_rate_exact_match_and_mse, abs_tgt_success_rate_equal_or_lessthan,
                                      success_rate_exact_match, success_rate_equal_or_lessthan,
                                      f'predicted_ratio_files {args["predicted_ratio_files"]} obtained_ratio_file, {args["obtained_ratio_file"]}')


    # calculate_success_rate(predicted_ratio_file=args["predicted_ratio_file"],
    #                       obtained_ratio_file=args["obtained_ratio_file"],
    #                       ratio_feature_name=args["predicted_ratio_feature_name"],
    #                       success_rate_type="exact_match")
    #
    # calculate_success_rate(predicted_ratio_file=args["predicted_ratio_file"],
    #                       obtained_ratio_file=args["obtained_ratio_file"],
    #                       ratio_feature_name=args["predicted_ratio_feature_name"],
    #                       success_rate_type="equal_or_lessthan")

    # calculate_success_rate_fixed_ratio(0.83,
    #                                    obtained_ratio_file=args["obtained_ratio_file"],
    #                                    ratio_feature_name=args["predicted_ratio_feature_name"],
    #                                    success_rate_type="exact_match")
    #
    # calculate_success_rate_fixed_ratio(0.83,
    #                                    obtained_ratio_file=args["obtained_ratio_file"],
    #                                    ratio_feature_name=args["predicted_ratio_feature_name"],
    #                                    success_rate_type="equal_or_lessthan")

# python llm_based_control_rewrite/scores/calculate_success_rate.py \
#     --predicted_ratio_file experiments/data_filtered_regression_model/linear_regression/2_maxdepdepth/wikilarge/valid/ratio_stats_with_LR_predicted_ratio.csv \
#     --obtained_ratio_file experiments/data_filtered_regression_model/linear_regression/2_maxdepdepth/filtered_wiki.valid.src_gpt-4_examples_0_temp_0.3_chain_False/calibration_1/maxdepdepth_-1_maxdeplength_-1_diffwordscount_-1_avgwordcount_-1_length_-1_leven_-1/ratio_stats.csv \
#     --predicted_ratio_feature_name 'MaxDepDepth_ratio' \

# python llm_based_control_rewrite/scores/calculate_success_rate.py \
#     --predicted_ratio_file experiments/data_filtered_regression_model/linear_regression/2_maxdepdepth/wikilarge/valid/ratio_stats_with_LR_predicted_ratio.csv \
#     --obtained_ratio_file experiments/data_filtered_regression_model/linear_regression/2_maxdepdepth/filtered_wiki.valid.src_gpt-4_examples_0_temp_0.3_chain_False/maxdepdepth_-1_maxdeplength_-1_diffwordscount_-1_avgwordcount_-1_length_-1_leven_-1/ratio_stats.csv \
#     --predicted_ratio_feature_name 'MaxDepDepth_ratio' \


# python llm_based_control_rewrite/scores/calculate_success_rate.py \
#     --obtained_ratio_file experiments/data_filtered_regression_model/linear_regression/2_maxdepdepth/fixed_ratio_0.83_filtered_wiki.valid.src_gpt-4_examples_0_temp_0.3_chain_False/maxdepdepth_0.83_maxdeplength_-1_diffwordscount_-1_avgwordcount_-1_length_-1_leven_-1/ratio_stats.csv \
#     --predicted_ratio_feature_name 'MaxDepDepth_ratio' \

# python llm_based_control_rewrite/scores/calculate_success_rate.py \
#     --obtained_ratio_file experiments/data_filtered_regression_model/linear_regression/2_maxdepdepth/fixed_ratio_0.83_filtered_wiki.valid.src_gpt-4_examples_0_temp_0.3_chain_False/calibration_1/maxdepdepth_-1_maxdeplength_-1_diffwordscount_-1_avgwordcount_-1_length_-1_leven_-1/ratio_stats.csv \
#     --predicted_ratio_feature_name 'MaxDepDepth_ratio'

# python llm_based_control_rewrite/scores/calculate_success_rate.py \
#     --predicted_ratio_file experiments/data_filtered_regression_model/linear_regression/2_maxdepdepth/wikilarge/valid/ratio_stats_with_LR_predicted_ratio.csv \
#     --obtained_ratio_file experiments/data_filtered_regression_model/linear_regression/2_maxdepdepth/wikilarge/valid/gpt4_zs/prompt_1_temp_0/maxdepdepth_-1_maxdeplength_-1_diffwordscount_-1_avgwordcount_-1_length_-1_leven_-1/ratio_stats.csv\
#     --predicted_ratio_feature_name 'MaxDepDepth_ratio' \


# python llm_based_control_rewrite/scores/calculate_success_rate.py \
# --predicted_ratio_file experiments/data_filtered_regression_model/linear_regression/2_maxdepdepth/wikilarge/valid/ratio_stats_with_LR_predicted_ratio.csv \
# --obtained_ratio_file experiments/data_filtered_regression_model/linear_regression/2_maxdepdepth/wikilarge/valid/gpt4_zs/prompt_1_temp_0_maxdepdeth_ratio_0.82/maxdepdepth_0.82_maxdeplength_-1_diffwordscount_-1_avgwordcount_-1_length_-1_leven_-1/ratio_stats.csv \
# --predicted_ratio_feature_name 'MaxDepDepth_ratio' > experiments/data_filtered_regression_model/linear_regression/2_maxdepdepth/wikilarge/valid/gpt4_zs/prompt_1_temp_0_maxdepdeth_ratio_0.82/logs_success_rate
