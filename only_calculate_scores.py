"""
THIS IS A STAND-ALONE SCRIPT
This script takes command line arguments.
"""
import os
import argparse

from llm_based_control_rewrite.utils.helpers import load_yaml, map_control_token_values
from llm_based_control_rewrite.feature_value_calculation.ratio_calculation import calculate_ratios, calculate_ratios_for_grade
from llm_based_control_rewrite.scores.calculate_success_rate import calculate_RatioSuccess_rate, \
    calculate_RatioSuccess_rate_fixed_ratio, \
    calculate_abs_tgt_value_metrics_for_lr, calculate_abs_tgt_value_metrics_for_fr, \
    calculate_RatioSuccess_rate_gold_ref, calculate_abs_tgt_metrics_for_gold_ref, \
    calculate_abs_tgt_value_metrics_for_grade, calculate_RatioSuccess_rate_for_Grade
from llm_based_control_rewrite.scores.easse import calculate_easse, calculate_easse_on_mac
from llm_based_control_rewrite.scores.sentence_level_readability_scores import \
    calculate_readability_scores_doc_level, calculate_readability_scores_sentence_level
from llm_based_control_rewrite.utils.helpers import write_final_test_stats, \
    write_success_rate_test_stats, write_easse_stats
from llm_based_control_rewrite.evaluation.feature_based_evaluation import calcuate_feature_values

def calculate_only_scores(**kwargs):
    config_yaml = load_yaml(kwargs["config"])
    requested_feature_dict_vals, requested_absolute_value, output_folder_name = map_control_token_values(kwargs)
    print("Start calculate scores!")
    output_generation_path = config_yaml["path_to_output_generation"] + "/" + output_folder_name
    if not os.path.exists(output_generation_path):
        os.makedirs(output_generation_path)
    generate_scores(config_yaml, requested_feature_dict_vals, output_generation_path,
                         kwargs["predicted_ratio_file_given"] if kwargs["predicted_ratio_file_given"] is not None else False)


def generate_scores(config_yaml, requested_feature_dict_vals, output_generation_path,
                         predicted_ratio_file_given=False):

    path_to_write_output_generation = output_generation_path + "/output.txt"
    path_to_write_input_lines_of_outputs = output_generation_path + "/input.txt"
    path_to_gold_ref = output_generation_path + "/gold_ref.txt"
    # Write final results stats file path
    path_to_final_results_stats_file = config_yaml["path_to_output_generation"] + "/output_stats.csv"
    print(f"predicted_ratio_file_given is: {predicted_ratio_file_given}")
    count_examples_NOT_found_cases = 0

    doc_level_readability_scores = calculate_readability_scores_doc_level(path_to_write_output_generation,
                                                                          config_yaml["path_to_output_generation"])
    sent_level_readability_scores = calculate_readability_scores_sentence_level(
        input_sent_file=path_to_write_output_generation,
        readability_scores_file_path=output_generation_path,
        avg_score_file_path=config_yaml["path_to_output_generation"],
        total_line_number=None)

    # evaluate the output Always calculate absolute feature values, so it is easy to generate ratio csv.
    average_feature_value_dict_of_output, output_feature_value_path = calcuate_feature_values(config_yaml["lang"].lower(),
                                                         in_src_PATH = path_to_write_input_lines_of_outputs,
                                                         in_tgt_PATH = path_to_write_output_generation,
                                                         actual_feature_value_file_path = output_generation_path,
                                                         analyze_features = config_yaml["analyze_features"],
                                                         requested_dependency_depth = requested_feature_dict_vals["MaxDepDepth"],
                                                         requested_dependency_length = requested_feature_dict_vals["MaxDepLength"],
                                                         requested_difficult_words = requested_feature_dict_vals["DiffWords"],
                                                         requested_word_count= requested_feature_dict_vals["WordCount"],
                                                         # requested_frequency= -1,  # requested_feature_dict_vals["FreqRank"]
                                                         # requested_length = requested_feature_dict_vals["Length"],
                                                         # requested_levenshtein = requested_feature_dict_vals["Leven"],
                                                         # requested_grade_level=requested_feature_dict_vals["Grade"],
                                                         absolute = True, output_file_prefix="output")
    # evaluate Input.txt feature value.
    average_feature_value_dict_of_input, input_feature_value_path = calcuate_feature_values(config_yaml["lang"].lower(),
                                                         in_src_PATH = path_to_write_output_generation,
                                                         in_tgt_PATH = path_to_write_input_lines_of_outputs,
                                                         actual_feature_value_file_path = output_generation_path,
                                                         analyze_features = config_yaml["analyze_features"],
                                                         requested_dependency_depth= requested_feature_dict_vals["MaxDepDepth"],
                                                         requested_dependency_length= requested_feature_dict_vals["MaxDepLength"],
                                                         requested_difficult_words=requested_feature_dict_vals["DiffWords"],
                                                         requested_word_count=requested_feature_dict_vals["WordCount"],
                                                         # requested_frequency=-1, # requested_feature_dict_vals["FreqRank"]
                                                         # requested_length= requested_feature_dict_vals["Length"],
                                                         # requested_levenshtein= requested_feature_dict_vals[ "Leven"],
                                                         # requested_grade_level=requested_feature_dict_vals["Grade"],
                                                         absolute=True,output_file_prefix="input")

    avg_feature_dict_ratio = calculate_ratios(input_feature_value_path, output_feature_value_path, output_generation_path)



    # # evaluate the output Always calculate absolute feature values, so it is easy to generate ratio csv.
    # average_feature_value_dict_of_output, output_feature_value_path = calcuate_feature_values(config_yaml["lang"].lower(),
    #                                                      in_src_PATH = path_to_write_input_lines_of_outputs,
    #                                                      in_tgt_PATH = path_to_write_output_generation,
    #                                                      actual_feature_value_file_path = output_generation_path,
    #                                                      analyze_features = config_yaml["analyze_features"],
    #                                                      requested_grade_level=requested_feature_dict_vals["Grade"],
    #                                                      absolute = True, output_file_prefix="output")
    # # evaluate Input.txt feature value.
    # average_feature_value_dict_of_input, input_feature_value_path = calcuate_feature_values(config_yaml["lang"].lower(),
    #                                                      in_src_PATH = path_to_write_output_generation,
    #                                                      in_tgt_PATH = path_to_write_input_lines_of_outputs,
    #                                                      actual_feature_value_file_path = output_generation_path,
    #                                                      analyze_features = config_yaml["analyze_features"],
    #                                                      requested_grade_level=requested_feature_dict_vals["Grade"],
    #                                                      absolute=True,output_file_prefix="input")
    #
    # avg_feature_dict_ratio = calculate_ratios_for_grade(input_feature_value_path, output_feature_value_path, output_generation_path)





    if predicted_ratio_file_given:
        abs_tgt_success_rate_exact_match_and_mse = calculate_abs_tgt_value_metrics_for_lr(
            predicted_ratio_files=config_yaml["predicted_ratio_file_path"],
            obtained_ratio_file=output_generation_path + "/ratio_stats.csv",
            feature_names=config_yaml["feature_names"],
            success_rate_type="exact_match",
            feature_range=[0, 0, 0, 0] if "feature_range" not in config_yaml else config_yaml["feature_range"].split(','),
            default_input_src=config_yaml["path_to_input_test_data"],
            tested_input_src=path_to_write_input_lines_of_outputs,
            default_ref_tgt=config_yaml["path_to_test_gold_ref_data"],
            tested_ref_tgt=path_to_gold_ref,
            output_generation_path=output_generation_path)

        abs_tgt_success_rate_equal_or_lessthan = calculate_abs_tgt_value_metrics_for_lr(
            predicted_ratio_files=config_yaml["predicted_ratio_file_path"],
            obtained_ratio_file=output_generation_path + "/ratio_stats.csv",
            feature_names=config_yaml["feature_names"],
            success_rate_type="equal_or_lessthan",
            feature_range=[0, 0, 0, 0] if "feature_range" not in config_yaml else config_yaml["feature_range"].split(','),
            default_input_src=config_yaml["path_to_input_test_data"],
            tested_input_src=path_to_write_input_lines_of_outputs,
            default_ref_tgt=config_yaml["path_to_test_gold_ref_data"],
            tested_ref_tgt=path_to_gold_ref,
            output_generation_path=output_generation_path)

        #  ratio_success_rate
        success_rate_exact_match = calculate_RatioSuccess_rate(
            predicted_ratio_files=config_yaml["predicted_ratio_file_path"],
            obtained_ratio_file=output_generation_path + "/ratio_stats.csv",
            feature_names=config_yaml["feature_names"],
            success_rate_type="exact_match",
            default_input_src=config_yaml["path_to_input_test_data"],
            tested_input_src=path_to_write_input_lines_of_outputs,
            default_ref_tgt=config_yaml["path_to_test_gold_ref_data"],
            tested_ref_tgt=path_to_gold_ref,
            output_generation_path=output_generation_path)
        success_rate_equal_or_lessthan = calculate_RatioSuccess_rate(
            predicted_ratio_files=config_yaml["predicted_ratio_file_path"],
            obtained_ratio_file=output_generation_path + "/ratio_stats.csv",
            feature_names=config_yaml["feature_names"],
            success_rate_type="equal_or_lessthan",
            default_input_src=config_yaml["path_to_input_test_data"],
            tested_input_src=path_to_write_input_lines_of_outputs,
            default_ref_tgt=config_yaml["path_to_test_gold_ref_data"],
            tested_ref_tgt=path_to_gold_ref,
            output_generation_path=output_generation_path)

    elif "gold_ratio_file_given"  in config_yaml:
        new_feature_list = config_yaml["feature_names"] + ",Grade"
        print(f"new_feature_list: {new_feature_list}")
        abs_tgt_success_rate_exact_match_and_mse = calculate_abs_tgt_metrics_for_gold_ref(
            gold_ref_file=config_yaml["gold_ratio_file_given"],
            obtained_ratio_file=output_generation_path + "/ratio_stats.csv",
            feature_names=config_yaml["feature_names"],
            success_rate_type="exact_match",
            feature_range=[0, 0, 0, 0] if "feature_range" not in config_yaml else config_yaml["feature_range"].split(','),
            default_input_src=config_yaml["path_to_input_test_data"],
            tested_input_src=path_to_write_input_lines_of_outputs,
            default_ref_tgt=config_yaml["path_to_test_gold_ref_data"],
            tested_ref_tgt=path_to_gold_ref,
            output_generation_path=output_generation_path)

        abs_tgt_success_rate_equal_or_lessthan = calculate_abs_tgt_metrics_for_gold_ref(
            gold_ref_file=config_yaml["gold_ratio_file_given"],
            obtained_ratio_file=output_generation_path + "/ratio_stats.csv",
            feature_names=config_yaml["feature_names"],
            success_rate_type="equal_or_lessthan",
            feature_range=[0, 0, 0, 0] if "feature_range" not in config_yaml else config_yaml["feature_range"].split(','),
            default_input_src=config_yaml["path_to_input_test_data"],
            tested_input_src=path_to_write_input_lines_of_outputs,
            default_ref_tgt=config_yaml["path_to_test_gold_ref_data"],
            tested_ref_tgt=path_to_gold_ref,
            output_generation_path=output_generation_path)

        #  ratio_success_rate
        success_rate_exact_match = calculate_RatioSuccess_rate_gold_ref(
            gold_ref_file=config_yaml["gold_ratio_file_given"],
            obtained_ratio_file=output_generation_path + "/ratio_stats.csv",
            feature_names=config_yaml["feature_names"],
            success_rate_type="exact_match",
            default_input_src=config_yaml["path_to_input_test_data"],
            tested_input_src=path_to_write_input_lines_of_outputs,
            default_ref_tgt=config_yaml["path_to_test_gold_ref_data"],
            tested_ref_tgt=path_to_gold_ref,
            output_generation_path=output_generation_path)
        success_rate_equal_or_lessthan = calculate_RatioSuccess_rate_gold_ref(
            gold_ref_file=config_yaml["gold_ratio_file_given"],
            obtained_ratio_file=output_generation_path + "/ratio_stats.csv",
            feature_names=config_yaml["feature_names"],
            success_rate_type="equal_or_lessthan",
            default_input_src=config_yaml["path_to_input_test_data"],
            tested_input_src=path_to_write_input_lines_of_outputs,
            default_ref_tgt=config_yaml["path_to_test_gold_ref_data"],
            tested_ref_tgt=path_to_gold_ref,
            output_generation_path=output_generation_path)

    elif "" in config_yaml["feature_names"].split(","):
        print("There are empty strings in the feature names.")
        abs_tgt_success_rate_exact_match_and_mse=""
        abs_tgt_success_rate_equal_or_lessthan=""
        success_rate_exact_match=""
        success_rate_equal_or_lessthan=""

    else:
        # because with calibration gpt_chat_model (GPT) object's feature variables are updated.
        fixed_feature_dict = {"MaxDepDepth": requested_feature_dict_vals["MaxDepDepth"], "MaxDepLength": requested_feature_dict_vals["MaxDepLength"],
                              "DiffWords": requested_feature_dict_vals["DiffWords"], "WordCount": requested_feature_dict_vals["WordCount"]}
        abs_tgt_success_rate_exact_match_and_mse = calculate_abs_tgt_value_metrics_for_fr(
            predicted_fixed_ratio_dict=fixed_feature_dict,
            obtained_ratio_file=output_generation_path + "/ratio_stats.csv",
            feature_names=config_yaml["feature_names"],
            success_rate_type="exact_match",
            feature_range=[0, 0, 0, 0] if "feature_range" not in config_yaml else config_yaml["feature_range"].split(','))

        abs_tgt_success_rate_equal_or_lessthan = calculate_abs_tgt_value_metrics_for_fr(
            predicted_fixed_ratio_dict=fixed_feature_dict,
            obtained_ratio_file=output_generation_path + "/ratio_stats.csv",
            feature_names=config_yaml["feature_names"],
            success_rate_type="equal_or_lessthan",
            feature_range=[0, 0, 0, 0] if "feature_range" not in config_yaml else config_yaml["feature_range"].split(','))

        #  ratio_success_rate
        success_rate_exact_match = calculate_RatioSuccess_rate_fixed_ratio(
            predicted_fixed_ratio_dict=fixed_feature_dict,
            obtained_ratio_file=output_generation_path + "/ratio_stats.csv",
            feature_names=config_yaml["feature_names"],
            success_rate_type="exact_match")
        success_rate_equal_or_lessthan = calculate_RatioSuccess_rate_fixed_ratio(
            predicted_fixed_ratio_dict=fixed_feature_dict,
            obtained_ratio_file=output_generation_path + "/ratio_stats.csv",
            feature_names=config_yaml["feature_names"],
            success_rate_type="equal_or_lessthan")

    # # Grade
    # # abs_tgt_success_rate_exact_match_and_mse = ""
    # # abs_tgt_success_rate_equal_or_lessthan = ""
    # # success_rate_exact_match = ""
    # # success_rate_equal_or_lessthan = ""
    # fixed_feature_dict = {"Grade": requested_feature_dict_vals["Grade"]}
    # abs_tgt_success_rate_exact_match_and_mse += ", " + calculate_abs_tgt_value_metrics_for_grade(
    #     predicted_fixed_ratio_dict=fixed_feature_dict,
    #     obtained_ratio_file=output_generation_path + "/ratio_stats.csv",
    #     feature_names="Grade",
    #     success_rate_type="exact_match",
    #     feature_range=[1])
    #
    # abs_tgt_success_rate_equal_or_lessthan += ", " + calculate_abs_tgt_value_metrics_for_grade(
    #     predicted_fixed_ratio_dict=fixed_feature_dict,
    #     obtained_ratio_file=output_generation_path + "/ratio_stats.csv",
    #     feature_names="Grade",
    #     success_rate_type="equal_or_lessthan",
    #     feature_range=[1])
    #
    # #  ratio_success_rate
    # success_rate_exact_match += ", " + calculate_RatioSuccess_rate_for_Grade(
    #     predicted_fixed_ratio_dict=fixed_feature_dict,
    #     obtained_ratio_file=output_generation_path + "/ratio_stats.csv",
    #     feature_names="Grade",
    #     success_rate_type="exact_match")
    # success_rate_equal_or_lessthan += ", " + calculate_RatioSuccess_rate_for_Grade(
    #     predicted_fixed_ratio_dict=fixed_feature_dict,
    #     obtained_ratio_file=output_generation_path + "/ratio_stats.csv",
    #     feature_names="Grade",
    #     success_rate_type="equal_or_lessthan")


    final_success_rate = write_success_rate_test_stats(config_yaml["path_to_output_generation"] + "/success_rate.csv",
                                  abs_tgt_success_rate_exact_match_and_mse, abs_tgt_success_rate_equal_or_lessthan,
                                  success_rate_exact_match, success_rate_equal_or_lessthan, output_generation_path)

    easse_output = calculate_easse(path_to_orig_sents=path_to_write_input_lines_of_outputs,
                                   path_to_refs_sents=path_to_gold_ref,
                                   path_to_sys_sents=path_to_write_output_generation)

    # easse_output = calculate_easse_on_mac(path_to_orig_sents=path_to_write_input_lines_of_outputs,
    #                                path_to_refs_sents=path_to_gold_ref,
    #                                path_to_sys_sents=path_to_write_output_generation)
    easse_output_final_write = write_easse_stats(config_yaml["path_to_output_generation"] + "/easse_stats.csv", easse_output, output_generation_path)

    write_final_test_stats(requested_feature_dict_vals, average_feature_value_dict_of_input,
                           average_feature_value_dict_of_output,
                           avg_feature_dict_ratio, "doc_level_readability_scores", count_examples_NOT_found_cases,
                           "messages", path_to_final_results_stats_file, "sent_level_readability_scores",
                           final_success_rate,
                           "easse_output_final_write")

    # write_final_test_stats(requested_feature_dict_vals, "average_feature_value_dict_of_input",
    #                        "average_feature_value_dict_of_output",
    #                        "avg_feature_dict_ratio", doc_level_readability_scores, count_examples_NOT_found_cases,
    #                        "messages", path_to_final_results_stats_file, "sent_level_readability_scores",
    #                        "final_success_rate", "easse_output_final_write")


    # lens_score(complex_file_path=path_to_write_input_lines_of_outputs, simple_file_path=path_to_write_output_generation, ref_file_path=path_to_gold_ref,
    #            model_path="data_auxiliary/en/lens/LENS/checkpoints/epoch=5-step=6102.ckpt")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="yaml config file for prompt based generation")
    parser.add_argument("--requested_dependency_depth", required=False, help="Target sentence's dependency depth")
    parser.add_argument("--requested_dependency_length", required=False, help="Target sentence's dependency length")
    parser.add_argument("--requested_difficult_words", required=False, help="Target sentence's no of difficult words")
    parser.add_argument("--requested_frequency", required=False, help="Target sentence's frequency")
    parser.add_argument("--requested_length", required=False, help="Target sentence's character length")
    parser.add_argument("--requested_levenshtein", required=False, help="Target sentence's levenshtein")
    parser.add_argument("--requested_word_count", required=False, help="Target sentence's word count")
    parser.add_argument("--requested_grade_level", required=False, help="Target sentence's word count")
    parser.add_argument("--requested_absolute_value", required=False, help="Default requested feature values are in "
                                                                           "ratio, make this param true, if you want to"
                                                                           "treat feature value as it is")
    parser.add_argument("--predicted_ratio_file_given", action="store_true", required=False, help="Predicted ratio using linear regression file is given")
    # parser.add_argument("--requested_frequency_category", required=False, help="Target sentence's frequency")
    args = vars(parser.parse_args())
    calculate_only_scores(**args)
