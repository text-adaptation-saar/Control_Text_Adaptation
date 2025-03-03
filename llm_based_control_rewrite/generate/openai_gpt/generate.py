import pandas as pd
from llm_based_control_rewrite.feature_value_calculation.ratio_calculation import calculate_ratios
from llm_based_control_rewrite.generate.openai_gpt.batch_process import send_batch_request
from llm_based_control_rewrite.scores.calculate_success_rate import calculate_RatioSuccess_rate, \
    calculate_RatioSuccess_rate_fixed_ratio, \
    calculate_abs_tgt_value_metrics_for_lr, calculate_abs_tgt_value_metrics_for_fr, \
    calculate_RatioSuccess_rate_gold_ref, calculate_abs_tgt_metrics_for_gold_ref
from llm_based_control_rewrite.scores.easse import calculate_easse
from llm_based_control_rewrite.scores.sentence_level_readability_scores import \
    calculate_readability_scores_doc_level, calculate_readability_scores_sentence_level
from llm_based_control_rewrite.utils.feature_extraction import get_max_dep_depth_of_given_sent
from llm_based_control_rewrite.utils.helpers import read_content, write_test_prompt_details, write_final_test_stats, \
    write_success_rate_test_stats, write_easse_stats, read_csv
from llm_based_control_rewrite.models.GPT import GPT
from llm_based_control_rewrite.evaluation.feature_based_evaluation import calcuate_feature_values
import random

from only_calculate_scores import generate_scores


def generate_with_prompt(config_yaml, requested_feature_dict_vals, requested_absolute_value, output_generation_path,
                         predicted_ratio_file_given=False):
    prompt_csv = bool(config_yaml.get("prompt_csv", False))
    if prompt_csv:
        system_prompt = read_csv(csv_file=config_yaml["prompt_csv"],
                                     level_value=config_yaml["system_prompt"],
                                     column_name="system_prompt")
        user_prompt = read_csv(csv_file=config_yaml["prompt_csv"],
                                     level_value=config_yaml["user_prompt"],
                                     column_name="user_prompt")
        output_prompt = read_csv(csv_file=config_yaml["prompt_csv"],
                                     level_value=config_yaml["output_prompt"],
                                     column_name="output_prompt")
    else:
        system_prompt = read_content(config_yaml["system_prompt"])
        user_prompt = read_content(config_yaml["user_prompt"])
        output_prompt = "" if "output_prompt" not in config_yaml else read_content(config_yaml["output_prompt"])

    gpt_chat_model = GPT(requested_dependency_depth = requested_feature_dict_vals["MaxDepDepth"],
                         requested_dependency_length = requested_feature_dict_vals["MaxDepLength"],
                         requested_difficult_words = requested_feature_dict_vals["DiffWords"],
                         requested_word_count = requested_feature_dict_vals["WordCount"],
                         requested_length=requested_feature_dict_vals["Length"],
                         requested_levenshtein=requested_feature_dict_vals["Leven"],
                         requested_frequency= None if "FreqRank" not in requested_feature_dict_vals else requested_feature_dict_vals["FreqRank"],
                         model_path = config_yaml["model"],
                         temperature = config_yaml["temperature"],
                         system_prompt = system_prompt,
                         user_prompt = user_prompt,
                         output_prompt = output_prompt,
                         requested_absolute_feature_value = requested_absolute_value,
                         openai_key_name =  None if "openai_key_name" not in config_yaml else config_yaml["openai_key_name"],
                         requested_grade=None if "Grade" not in requested_feature_dict_vals else requested_feature_dict_vals["Grade"]
                         )

    path_to_write_output_generation = output_generation_path + "/output.txt"
    path_to_write_input_lines_of_outputs = output_generation_path + "/input.txt"
    path_to_gold_ref = output_generation_path + "/gold_ref.txt"
    # Write final results stats file path
    path_to_final_results_stats_file = config_yaml["path_to_output_generation"] + "/output_stats.csv"

    batch_request = bool(config_yaml.get("batch_request", False))
    print(f"BATCH request process: {batch_request}!")

    # chain_of_thought = False if "chain_of_thought" not in config_yaml else config_yaml["chain_of_thought"]
    print(f"predicted_ratio_file_given is: {predicted_ratio_file_given}")
    count_test = 0
    count_examples_NOT_found_cases = 0
    with open(config_yaml["path_to_input_test_data"], 'r') as test_input, \
            open(config_yaml["path_to_test_gold_ref_data"], 'r') as test_gold_ref:
        test_input_dataset = test_input.readlines()
        test_gold_ref_dataset = test_gold_ref.readlines()

        for line_number, (src_sentence, gold_ref) in enumerate(zip(test_input_dataset, test_gold_ref_dataset), start=1):
            print(f"Reading Line {line_number}: {src_sentence.strip()}")

            if count_test >= config_yaml["number_of_lines_to_test"]:
                break
            # #     skip short sentences
            # max_dep_depth_of_test_src = get_max_dep_depth_of_given_sent(src_sentence)
            # if max_dep_depth_of_test_src <= 6:
            #     print(f"Max. dependency depth is {max_dep_depth_of_test_src}, so skipping the line number {line_number}..")
            #     continue
            line_to_skip_in_index = int(config_yaml["line_to_skip"]) - 1 #line number from output.txt -1
            if count_test <= line_to_skip_in_index:
                print("Index: %s" % (count_test))
                count_test += 1
                continue
            count_test += 1
            print("**** Start OpenAI LLM inference for given dataset, test count: %s and reading line no:%s from test set ****" % (count_test, line_number))
            # update feature ratio/abs given value via command line, because when we do calibration we are updating GPT's class feature variables.
            gpt_chat_model.dependency_depth = requested_feature_dict_vals["MaxDepDepth"]
            gpt_chat_model.dependency_length= requested_feature_dict_vals["MaxDepLength"]
            gpt_chat_model.difficult_words = requested_feature_dict_vals["DiffWords"]
            gpt_chat_model.word_count = requested_feature_dict_vals["WordCount"]
            gpt_chat_model.length = requested_feature_dict_vals["Length"]
            gpt_chat_model.levenshtein = requested_feature_dict_vals["Leven"]
            gpt_chat_model.frequency = None if "FreqRank" not in requested_feature_dict_vals else requested_feature_dict_vals["FreqRank"]
            gpt_chat_model.grade = None if "Grade" not in requested_feature_dict_vals else requested_feature_dict_vals["Grade"]

            if predicted_ratio_file_given:
                assign_predicted_ratio_to_respective_feature_parameters(config_yaml, gpt_chat_model, line_number)

            if "gold_ratio_file_given"  in config_yaml:
                print("Conduct experiments by requesting feature values exactly as in GOLD reference: ")
                assign_abs_tgt_gold_to_respective_feature_parameters(config_yaml, gpt_chat_model, line_number)

            # if chain_of_thought:
            line_dict = {"line": count_test, "src": src_sentence.strip(), "gold_ref": gold_ref.strip()}
            output, messages, found_example_count = gpt_chat_model.generate_with_chain_of_thought(line_dict, config_yaml,
                                                                                                      path_to_write_output_generation,
                                                                                                      path_to_write_input_lines_of_outputs,
                                                                                                      path_to_gold_ref,
                                                                                                      output_generation_path
                                                                                                  )
            if output is None:
                count_examples_NOT_found_cases += 1
                continue

            # output evaluation wrt. each feature
            write_test_prompt_details(src_sentence, found_example_count, messages, output, output_generation_path)

            print("**** Finish OpenAI LLM inference for test count: %s and reading line no:%s from test set ****" % (count_test, line_number))

    if batch_request and not bool(config_yaml.get("chain_of_thought", False)):
        output_generation_path_array = output_generation_path.split("/")
        batch_request_jsonl_file = f"{output_generation_path}/batch_request_{output_generation_path_array[-2]}_{output_generation_path_array[-1]}.jsonl"
        batch_request_details_config = f'{config_yaml["path_to_output_generation"]}/grade_{requested_feature_dict_vals["Grade"]}-batch_request_details.json'
        send_batch_request(batch_request_jsonl_file, config_yaml, batch_request_details_config)
    # else:
        # generate_scores(config_yaml, requested_feature_dict_vals, output_generation_path,
        #                 predicted_ratio_file_given=predicted_ratio_file_given)


def assign_predicted_ratio_to_respective_feature_parameters(config_yaml, gpt_chat_model, line_number):
    for feature, predicted_Ratio_file in zip(config_yaml["feature_names"].split(','),
                                             config_yaml["predicted_ratio_file_path"].split(',')):
        print(f"feature: {feature}\t predicted_Ratio_file:{predicted_Ratio_file}")
        df = pd.read_csv(predicted_Ratio_file)
        # Specify the row and column index to read the value
        row_index = line_number - 1  # Row 3 (0-based index)
        # column_name = "predicted_"+feature+"_ratio" if "calibrated_feature_names" not in config_yaml else config_yaml["calibrated_feature_names"]
        column_name = "predicted_abs_tgt_" + feature
        # Access the specific value using .at[]
        value = df.at[row_index, column_name]
        print("predicted_ratio_file is given, reading row_index:%s, column_name:%s, value:%s, Line:%s" % (
        row_index, column_name, value, df.at[row_index, "Line"]))
        if 'WordCount' in column_name:
            gpt_chat_model.word_count = value
            # gpt_chat_model.word_count = randomize_ratio(value, seed=26)
            print("row_index:%s, column_name:%s, gpt_chat_model.word_count:%s, Line:%s" % (
                row_index, column_name, gpt_chat_model.word_count, df.at[row_index, "Line"]))
        if 'MaxDepDepth' in column_name:
            gpt_chat_model.dependency_depth = value
            # gpt_chat_model.dependency_depth = randomize_ratio(value, seed=26)
            print("row_index:%s, column_name:%s, gpt_chat_model.dependency_depth:%s, Line:%s" % (
                row_index, column_name, gpt_chat_model.dependency_depth, df.at[row_index, "Line"]))
        if 'MaxDepLength' in column_name:
            gpt_chat_model.dependency_length = value
            # gpt_chat_model.dependency_length = randomize_ratio(value, seed=26)
            print("row_index:%s, column_name:%s, gpt_chat_model.dependency_length:%s, Line:%s" % (
                row_index, column_name, gpt_chat_model.dependency_length, df.at[row_index, "Line"]))
        if 'DiffWords' in column_name:
            gpt_chat_model.difficult_words = value
            # gpt_chat_model.difficult_words = randomize_ratio(value, seed=26)
            print("row_index:%s, column_name:%s, gpt_chat_model.difficult_words:%s, Line:%s" % (
                row_index, column_name, gpt_chat_model.difficult_words, df.at[row_index, "Line"]))


def assign_abs_tgt_gold_to_respective_feature_parameters(config_yaml, gpt_chat_model, line_number):
    df = pd.read_csv(config_yaml["gold_ratio_file_given"])
    if not gpt_chat_model.requested_absolute_feature_value:
        print("setting requested_absolute_feature_value to True. since we do on GOLD ref.")
        gpt_chat_model.requested_absolute_feature_value= True

    for feature in config_yaml["feature_names"].split(','):
        # Specify the row and column index to read the value
        row_index = line_number - 1  # Row 3 (0-based index)
        column_name = "abs_tgt_" + feature
        # Access the specific value using .at[]
        value = df.at[row_index, column_name]
        print("gold_ratio_file is given, reading row_index:%s, column_name:%s, value:%s, Line:%s" % (
        row_index, column_name, value, df.at[row_index, "Line"]))
        if 'WordCount' in column_name:
            gpt_chat_model.word_count = value
            print("row_index:%s, column_name:%s, gpt_chat_model.word_count:%s, Line:%s" % (
                row_index, column_name, gpt_chat_model.word_count, df.at[row_index, "Line"]))
        if 'MaxDepDepth' in column_name:
            gpt_chat_model.dependency_depth = value
            print("row_index:%s, column_name:%s, gpt_chat_model.dependency_depth:%s, Line:%s" % (
                row_index, column_name, gpt_chat_model.dependency_depth, df.at[row_index, "Line"]))
        if 'MaxDepLength' in column_name:
            gpt_chat_model.dependency_length = value
            print("row_index:%s, column_name:%s, gpt_chat_model.dependency_length:%s, Line:%s" % (
                row_index, column_name, gpt_chat_model.dependency_length, df.at[row_index, "Line"]))
        if 'DiffWords' in column_name:
            gpt_chat_model.difficult_words = value
            print("row_index:%s, column_name:%s, gpt_chat_model.difficult_words:%s, Line:%s" % (
                row_index, column_name, gpt_chat_model.difficult_words, df.at[row_index, "Line"]))
        if 'Length' in column_name:
            gpt_chat_model.length = value
            print("row_index:%s, column_name:%s, gpt_chat_model.length:%s, Line:%s" % (
                row_index, column_name, gpt_chat_model.length, df.at[row_index, "Line"]))


        column_name_g = "abs_tgt_FKGL_Grade"
        value = df.at[row_index, column_name_g]
        gpt_chat_model.grade = value
        print("row_index:%s, column_name:%s, gpt_chat_model.grade:%s, Line:%s" % (
            row_index, column_name_g, gpt_chat_model.grade, df.at[row_index, "Line"]))



# Example command:
# python prompt_based_generate_with_single_feature.py --config experiments/1_wc/gpt4_zs/prompt_based_generation_wc_only.yaml
# --requested_dependency_length -1
# --requested_length -1
# --requested_levenshtein -1
# --requested_frequency -1
# --requested_dependency_depth -1
# --requested_word_count 0.1
