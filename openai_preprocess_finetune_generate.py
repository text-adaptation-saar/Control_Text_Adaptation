"""
THIS IS A STAND-ALONE SCRIPT
This script takes command line arguments.
"""
import os
import argparse

from llm_based_control_rewrite.generate.llama.generate_llama import generate_with_prompt_for_llama
from llm_based_control_rewrite.utils.helpers import load_yaml, map_control_token_values
from llm_based_control_rewrite.generate.openai_gpt.generate import generate_with_prompt
import time
from datetime import datetime

def openai_preprocess_finetune_generate(**kwargs):
    config_yaml = load_yaml(kwargs["config"])
    requested_feature_dict_vals, requested_absolute_value, output_folder_name = map_control_token_values(kwargs)

    if  kwargs["do_train"]:
        print("There is no code for training!")

    if kwargs["do_eval"]:
        print("Start processing for inference!")
        output_generation_path = config_yaml["path_to_output_generation"] + "/" + output_folder_name
        if not os.path.exists(output_generation_path):
            os.makedirs(output_generation_path)

        generate_with_prompt(config_yaml, requested_feature_dict_vals, requested_absolute_value, output_generation_path,
                                 kwargs["predicted_ratio_file_given"] if kwargs["predicted_ratio_file_given"] is not None else False)


if __name__=="__main__":
    # Record the start time
    start_time = time.time()
    start_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Script started at: {start_timestamp}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="yaml config file for prompt based generation")
    parser.add_argument("--requested_dependency_depth", required=False, help="Target sentence's dependency depth")
    parser.add_argument("--requested_dependency_length", required=False, help="Target sentence's dependency length")
    parser.add_argument("--requested_difficult_words", required=False, help="Target sentence's no of difficult words")
    parser.add_argument("--requested_frequency", required=False, help="Target sentence's frequency")
    parser.add_argument("--requested_length", required=False, help="Target sentence's character length")
    parser.add_argument("--requested_levenshtein", required=False, help="Target sentence's levenshtein")
    parser.add_argument("--requested_word_count", required=False, help="Target sentence's word count")
    parser.add_argument("--requested_absolute_value", required=False, help="Default requested feature values are in "
                                                                           "ratio, make this param true, if you want to"
                                                                           "treat feature value as it is")
    parser.add_argument("--requested_grade_level", required=False, help="Target sentence's word count")
    parser.add_argument("--do_train", action="store_true", required=False, help="Request to conduct training")
    parser.add_argument("--do_eval", action="store_true", required=False, help="Request to conduct evaluation")
    parser.add_argument("--predicted_ratio_file_given", action="store_true", required=False, help="Predicted ratio using linear regression file is given")
    # parser.add_argument("--requested_frequency_category", required=False, help="Target sentence's frequency")
    args = vars(parser.parse_args())
    openai_preprocess_finetune_generate(**args)

    # Record the end time
    end_time = time.time()
    end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elapsed_time = end_time - start_time

    # Convert elapsed time to hours, minutes, and seconds
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)

    print(f"Script ended at: {end_timestamp}")
    print(f"Total runtime: {hours} hours, {minutes} minutes, {seconds} seconds")

# Example command:
# python openai_preprocess_finetune_generate.py --config experiments/with_meaning/gpt_openai/1_wc/gpt4_zs/prompt_based_generation_wc_only.yaml
# --requested_dependency_depth -1
# --requested_dependency_length -1
# --requested_frequency -1
# --requested_length -1
# --requested_levenshtein -1
# --requested_word_count 0.1
# --requested_absolute_value
# --do_eval
