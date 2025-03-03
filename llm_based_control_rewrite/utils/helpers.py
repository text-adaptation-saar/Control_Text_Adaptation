from contextlib import contextmanager
from pathlib import Path
from itertools import zip_longest

import pandas as pd
import yaml
import re

feature2spec_token = {"dependency_depth": "MaxDepDepth", "frequency": "FreqRank", "length": "Length", "levenshtein": "Leven"}

@contextmanager
def open_files(filepaths, mode='r'):
    # pass a list of filenames as arguments and yield a list of open file objects
    # this function is useful also for readily preprocessed files that have text features
    files = []
    try:
        files = [Path(filepath).open(mode, encoding="utf-8") for filepath in filepaths]
        yield files
    finally:
        [f.close() for f in files]


def yield_lines_in_parallel(filepaths, strip=True, strict=True, n_lines=float('inf')):
    # read in files (meant for 2: source and target) in parallel line by line and yield tuples of parallel strings
    # this function is useful also for readily preprocessed files that have text features
    assert type(filepaths) == list
    with open_files(filepaths) as files:
        for i, parallel_lines in enumerate(zip_longest(*files)):
            if i >= n_lines:
                break
            if None in parallel_lines:
                assert not strict, f'Files don\'t have the same number of lines: {filepaths}, use strict=False'
            if strip:
                parallel_lines = [line.strip() if line is not None else None for line in parallel_lines]
            yield parallel_lines

def load_yaml(file_path):
    file_path = file_path.strip()
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def read_content (file):
    with open(file, 'r') as prompt_file:
        lines = prompt_file.readlines()
        prompt=""
        for line in lines:
            prompt += line
        print(prompt)
    return prompt

def read_csv(csv_file, level_value, column_name):
    df = pd.read_csv(csv_file)
    prompt = ""

    # Filter the DataFrame for the given level_value
    filtered_df = df[df['Level'] == level_value]

    if not filtered_df.empty:
        prompt_value = filtered_df[column_name].values[0]
        if pd.isna(prompt_value):
            print(f"{column_name}: No prompt available")
        else:
            prompt = prompt_value.strip()
            print(f"{column_name}: {prompt}")
    else:
        print("Level value not found")

    return prompt
def write_test_prompt_details(src, found_example_count, final_prompt, output, output_generation_path):
    path_to_write_word_ratio_values = output_generation_path + "/input_output_prompt_details.csv"
    # Because prompt contains comma, I use \t as seperator for csv.
    with open(path_to_write_word_ratio_values, 'a') as fp:
        final_line_to_write = "input_src: " + str(src.strip()) + "\t" + \
                              "output: " + str(output.strip()) + "\t" + \
                              "number_of_examples_found: " + str(found_example_count) + "\t" + \
                              "prompt: " + str(final_prompt) + "\n"
        fp.write(final_line_to_write)


def write_final_test_stats(requested_feature_dict_vals, average_feature_value_dict_of_input, average_feature_value_dict_of_output,
                           avg_feature_dict_ratio,
                           doc_level_readability_scores, count_examples_NOT_found_cases, final_prompt_to_write_in_file,
                           path_to_final_results_stats_file, sent_level_readability_scores, final_success_rate, easse_output_final_write):
                           # success_rate_exact_match, success_rate_equal_or_lessthan):
    # requested_feature_dict_vals, average_feature_value_dict_of_input, average_feature_value_dict_of_output,
    # avg_feature_dict_ratio, doc_level_readability_scores, count_examples_NOT_found_cases,
    # "messages", path_to_final_results_stats_file, sent_level_readability_scores,
    # success_rate_exact_match, success_rate_equal_or_lessthan


    with open(path_to_final_results_stats_file, 'a') as fp:
        final_line_to_write = ", ".join("requested_" + k + ", " + str(v) for k, v in requested_feature_dict_vals.items()) + ", " \
                              + ", ".join("input_avg_" + k + ", " + str(v) for k, v in average_feature_value_dict_of_input.items()) + ", " \
                              + ", ".join("output_avg_" + k + ", " + str(v) for k, v in average_feature_value_dict_of_output.items()) + ", " \
                              + ", ".join(k + ", " + str(v) for k, v in avg_feature_dict_ratio.items()) + ", " \
                              + doc_level_readability_scores.strip() + ", " \
                              + "count_examples_NOT_found_cases, " + str(count_examples_NOT_found_cases) + ", " \
                              + sent_level_readability_scores.strip() +  ", " \
                              + final_success_rate.strip() + ", " \
                              + easse_output_final_write.strip() \
                              + "\n"
                                # + success_rate_exact_match + ", " \
                                # + success_rate_equal_or_lessthan + ", " \
                                # + "prompt: " + str(final_prompt_to_write_in_file) + "\t" \

        fp.write(final_line_to_write)

def write_success_rate_test_stats(path_to_final_results_stats_file,
                                  abs_tgt_success_rate_exact_match_and_mse, abs_tgt_success_rate_equal_or_lessthan,
                                  success_rate_exact_match, success_rate_equal_or_lessthan, experiment_path):

    with open(path_to_final_results_stats_file, 'a') as fp:
        final_line_to_write = abs_tgt_success_rate_exact_match_and_mse   + ", " \
                              + abs_tgt_success_rate_equal_or_lessthan + ", " \
                              + success_rate_exact_match + ", " \
                              + success_rate_equal_or_lessthan \
                              + ", experiment_path, " + experiment_path \
                              + "\n"
                             # + easse_output \

        fp.write(final_line_to_write)
    return final_line_to_write

def write_easse_stats(path_to_final_results_stats_file, easse_output, experiment_path):

    with open(path_to_final_results_stats_file, 'a') as fp:
        final_line_to_write = easse_output \
                              + ", experiment_path, " + experiment_path \
                              + "\n"

        fp.write(final_line_to_write)
    return final_line_to_write

def map_control_token_values(kwargs):
    requested_feature_dict_vals = {}
    output_folder_name = ""
    if kwargs["requested_dependency_depth"]:
        requested_feature_dict_vals["MaxDepDepth"] = float(kwargs["requested_dependency_depth"])
        output_folder_name += "maxdepdepth_" + str(kwargs["requested_dependency_depth"]) + "_"
    if kwargs["requested_dependency_length"]:
        requested_feature_dict_vals["MaxDepLength"] = float(kwargs["requested_dependency_length"])
        output_folder_name += "maxdeplength_" + str(kwargs["requested_dependency_length"]) + "_"
    if kwargs["requested_difficult_words"]:
            requested_feature_dict_vals["DiffWords"] = float(kwargs["requested_difficult_words"])
            output_folder_name += "diffwordscount_" + str(kwargs["requested_difficult_words"]) + "_"
    if kwargs["requested_word_count"]:
        requested_feature_dict_vals["WordCount"] = float(kwargs["requested_word_count"])
        output_folder_name += "avgwordcount_" + str(kwargs["requested_word_count"]) + "_"
    if kwargs["requested_frequency"]:
        requested_feature_dict_vals["FreqRank"] = float(kwargs["requested_frequency"])
        output_folder_name += "freqrank_" + str(kwargs["requested_frequency"]) + "_"
    if kwargs["requested_length"]:
        requested_feature_dict_vals["Length"] = float(kwargs["requested_length"])
        output_folder_name += "length_" + str(kwargs["requested_length"]) + "_"
    if kwargs["requested_levenshtein"]:
        requested_feature_dict_vals["Leven"] = float(kwargs["requested_levenshtein"])
        output_folder_name += "leven_" + str(kwargs["requested_levenshtein"]) + "_"

    if kwargs["requested_grade_level"]:
        requested_feature_dict_vals["Grade"] = float(kwargs["requested_grade_level"])
        output_folder_name += "grade_" + str(kwargs["requested_grade_level"])

    requested_absolute_value = kwargs["requested_absolute_value"] if kwargs["requested_absolute_value"] is not None else False
    print(f"Mapped requested feature params with the values and requested_absolute_value is: {requested_absolute_value}")
    print(f"Mapped requested feature params: {requested_feature_dict_vals}")
    return requested_feature_dict_vals, requested_absolute_value, output_folder_name


def extract_rewritten_sentences(final_out):
    # This pattern matches non-empty sequences inside curly braces
    pattern = r"\{([^}]+)\}"
    matches = re.findall(pattern, final_out)

    # Checking how many {} pairs are consecutively at the end
    end_pattern = r"(\{[^}]+\}\s*)+$"
    end_match = re.search(end_pattern, final_out)
    if end_match:
        # Counting the number of ending curly braces
        end_blocks = re.findall(pattern, end_match.group())
        # Join the end_blocks into a single string separated by a specified separator
        return " ".join(end_blocks)  # Join with a space or any other preferred separator
    else:
        # This pattern matches non-empty sequences inside curly braces
        pattern = r"\{([^}]+)\}"
        matches = re.findall(pattern, final_out)

        # Select the last match if there are any matches
        if matches:
            rewritten_sentence = matches[-1]  # Last match
        else:
            rewritten_sentence = ""
            # rewritten_sentence = extract_rewritten_sentences_from_double_quotes(final_out)
            # rewritten_sentence = final_out

        return rewritten_sentence

    # # This pattern matches non-empty sequences inside curly braces
    # pattern = r"\{([^}]+)\}"
    # matches = re.findall(pattern, final_out)
    #
    # # Select the last match if there are any matches
    # if matches:
    #     rewritten_sentence = matches[-1]  # Last match
    # else:
    #     rewritten_sentence = ""
    #     # rewritten_sentence = final_out
    #
    # return rewritten_sentence
# --***

    # pattern = r"Rewritten sentence\(s\): {(.+)}"
    # pattern_2 = r"rewritten sentence\(s\): {(.+?)}"
    # match = re.search(pattern, final_out)
    # match_2 = re.search(pattern_2, final_out)
    # # Extract the rewritten sentence if a match is found
    #
    # if match:
    #     rewritten_sentence = match.group(1)
    # elif match_2:
    #     rewritten_sentence = match_2.group(1)
    # else:
    #     rewritten_sentence = final_out
    # return rewritten_sentence

    # if "Rewritten sentence(s):" in final_out:
    #     # Extracting the rewritten sentence(s)
    #     start_phrase = "Rewritten sentence(s): "
    #     end_phrase = " (1) Rewritten sentence's maximum dependency depth ="
    #     # Find the start and end indexes
    #     start_idx = final_out.find(start_phrase) + len(start_phrase)
    #     end_idx = final_out.find(end_phrase)
    #     if end_idx == -1:
    #         end_phrase = " Rewritten sentence's maximum dependency depth ="
    #         end_idx = final_out.find(end_phrase)
    #
    #         # Extract the sentence
    #     rewritten_sentences = final_out[start_idx:end_idx].strip()
    #     return rewritten_sentences
    # else:
    #     return final_out


def extract_rewritten_sentences_from_double_quotes(final_out):
    # This pattern matches text inside double quotes
    pattern = r'\"([^\"]+)\"'
    matches = re.findall(pattern, final_out)

    # Select the first match if there are any matches
    if matches:
        extracted_sentence = matches[0]  # First match
    else:
        extracted_sentence = ""

    return extracted_sentence