import os
import re
from llm_based_control_rewrite.utils.helpers import extract_rewritten_sentences
import argparse

def update_output_file(directory_path):
    """
    Updates empty lines in the output file using replacement text from the CSV file in the same directory.
    """
    # Extract the 'skip' value from the directory path
    skip_value_match = re.search(r'line_to_skip_(\d+)', directory_path)
    if skip_value_match:
        skip = int(skip_value_match.group(1))

    else:
        skip = 0  # Exit if no 'line_to_skip' is found

    # output_file_path = f"{directory_path}/maxdepdepth_-1_maxdeplength_-1_diffwordscount_-1_avgwordcount_-1_length_-1_leven_-1_grade_-1/output.txt"
    # input_file_path = f"{directory_path}/maxdepdepth_-1_maxdeplength_-1_diffwordscount_-1_avgwordcount_-1_length_-1_leven_-1_grade_-1/input.txt"
    # csv_file_path = f"{directory_path}/maxdepdepth_-1_maxdeplength_-1_diffwordscount_-1_avgwordcount_-1_length_-1_leven_-1_grade_-1/Feedback_loop_iternations.csv"

    output_file_path = f"{directory_path}/output.txt"
    input_file_path = f"{directory_path}/input.txt"
    csv_file_path = f"{directory_path}/Feedback_loop_iternations.csv"

    # Read the output file
    with open(output_file_path, "r") as output_file:
        output_lines = output_file.readlines()

    # Read the output file
    with open(input_file_path, "r") as input_file:
        input_lines = input_file.readlines()

    # Read the CSV file as a text file
    with open(csv_file_path, "r") as csv_file:
        csv_lines = csv_file.readlines()

    # Backup the original output file
    original_output = output_file_path.replace("output.txt", "output_original.txt")
    with open(original_output, "w") as original_output_fp:
        original_output_fp.writelines(output_lines)

    # Process and update empty lines in the output file
    updated_lines = []
    for line_number, (line, input_line) in enumerate(zip(output_lines, input_lines)):
        if line.strip() == "":  # Check for empty lines
            replacement_text = None

            # Loop through iterations, starting from 9 down to 1
            for iteration in range(9, 0, -1):
                # Use the skip value and the line number to construct the empty line identifier
                empty_line_identifier = f"line_number, {skip + line_number + 1}, iternation, {iteration},"
                matching_csv_line = next((l for l in csv_lines if l.startswith(empty_line_identifier)), None)

                if matching_csv_line:
                    replacement_text = extract_rewritten_sentences(matching_csv_line).strip()
                    if replacement_text:  # Use this line if rewritten text is not empty
                        break

            # Append the replacement text if found, or retain the empty line
            if replacement_text:
                updated_lines.append(replacement_text + "\n")
            else:
                print(f"No valid replacement found for line {line_number + 1}")
                # print(f"No valid replacement found for line {line_number + 1} in {output_file_path}, copying input sentence from input.txt")
                # updated_lines.append(input_line.strip() + "\n")  # Retain the empty line if no match is found
        else:
            updated_lines.append(line)

    # Write the updated content back to the output file
    with open(output_file_path, "w") as output_file:
        output_file.writelines(updated_lines)

    # print(f"Empty lines in {output_file_path} have been updated using the CSV file.")


def update_output_file_wo_feedback_iterations(directory_path):
    """
    Updates empty lines in the output file using replacement text from the CSV file in the same directory.
    """
    # Extract the 'skip' value from the directory path
    print(f"Processing without feedback loop: {directory}")

    output_file_path = f"{directory_path}/maxdepdepth_-1_maxdeplength_-1_diffwordscount_-1_avgwordcount_-1_length_-1_leven_-1_grade_-1/output.txt"
    input_file_path = f"{directory_path}/maxdepdepth_-1_maxdeplength_-1_diffwordscount_-1_avgwordcount_-1_length_-1_leven_-1_grade_-1/input.txt"
    csv_file_path = f"{directory_path}/maxdepdepth_-1_maxdeplength_-1_diffwordscount_-1_avgwordcount_-1_length_-1_leven_-1_grade_-1/full_output_response.txt"

    # Read the output file
    with open(output_file_path, "r") as output_file:
        output_lines = output_file.readlines()

    # Read the output file
    with open(input_file_path, "r") as input_file:
        input_lines = input_file.readlines()

    # Read the CSV file as a text file
    with open(csv_file_path, "r") as csv_file:
        csv_lines = csv_file.readlines()

    # Process and update empty lines in the output file
    updated_lines = []
    for line_number, (line, full_response_line, input_line) in enumerate(zip(output_lines, csv_lines, input_lines)):
        if line.strip() == "":  # Check for empty lines
            replacement_text = full_response_line.strip("{}").strip()

            # Append the replacement text if found, or retain the empty line
            if replacement_text:
                updated_lines.append(replacement_text + "\n")
            else:
                print(f"No valid replacement found for line {line_number + 1}, copying input sentence from input.txt")
                updated_lines.append(input_line.strip() + "\n")  # Retain the empty line if no match is found
        else:
            updated_lines.append(line.strip()+ "\n")

    # Write the updated content back to the output file
    with open(output_file_path, "w") as output_file:
        output_file.writelines(updated_lines)

    # print(f"Empty lines in {output_file_path} have been updated using the CSV file.")


if __name__ == "__main__":
    # Set up argument parsing
    # example:  "phase_2_experiments/fvp_with_synthetic_data_creation_methods/grade-9/cot_feedback_-grade_9_original_parallel_train.tgt-1000_llama_3_70b_instruct_sglang_examples_5_temp_0_chain_True_seed_184623_line_to_skip_-1",
    parser = argparse.ArgumentParser(description="Process a single experiment directory.")
    parser.add_argument(
        "--directory",
        required=True,
        help="Path to the experiment directory to process.",
    )
    parser.add_argument(
        "--feedback",
        type=lambda x: str(x).lower() in {"true", "1", "yes"},
        default=True,
        help="Specify if the feedback loop is present (true/false). Default is true.",
    )

    # Parse arguments
    args = parser.parse_args()
    directory = args.directory
    feedback = args.feedback

    # Process the directory based on the feedback argument
    if feedback:
        update_output_file(directory)
    else:
        update_output_file_wo_feedback_iterations(directory)

# if __name__=="__main__":
    # # List of directories containing the files
    # experiment_directories = [
    #     "phase_2_experiments/fvp_with_synthetic_data_creation_methods/grade-1/data_creation_method_4/syntactic_data_generation/llama_3_70b_instruct_sglang/cot_feedback_-grade_1_train_original_target.tgt-1000_llama_3_70b_instruct_sglang_examples_5_temp_0_chain_True_seed_184623_line_to_skip_-1",
    #     "phase_2_experiments/fvp_with_synthetic_data_creation_methods/grade-1/data_creation_method_4/syntactic_data_generation/llama_3_70b_instruct_sglang/cot_feedback_-grade_1_train_original_target.tgt-2000_llama_3_70b_instruct_sglang_examples_5_temp_0_chain_True_seed_184623_line_to_skip_999",
    #     "phase_2_experiments/fvp_with_synthetic_data_creation_methods/grade-1/data_creation_method_4/syntactic_data_generation/llama_3_70b_instruct_sglang/cot_feedback_-grade_1_train_original_target.tgt-3612_llama_3_70b_instruct_sglang_examples_5_temp_0_chain_True_seed_184623_line_to_skip_1999",
    #     "phase_2_experiments/fvp_with_synthetic_data_creation_methods/grade-1/data_creation_method_4/syntactic_data_generation/llama_3_70b_instruct_sglang/cot_feedback_-grade_1_train_original_target.tgt-3612_llama_3_70b_instruct_sglang_examples_5_temp_0_chain_True_seed_184623_line_to_skip_3512",
    #
    #     "phase_2_experiments/fvp_with_synthetic_data_creation_methods/grade-1/data_creation_method_4/syntactic_data_generation/llama_3_70b_instruct_sglang/cot_feedback_-grade_1_eval_original_target.tgt-250_llama_3_70b_instruct_sglang_examples_5_temp_0_chain_True_seed_184623_line_to_skip_-1",
    #     "phase_2_experiments/fvp_with_synthetic_data_creation_methods/grade-1/data_creation_method_4/syntactic_data_generation/llama_3_70b_instruct_sglang/cot_feedback_-grade_1_eval_original_target.tgt-500_llama_3_70b_instruct_sglang_examples_5_temp_0_chain_True_seed_184623_line_to_skip_250",
    #     "phase_2_experiments/fvp_with_synthetic_data_creation_methods/grade-1/data_creation_method_4/syntactic_data_generation/llama_3_70b_instruct_sglang/cot_feedback_-grade_1_eval_original_target.tgt-750_llama_3_70b_instruct_sglang_examples_5_temp_0_chain_True_seed_184623_line_to_skip_500",
    #
    # ]


    # # List of directories containing the files
    # main_path="phase_2_experiments/fvp_with_synthetic_data_creation_methods/grade-12/data_creation_method_4/syntactic_data_generation/llama_3_70b_instruct_sglang"
    # experiment_directories = [
    #     f"{main_path}/cot_feedback_-grade_12_original_parallel_train.tgt-1000_llama_3_70b_instruct_sglang_examples_5_temp_0_chain_True_seed_184623_line_to_skip_98",
    #     f"{main_path}/cot_feedback_-grade_12_original_parallel_train.tgt-1000_llama_3_70b_instruct_sglang_examples_5_temp_0_chain_True_seed_184623_line_to_skip_-1",
    #     f"{main_path}/cot_feedback_-grade_12_original_parallel_train.tgt-2000_llama_3_70b_instruct_sglang_examples_5_temp_0_chain_True_seed_184623_line_to_skip_1000",
    #     f"{main_path}/cot_feedback_-grade_12_original_parallel_train.tgt-2000_llama_3_70b_instruct_sglang_examples_5_temp_0_chain_True_seed_184623_line_to_skip_1095",
    #     f"{main_path}/cot_feedback_-grade_12_original_parallel_train.tgt-3000_llama_3_70b_instruct_sglang_examples_5_temp_0_chain_True_seed_184623_line_to_skip_2000",
    #     f"{main_path}/cot_feedback_-grade_12_original_parallel_train.tgt-3000_llama_3_70b_instruct_sglang_examples_5_temp_0_chain_True_seed_184623_line_to_skip_2089",
    #     f"{main_path}/cot_feedback_-grade_12_original_parallel_train.tgt-4000_llama_3_70b_instruct_sglang_examples_5_temp_0_chain_True_seed_184623_line_to_skip_3000",
    #     f"{main_path}/cot_feedback_-grade_12_original_parallel_train.tgt-4000_llama_3_70b_instruct_sglang_examples_5_temp_0_chain_True_seed_184623_line_to_skip_3095",
    #     f"{main_path}/cot_feedback_-grade_12_original_parallel_train.tgt-5000_llama_3_70b_instruct_sglang_examples_5_temp_0_chain_True_seed_184623_line_to_skip_4000",
    #     f"{main_path}/cot_feedback_-grade_12_original_parallel_train.tgt-5000_llama_3_70b_instruct_sglang_examples_5_temp_0_chain_True_seed_184623_line_to_skip_4096",
    #     f"{main_path}/cot_feedback_-grade_12_original_parallel_train.tgt-6000_llama_3_70b_instruct_sglang_examples_5_temp_0_chain_True_seed_184623_line_to_skip_5000",
    #     f"{main_path}/cot_feedback_-grade_12_original_parallel_train.tgt-6000_llama_3_70b_instruct_sglang_examples_5_temp_0_chain_True_seed_184623_line_to_skip_5097",
    #     f"{main_path}/cot_feedback_-grade_12_original_parallel_train.tgt-7000_llama_3_70b_instruct_sglang_examples_5_temp_0_chain_True_seed_184623_line_to_skip_6000",
    #     f"{main_path}/cot_feedback_-grade_12_original_parallel_train.tgt-7000_llama_3_70b_instruct_sglang_examples_5_temp_0_chain_True_seed_184623_line_to_skip_6089",
    #     f"{main_path}/cot_feedback_-grade_12_original_parallel_train.tgt-8000_llama_3_70b_instruct_sglang_examples_5_temp_0_chain_True_seed_184623_line_to_skip_7000",
    #     f"{main_path}/cot_feedback_-grade_12_original_parallel_train.tgt-8000_llama_3_70b_instruct_sglang_examples_5_temp_0_chain_True_seed_184623_line_to_skip_7098",
    #     f"{main_path}/cot_feedback_-grade_12_original_parallel_train.tgt-9000_llama_3_70b_instruct_sglang_examples_5_temp_0_chain_True_seed_184623_line_to_skip_8000",
    #     f"{main_path}/cot_feedback_-grade_12_original_parallel_train.tgt-9000_llama_3_70b_instruct_sglang_examples_5_temp_0_chain_True_seed_184623_line_to_skip_8101",
    #     f"{main_path}/cot_feedback_-grade_12_original_parallel_train.tgt-10000_llama_3_70b_instruct_sglang_examples_5_temp_0_chain_True_seed_184623_line_to_skip_9000",
    #     f"{main_path}/cot_feedback_-grade_12_original_parallel_train.tgt-10000_llama_3_70b_instruct_sglang_examples_5_temp_0_chain_True_seed_184623_line_to_skip_9099",
    #     # f"{main_path}/",
    #
    # ]


    # experiment_directories = [
    #     "phase_2_experiments/fvp_with_synthetic_data_creation_methods/grade-6/data_creation_method_4/syntactic_data_generation/llama_3_70b_instruct_sglang/cot_feedback_-grade_6_original_parallel_train.tgt-1000_llama_3_70b_instruct_sglang_examples_5_temp_0_chain_True_seed_184623_line_to_skip_-1"
    # ]

    # experiment_directories = [
    #     # "phase_2_experiments/fvp_with_synthetic_data_creation_methods/grade-1/data_creation_method_4/syntactic_data_generation/llama_3_70b_instruct_sglang/cot_feedback_-grade_1_eval_original_target.tgt-750_llama_3_70b_instruct_sglang_examples_5_temp_0_chain_True_seed_184623_line_to_skip_-1",
    #     # "phase_2_experiments/fvp_with_synthetic_data_creation_methods/grade-3/data_creation_method_4/syntactic_data_generation/llama_3_70b_instruct_sglang/cot_feedback_-grade_3_original_parallel_eval.tgt-750_llama_3_70b_instruct_sglang_examples_5_temp_0_chain_True_seed_184623_line_to_skip_-1",
    #     # "phase_2_experiments/fvp_with_synthetic_data_creation_methods/grade-3/data_creation_method_4/syntactic_data_generation/llama_3_70b_instruct_sglang/cot_feedback_-grade_3_original_parallel_train.tgt-1000_llama_3_70b_instruct_sglang_examples_5_temp_0_chain_True_seed_184623_line_to_skip_-1",
    #     # "phase_2_experiments/fvp_with_synthetic_data_creation_methods/grade-6/data_creation_method_4/syntactic_data_generation/llama_3_70b_instruct_sglang/cot_feedback_-grade_6_original_parallel_eval.tgt-750_llama_3_70b_instruct_sglang_examples_5_temp_0_chain_True_seed_184623_line_to_skip_-1",
    #     # "phase_2_experiments/fvp_with_synthetic_data_creation_methods/grade-9/data_creation_method_4/syntactic_data_generation/llama_3_70b_instruct_sglang/cot_feedback_-grade_9_original_parallel_eval.tgt-750_llama_3_70b_instruct_sglang_examples_5_temp_0_chain_True_seed_184623_line_to_skip_-1",
    #     "phase_2_experiments/fvp_with_synthetic_data_creation_methods/grade-9/data_creation_method_4/syntactic_data_generation/llama_3_70b_instruct_sglang/cot_feedback_-grade_9_original_parallel_train.tgt-1000_llama_3_70b_instruct_sglang_examples_5_temp_0_chain_True_seed_184623_line_to_skip_-1",
    #     # "phase_2_experiments/fvp_with_synthetic_data_creation_methods/grade-12/data_creation_method_4/syntactic_data_generation/llama_3_70b_instruct_sglang/cot_feedback_-grade_12_original_parallel_eval.tgt-750_llama_3_70b_instruct_sglang_examples_5_temp_0_chain_True_seed_184623_line_to_skip_-1"
    # ]
    # # Process each directory
    # for directory in experiment_directories:
    #     update_output_file(directory)
    #
    #
    # # fill empty lines in output.txt with previous iternations outputs.
    # # if all the previous iternation still results empty lines then just copy paste the input sentence from input.txt