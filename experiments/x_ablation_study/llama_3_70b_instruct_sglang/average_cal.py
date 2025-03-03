import os
import csv
import numpy as np


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def read_and_average_csv(file_paths, feature, prompt_level):
    data = []
    headers = None

    for file_path in file_paths:
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            row = next(reader)
            if headers is None:
                headers = row  # Capture headers from the first file
            data.append(row)

    # Convert the data to a numpy array for easier manipulation
    data_array = np.array(data, dtype=object)

    # Identify the numeric columns and average them
    numeric_indices = [i for i, v in enumerate(data_array[0]) if is_number(v)]
    numeric_data = data_array[:, numeric_indices].astype(float)
    averaged_data = np.mean(numeric_data, axis=0)

    # Construct the averaged row
    averaged_row = data_array[0].copy()
    for i, index in enumerate(numeric_indices):
        averaged_row[index] = averaged_data[i]

    # Add the experiment paths
    experiment_paths = "; ".join(file_paths)
    averaged_row = np.append(averaged_row, experiment_paths)

    # Convert averaged_row to a list and add feature and prompt level info at the start
    averaged_row = [feature, prompt_level] + averaged_row.tolist()

    return headers, averaged_row


def save_averaged_data(output_file, combined_data):
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in combined_data:
            writer.writerow(row)


def main(base_dir, feature_list, prompt_levels, seeds, output_file):
    combined_data = []

    for feature in feature_list:
        for prompt_level in prompt_levels:
            file_paths = []
            for seed in seeds:
                file_path = os.path.join(base_dir, feature, f"{prompt_level}_seed_{seed}", "success_rate.csv")
                # file_path = os.path.join(base_dir, feature, f"{prompt_level}_seed_{seed}", "easse_stats.csv")
                if os.path.exists(file_path):
                    file_paths.append(file_path)
                else:
                    print(f"Warning: {file_path} does not exist.")

            if file_paths:
                headers, averaged_row = read_and_average_csv(file_paths, feature, prompt_level)
                combined_data.append(averaged_row)

    save_averaged_data(output_file, combined_data)


if __name__ == "__main__":
    base_dir = "experiments/x_ablation_study/llama_3_70b_instruct_sglang"
    feature_list = ["MaxDepDepth", "MaxDepLength", "DiffWords", "WordCount"]
    prompt_levels = [
        "free_style/free_style-gold-filtered_wiki.valid_v1.1.src-100_llama_3_70b_instruct_sglang_examples_0_temp_0_chain_False",
        "no_sys_prompt/no_sys_prompt-gold-filtered_wiki.valid_v1.1.src-100_llama_3_70b_instruct_sglang_examples_0_temp_0_chain_False",
        "level-2.1_prompt/level-2.1_prompt-gold-filtered_wiki.valid_v1.1.src-100_llama_3_70b_instruct_sglang_examples_0_temp_0_chain_False",
        "level-3_prompt/level-3_prompt-gold-filtered_wiki.valid_v1.1.src-100_llama_3_70b_instruct_sglang_examples_0_temp_0_chain_False",
        "level-3_prompt/level-3_prompt-gold-filtered_wiki.valid_v1.1.src-100_llama_3_70b_instruct_sglang_examples_5_temp_0_chain_False",
        "level-4_prompt/level-4_prompt-gold-filtered_wiki.valid_v1.1.src-100_llama_3_70b_instruct_sglang_examples_0_temp_0_chain_False",
        "level-4_prompt_fs/level-4_prompt-gold-filtered_wiki.valid_v1.1.src-100_llama_3_70b_instruct_sglang_examples_5_temp_0_chain_False",
        "level-4_feedbackloop_prompt-gold-filtered_wiki.valid_v1.1.src-100_llama_3_70b_instruct_sglang_examples_5_temp_0_chain_True"
    ]
    seeds = [473829, 921405, 184623, 756301, 368914]
    output_file = os.path.join(base_dir, "all_successrate_averaged.csv")
    # output_file = os.path.join(base_dir, "all_easse_stats_averaged.csv")

    main(base_dir, feature_list, prompt_levels, seeds, output_file)
