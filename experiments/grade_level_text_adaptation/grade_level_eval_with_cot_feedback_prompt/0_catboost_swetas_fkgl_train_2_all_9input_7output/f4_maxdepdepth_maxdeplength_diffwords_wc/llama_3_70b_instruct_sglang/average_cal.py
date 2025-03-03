import os
import csv
import numpy as np

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def read_csv_rows(file_path):
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        return list(reader)

def average_rows_across_files(file_paths):
    data = None  # Initialize data as None

    # Read all rows from each file and store them in a list
    for file_path in file_paths:
        file_data = read_csv_rows(file_path)
        if data is None:
            data = np.array(file_data, dtype=object)  # Initialize with data from the first file
        else:
            # Stack the data from the next file along the third axis for averaging later
            data = np.dstack((data, np.array(file_data, dtype=object)))

    # Now, we average the data across the third axis (which corresponds to the different seeds)
    # We first identify numeric columns to average, while keeping the non-numeric ones constant
    numeric_indices = [i for i, v in enumerate(data[0, 0, :]) if is_number(v)]

    averaged_data = np.zeros_like(data[:, :, 0], dtype=object)
    averaged_data[:, :] = data[:, :, 0]  # Copy non-numeric columns from the first seed

    for row_index in range(data.shape[0]):
        for col_index in numeric_indices:
            numeric_values = data[row_index, col_index, :].astype(float)
            averaged_data[row_index, col_index] = np.mean(numeric_values)

    return averaged_data

def save_averaged_data(output_file, headers, combined_data):
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(combined_data)

def main(base_dir, seeds, output_file):
    combined_data = []

    # Construct file paths for all seeds
    file_paths = [
        os.path.join(
            base_dir,
            f"cot_feedback_with_catboost_swetas-filtered_wiki.valid_v1.1.src-200_llama_3_70b_instruct_sglang_examples_5_temp_0_chain_True_seed_{seed}",
            "doc_level_scores.csv"
            # "success_rate.csv"
            # "easse_stats.csv"
        )
        for seed in seeds
    ]

    # Check for existence of files and filter out missing files
    file_paths = [fp for fp in file_paths if os.path.exists(fp)]
    if not file_paths:
        print("No files found, exiting.")
        return

    # Read the header from the first file
    with open(file_paths[0], 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)

    # Average rows across all files
    averaged_data = average_rows_across_files(file_paths)

    # Save the averaged data
    save_averaged_data(output_file, headers, averaged_data)

if __name__ == "__main__":
    base_dir = "experiments/train_v3_and_val_v1.1_wo_line_46/grade_level_eval_with_cot_feedback_prompt/0_catboost_swetas_fkgl_train_2_all_9input_7output/f4_maxdepdepth_maxdeplength_diffwords_wc/llama_3_70b_instruct_sglang/seeds"
    seeds = [473829, 921405, 184623, 756301, 368914]
    # output_file = os.path.join(base_dir, "all_easse_stats_averaged.csv")
    # output_file = os.path.join(base_dir, "all_successrate_averaged.csv")
    output_file = os.path.join(base_dir, "all_doc_level_scores_averaged.csv")

    main(base_dir, seeds, output_file)
