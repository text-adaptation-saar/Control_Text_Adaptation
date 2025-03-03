import pandas as pd

# Function to extract lines from the source and target files
def extract_lines(file_path, line_numbers):
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        return [lines[i - 1].strip() for i in line_numbers]  # Line numbers are assumed to start at 1

# def extract_lines(file_path, line_numbers):
#     with open(file_path, 'r', encoding='utf-8') as file:
#         lines = file.readlines()
#
#     extracted_lines = []
#     for i in line_numbers:
#         if 0 < i <= len(lines):
#             extracted_lines.append(lines[i - 1].strip())
#         else:
#             print(f"Line number {i} is out of range for file {file_path}")
#
#     return extracted_lines

def create_dataset(filtered_ratio_csv_path, original_dataset_src_path, original_dataset_tgt_path, save_new_dataset_path):
    print("Data preparation is started!")

    # randomizse and pick 2,000 csv ratio lines and write to new file
    # rewrite the csv by elimiating 2,000 lines.
    # then generate the dataset using the csv file.

    # Output paths
    train_csv_ratios = save_new_dataset_path + '/ratio_stats_filtered_wiki_train_data.csv'
    eval_csv_ratios = save_new_dataset_path + '/ratio_stats_filtered_wiki_valid_data.csv'
    test_csv_ratios = save_new_dataset_path + '/ratio_stats_filtered_wiki_test_data.csv'
    train_src_data = save_new_dataset_path + '/filtered_wiki.train.src'
    train_tgt_data = save_new_dataset_path + '/filtered_wiki.train.tgt'
    eval_src_data = save_new_dataset_path + '/filtered_wiki.valid.src'
    eval_tgt_data = save_new_dataset_path + '/filtered_wiki.valid.tgt'
    test_src_data = save_new_dataset_path + '/filtered_wiki.test.src'
    test_tgt_data = save_new_dataset_path + '/filtered_wiki.test.tgt'

    # Load the CSV file
    df = pd.read_csv(filtered_ratio_csv_path)

    # Shuffle the DataFrame, as long as the seed (random_state=1) is the same and the input DataFrame hasn't changed,
    # the shuffled DataFrame df_shuffled will be the same each time.
    df_shuffled = df.sample(frac=1, random_state=1).reset_index(drop=True)

    # Split the data
    eval_df = df_shuffled[:1000]  # First 1000 rows for evaluation
    test_df = df_shuffled[1000:2000]  # Next 1000 rows for testing
    train_df = df_shuffled[2000:]  # The rest for training

    # Save the training and eval/test CSV files
    train_df.to_csv(train_csv_ratios, index=False)
    eval_df.to_csv(eval_csv_ratios, index=False)
    test_df.to_csv(test_csv_ratios, index=False)

    # Extract sentences for eval sets
    eval_src_lines = extract_lines(original_dataset_src_path, eval_df['Line'].tolist())
    eval_tgt_lines = extract_lines(original_dataset_tgt_path, eval_df['Line'].tolist())
    # Write sentences to new eval/test files
    with open(eval_src_data, 'w', encoding='utf-8') as file:
        file.write('\n'.join(eval_src_lines))
        file.write('\n')
    with open(eval_tgt_data, 'w', encoding='utf-8') as file:
        file.write('\n'.join(eval_tgt_lines))
        file.write('\n')

    # Extract sentences for TEST sets
    test_src_lines = extract_lines(original_dataset_src_path, test_df['Line'].tolist())
    test_tgt_lines = extract_lines(original_dataset_tgt_path, test_df['Line'].tolist())
    # Write sentences to new eval/test files
    with open(test_src_data, 'w', encoding='utf-8') as file:
        file.write('\n'.join(test_src_lines))
        file.write('\n')
    with open(test_tgt_data, 'w', encoding='utf-8') as file:
        file.write('\n'.join(test_tgt_lines))
        file.write('\n')


    # Extract sentences for the training set
    train_src_lines = extract_lines(original_dataset_src_path, train_df['Line'].tolist())
    train_tgt_lines = extract_lines(original_dataset_tgt_path, train_df['Line'].tolist())

    # Write sentences to new training files
    with open(train_src_data, 'w', encoding='utf-8') as file:
        file.write('\n'.join(train_src_lines))
        file.write('\n')
    with open(train_tgt_data, 'w', encoding='utf-8') as file:
        file.write('\n'.join(train_tgt_lines))
        file.write('\n')

    print("Data preparation is complete.")

def filter_dataset_from_filtered_ratio(filtered_ratio_csv_path, original_dataset_src_path, original_dataset_tgt_path,
                                       train_csv_ratios, train_src_data, train_tgt_data):
    print("Data preparation is started!")

    # create filtered wiki src-tgt text files from filtered ratio file.
    # Load the CSV file
    train_df = pd.read_csv(filtered_ratio_csv_path)
    # train_df['New Line'] = range(1, len(train_df) + 1)

    new_line_numbers = range(1, len(train_df) + 1)
    # Insert this new column at the first position (index 0) with the column name 'New Line'
    train_df.insert(0, 'New Line', new_line_numbers)


    # Save the training and eval/test CSV files
    train_df.to_csv(train_csv_ratios, index=False)

    # Extract sentences for the training set
    train_src_lines = extract_lines(original_dataset_src_path, train_df['Line'].tolist())
    train_tgt_lines = extract_lines(original_dataset_tgt_path, train_df['Line'].tolist())

    # Write sentences to new training files
    with open(train_src_data, 'w', encoding='utf-8') as file:
        file.write('\n'.join(train_src_lines))
        file.write('\n')
    with open(train_tgt_data, 'w', encoding='utf-8') as file:
        file.write('\n'.join(train_tgt_lines))
        file.write('\n')

    print("Data preparation is complete.")


def create_eval_by_fitler(file_paths, save_path, original_dataset_src_path, original_dataset_tgt_path):
    # Read all CSV files into DataFrames
    dataframes = [pd.read_csv(path) for path in file_paths]

    # Find common 'Line' values across all DataFrames
    common_lines = set(dataframes[0]['Line'])
    for df in dataframes[1:]:
        common_lines = common_lines.intersection(set(df['Line']))

    # Filter each DataFrame to only include common 'Line' values
    filtered_dataframes = [df[df['Line'].isin(common_lines)] for df in dataframes]

    # Save the filtered DataFrames to new CSV files
    for i, df in enumerate(filtered_dataframes):
        save_ratio_file = save_path + f'/mutually_filtered_ratio_{i}.csv'
        df.to_csv(save_ratio_file, index=False)

        # Extract sentences for eval sets
        eval_src_lines = extract_lines(original_dataset_src_path, df['Line'].tolist())
        eval_tgt_lines = extract_lines(original_dataset_tgt_path[i], df['Line'].tolist())
        # Write sentences to new eval/test files
        eval_src_data = save_path + f'/filtered_tune.truecase.detok.orig.{i}'
        eval_tgt_data = save_path + f'/filtered_tune.truecase.detok.simp.{i}'
        with open(eval_src_data, 'w', encoding='utf-8') as file:
            file.write('\n'.join(eval_src_lines))
            file.write('\n')
        with open(eval_tgt_data, 'w', encoding='utf-8') as file:
            file.write('\n'.join(eval_tgt_lines))
            file.write('\n')


if __name__=="__main__":
    # Create wiki train, eval, test dataset randomly select before filtering.
    create_dataset(filtered_ratio_csv_path="data_auxiliary/en/feature_values_analyse/WikiLarge_V2/full_original/wikilarge_all_v2/tgt/ratio_stats.csv",
                   original_dataset_src_path="data/en/WikiLarge/full_original_V2/wiki.full.aner.ori.train.detok.src",
                   original_dataset_tgt_path="data/en/WikiLarge/full_original_V2/wiki.full.aner.ori.train.detok.dst",
                   save_new_dataset_path="data/en/WikiLarge/train_val_test")

    # filter out WIKI TRAIN and create a filtered wiki train.
    csv_file = "data_auxiliary/en/feature_values_analyse/WikiLarge_V2/train_val_test/train/3_filter_ratio_and_abs_src_values/filtered_ratio_stats.csv"
    src_file = "data/en/WikiLarge/full_original_V2/wiki.full.aner.ori.train.detok.src"
    tgt_file = "data/en/WikiLarge/full_original_V2/wiki.full.aner.ori.train.detok.dst"
    output_file_save_path = "data_filtered/en/wikilarge_train_val_test/train"
    # create_separate_src_tgt_files(csv_file, src_file, tgt_file, output_file_save_path)
    filter_dataset_from_filtered_ratio(csv_file, src_file, tgt_file, output_file_save_path)

    # filter out WIKI VAL and create a filtered WIKI VAL.
    csv_file = "data_auxiliary/en/feature_values_analyse/WikiLarge_V2/train_val_test/val/3_filter_ratio_and_abs_src_values/filtered_ratio_stats.csv"
    src_file = "data/en/WikiLarge/full_original_V2/wiki.full.aner.ori.train.detok.src"
    tgt_file = "data/en/WikiLarge/full_original_V2/wiki.full.aner.ori.train.detok.dst"
    # Output paths
    output_csv_ratios =  'data_filtered/en/wikilarge_train_val_test/val/ratio_stats_filtered_wiki_val_data.csv'
    output_src_data =   'data_filtered/en/wikilarge_train_val_test/val/filtered_wiki.valid.src'
    output_tgt_data =  'data_filtered/en/wikilarge_train_val_test/val/filtered_wiki.valid.tgt'
    # create_separate_src_tgt_files(csv_file, src_file, tgt_file, output_file_save_path)
    filter_dataset_from_filtered_ratio(csv_file, src_file, tgt_file,output_csv_ratios, output_src_data, output_tgt_data)

    # filter out WIKI TEST and create a filtered WIKI TEST.
    csv_file = "data_auxiliary/en/feature_values_analyse/WikiLarge_V2/train_val_test/test/3_filter_ratio_and_abs_src_values/filtered_ratio_stats.csv"
    src_file = "data/en/WikiLarge/full_original_V2/wiki.full.aner.ori.train.detok.src"
    tgt_file = "data/en/WikiLarge/full_original_V2/wiki.full.aner.ori.train.detok.dst"
    # Output paths
    output_csv_ratios = 'data_filtered/en/wikilarge_train_val_test/test/ratio_stats_filtered_wiki_test_data.csv'
    output_src_data = 'data_filtered/en/wikilarge_train_val_test/test/filtered_wiki.test.src'
    output_tgt_data = 'data_filtered/en/wikilarge_train_val_test/test/filtered_wiki.test.tgt'
    # create_separate_src_tgt_files(csv_file, src_file, tgt_file, output_file_save_path)
    filter_dataset_from_filtered_ratio(csv_file, src_file, tgt_file, output_csv_ratios, output_src_data,
                                       output_tgt_data)
