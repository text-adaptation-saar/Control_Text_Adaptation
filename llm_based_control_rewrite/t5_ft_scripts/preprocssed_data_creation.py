# -- fix path --
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
# -- end fix path --
import shutil
import pandas as pd
import os
import argparse
from pathlib import Path

feature_mapping = {
	'WordCount': 'W',
	'Length': 'C',
	'Leven': 'L',
	'FreqRank': 'WR',
	'MaxDepDepth': 'DTD',
	'MaxDepLength': 'DTL',
	'DiffWords': 'DW',
}


def preprocess_dataset(dataset_complex_src_path, dataset_simple_tgt_path, ratio_files, processed_data_save_path,
                       feature_names,
                       fixed_ratios=None, gold_ref_ratios=None):

    processed_data_save_path = Path(processed_data_save_path)
    processed_data_save_path.mkdir(parents=True, exist_ok=True)
    print(f'Preprocessing dataset src: {dataset_complex_src_path}')
    print(f'Preprocessing dataset tgt: {dataset_simple_tgt_path}')
    print(f'Preprocessing dataset ratio files: {ratio_files}')

    # for phase in ["valid", "test"]:
    shutil_copy_complex_output_filepath = processed_data_save_path /  os.path.basename(dataset_complex_src_path)
    shutil_copy_simple_output_filepath = processed_data_save_path / os.path.basename(dataset_simple_tgt_path)

    processed_complex_output_filepath = processed_data_save_path /  f'{os.path.basename(dataset_complex_src_path)}_processed'

    if ratio_files is not None:
        ratio_files_array=ratio_files.split(",")
        prepend_ratios_to_source_text(dataset_complex_src_path, ratio_files_array, processed_complex_output_filepath)

    if fixed_ratios is not None:
        fixed_ratios = fixed_ratios.split(",")
        prepend_fixed_ratios_to_source_text(dataset_complex_src_path, fixed_ratios, processed_complex_output_filepath)

    if gold_ref_ratios is not None:
        prepend_gold_ratios_to_source_text(dataset_complex_src_path, gold_ref_ratios, processed_complex_output_filepath, feature_names)

    shutil.copy(dataset_complex_src_path, shutil_copy_complex_output_filepath)
    shutil.copy(dataset_simple_tgt_path, shutil_copy_simple_output_filepath)

    print(f'Preprocessing dataset "{dataset_complex_src_path,dataset_simple_tgt_path}" is finished.')
    print(f'Processed dataset saved : {processed_data_save_path}')


def prepend_ratios_to_source_text(source_file_path, ratios_file_paths, processed_source_file_path):
    # Read the source sentences from the file
    with open(source_file_path, 'r') as file:
        source_sentences = file.readlines()

    # Read ratio file. IMPORTANT to give in this order.
    ratios_df_dtd = pd.read_csv(ratios_file_paths[0])
    ratios_df_dtl = pd.read_csv(ratios_file_paths[1])
    ratios_df_dw = pd.read_csv(ratios_file_paths[2])
    ratios_df_wc = pd.read_csv(ratios_file_paths[3])

    # Create a new column 'current_line' which starts from 1 up to the length of the DataFrame
    ratios_df_dtd['current_line'] = range(1, len(ratios_df_dtd) + 1)
    ratios_df_dtl['current_line'] = range(1, len(ratios_df_dtl) + 1)
    ratios_df_dw['current_line'] = range(1, len(ratios_df_dw) + 1)
    ratios_df_wc['current_line'] = range(1, len(ratios_df_wc) + 1)

    # Get the number of lines in the source file
    num_source_sentences = len(source_sentences)
    # Compare the number of rows in each DataFrame to the number of source sentences
    all_match = (
            len(ratios_df_dtd) >= num_source_sentences and
            len(ratios_df_dtl) >= num_source_sentences and
            len(ratios_df_dw) >= num_source_sentences and
            len(ratios_df_wc) >= num_source_sentences
    )
    if all_match:
        print("All files have the same number of rows as the source sentences. Proceeding to include ratio prefix for complex source sentences.")

        # Open the output file for writing the new source lines
        with open(processed_source_file_path, 'w') as output_file:
            # Iterate over each source sentence and corresponding row in the DataFrame by line number
            for index, source_sentence in enumerate(source_sentences):
                # Adjust index to match 'current_line' which starts from 1
                line_number = index + 1

                # Retrieve the relevant ratios for the current sentence using 'current_line'
                if line_number in ratios_df_dtd['current_line'].values:
                    row_dtd = ratios_df_dtd[ratios_df_dtd['current_line'] == line_number].iloc[0]
                    row_dtl = ratios_df_dtl[ratios_df_dtl['current_line'] == line_number].iloc[0]
                    row_dw = ratios_df_dw[ratios_df_dw['current_line'] == line_number].iloc[0]
                    row_wc = ratios_df_wc[ratios_df_wc['current_line'] == line_number].iloc[0]

                    max_dep_depth_ratio = "{:.2f}".format(float(row_dtd['predicted_MaxDepDepth_ratio']))
                    max_dep_length_ratio = "{:.2f}".format(float(row_dtl['predicted_MaxDepLength_ratio']))
                    diff_words_ratio = "{:.2f}".format(float(row_dw['predicted_DiffWords_ratio']))
                    word_count_ratio = "{:.2f}".format(float(row_wc['predicted_WordCount_ratio']))

                    # Create the new source line with ratios prepended
                    new_source_line = f"DTD_{max_dep_depth_ratio} DTL_{max_dep_length_ratio} DW_{diff_words_ratio} W_{word_count_ratio}  {source_sentence}"

                    # Write the new source line to the output file
                    output_file.write(new_source_line)
                else:
                    print(f"Warning: No ratio data available for line {line_number}")
        print("Adding ratio prefix task completed!")
    else:
        print("Mismatch in row counts:")
        print("DTD rows:", len(ratios_df_dtd), "Expected:", num_source_sentences)
        print("DTL rows:", len(ratios_df_dtl), "Expected:", num_source_sentences)
        print("DW rows:", len(ratios_df_dw), "Expected:", num_source_sentences)
        print("WC rows:", len(ratios_df_wc), "Expected:", num_source_sentences)


def prepend_ratios_to_source_text_ACCESS(source_file_path, ratios_file_paths, processed_source_file_path):
    # order: #W_0.71 C_0.63 L_0.72 WR_1.00 DTD_1.12
    # Read the source sentences from the file
    with open(source_file_path, 'r') as file:
        source_sentences = file.readlines()

    # Read ratio file. IMPORTANT to give in this order.
    ratios_df_wc = pd.read_csv(ratios_file_paths[0])
    ratios_df_length = pd.read_csv(ratios_file_paths[1])
    ratios_df_leven = pd.read_csv(ratios_file_paths[2])
    ratios_df_freqrank = pd.read_csv(ratios_file_paths[3])
    ratios_df_dtd = pd.read_csv(ratios_file_paths[4])

    # Create a new column 'current_line' which starts from 1 up to the length of the DataFrame
    ratios_df_wc['current_line'] = range(1, len(ratios_df_wc) + 1)
    ratios_df_length['current_line'] = range(1, len(ratios_df_length) + 1)
    ratios_df_leven['current_line'] = range(1, len(ratios_df_leven) + 1)
    ratios_df_freqrank['current_line'] = range(1, len(ratios_df_freqrank) + 1)
    ratios_df_dtd['current_line'] = range(1, len(ratios_df_dtd) + 1)

    # Get the number of lines in the source file
    num_source_sentences = len(source_sentences)
    # Compare the number of rows in each DataFrame to the number of source sentences
    all_match = (
            len(ratios_df_wc) >= num_source_sentences and
            len(ratios_df_length) >= num_source_sentences and
            len(ratios_df_leven) >= num_source_sentences and
            len(ratios_df_freqrank) >= num_source_sentences and
            len(ratios_df_dtd) >= num_source_sentences
    )
    if all_match:
        print("All files have the same number of rows as the source sentences. Proceeding to include ratio prefix for complex source sentences.")

        # Open the output file for writing the new source lines
        with open(processed_source_file_path, 'w') as output_file:
            # Iterate over each source sentence and corresponding row in the DataFrame by line number
            for index, source_sentence in enumerate(source_sentences):
                # Adjust index to match 'current_line' which starts from 1
                line_number = index + 1

                # Retrieve the relevant ratios for the current sentence using 'current_line'
                if line_number in ratios_df_dtd['current_line'].values:
                    row_wc = ratios_df_wc[ratios_df_wc['current_line'] == line_number].iloc[0]
                    row_len = ratios_df_length[ratios_df_length['current_line'] == line_number].iloc[0]
                    row_leven = ratios_df_leven[ratios_df_leven['current_line'] == line_number].iloc[0]
                    row_freqrank = ratios_df_freqrank[ratios_df_freqrank['current_line'] == line_number].iloc[0]
                    row_dtd = ratios_df_dtd[ratios_df_dtd['current_line'] == line_number].iloc[0]

                    word_count_ratio = "{:.2f}".format(float(row_wc['predicted_WordCount_ratio']))
                    length_ratio = "{:.2f}".format(float(row_len['predicted_Length_ratio']))
                    leven_ratio = "{:.2f}".format(float(row_leven['predicted_Leven_ratio']))
                    freqrank_ratio = "{:.2f}".format(float(row_freqrank['predicted_FreqRank_ratio']))
                    max_dep_depth_ratio = "{:.2f}".format(float(row_dtd['predicted_MaxDepDepth_ratio']))

                    # Create the new source line with ratios prepended
                    # order: #W_0.71 C_0.63 L_0.72 WR_1.00 DTD_1.12
                    new_source_line = f"W_{word_count_ratio} C_{length_ratio} L_{leven_ratio} WR_{freqrank_ratio} DTD_{max_dep_depth_ratio}  {source_sentence}"

                    # Write the new source line to the output file
                    output_file.write(new_source_line)
                else:
                    print(f"Warning: No ratio data available for line {line_number}")
        print("Adding ratio prefix task completed!")
    else:
        print("Mismatch in row counts:")
        print("WC rows:", len(ratios_df_wc), "Expected:", num_source_sentences)
        print("Length rows:", len(ratios_df_length), "Expected:", num_source_sentences)
        print("Leven rows:", len(ratios_df_leven), "Expected:", num_source_sentences)
        print("FreqRank rows:", len(ratios_df_freqrank), "Expected:", num_source_sentences)
        print("DTD rows:", len(ratios_df_dtd), "Expected:", num_source_sentences)

def prepend_fixed_ratios_to_source_text(source_file_path, fixed_ratios, processed_source_file_path):
    # Read the source sentences from the file
    with open(source_file_path, 'r') as file:
        source_sentences = file.readlines()

    print("Proceeding to include fixed ratio prefix for complex source sentences.")

    # Open the output file for writing the new source lines
    with open(processed_source_file_path, 'w') as output_file:
        # Iterate over each source sentence and corresponding row in the DataFrame by line number
        for index, source_sentence in enumerate(source_sentences):
            max_dep_depth_ratio = "{:.2f}".format(float(fixed_ratios[0]))
            max_dep_length_ratio = "{:.2f}".format(float(fixed_ratios[1]))
            diff_words_ratio = "{:.2f}".format(float(fixed_ratios[2]))
            word_count_ratio = "{:.2f}".format(float(fixed_ratios[3]))

            # Create the new source line with ratios prepended
            new_source_line = f"DTD_{max_dep_depth_ratio} DTL_{max_dep_length_ratio} DW_{diff_words_ratio} WC_{word_count_ratio}  {source_sentence}"

            # Write the new source line to the output file
            output_file.write(new_source_line)

    print("Adding fixed ratio prefix task completed!")

def prepend_gold_ratios_to_source_text(source_file_path, gold_ref_ratios, processed_source_file_path, feature_names):
    # Read the source sentences from the file
    with open(source_file_path, 'r') as file:
        source_sentences = file.readlines()

    # Read ratio file. IMPORTANT to give in this order.
    ratios_df = pd.read_csv(gold_ref_ratios)

    # Create a new column 'current_line' which starts from 1 up to the length of the DataFrame
    ratios_df['current_line'] = range(1, len(ratios_df) + 1)

    # Get the number of lines in the source file
    num_source_sentences = len(source_sentences)
    # Compare the number of rows in each DataFrame to the number of source sentences
    all_match = (
            len(ratios_df) == num_source_sentences
    )
    if all_match:
        print("Gold ratio has the same number of rows as the source sentences. Proceeding to include gold ratio prefix for complex source sentences.")

        # Open the output file for writing the new source lines
        with open(processed_source_file_path, 'w') as output_file:
            # Iterate over each source sentence and corresponding row in the DataFrame by line number
            for index, source_sentence in enumerate(source_sentences):
                # Adjust index to match 'current_line' which starts from 1
                line_number = index + 1
                new_source_line=""

                # Retrieve the relevant ratios for the current sentence using 'current_line'
                if line_number in ratios_df['current_line'].values:
                    row = ratios_df[ratios_df['current_line'] == line_number].iloc[0]

                    for feature in feature_names:
                        ratio = "{:.2f}".format(float(row[f'{feature}_ratio']))
                        new_source_line+=f"{feature_mapping[feature]}_{ratio} "

                    # max_dep_depth_ratio = "{:.2f}".format(float(row['MaxDepDepth_ratio']))
                    # max_dep_length_ratio = "{:.2f}".format(float(row['MaxDepLength_ratio']))
                    # diff_words_ratio = "{:.2f}".format(float(row['DiffWords_ratio']))
                    # word_count_ratio = "{:.2f}".format(float(row['WordCount_ratio']))
                    # # Create the new source line with ratios prepended
                    # new_source_line = f"DTD_{max_dep_depth_ratio} DTL_{max_dep_length_ratio} DW_{diff_words_ratio} WC_{word_count_ratio}  {source_sentence}"

                    # Write the new source line to the output file
                    new_source_line += f" {source_sentence}"
                    output_file.write(new_source_line)
                else:
                    print(f"Warning: No ratio data available for line {line_number}")
        print("Adding gold ratio prefix task completed!")
    else:
        print("Mismatch in row counts:")
        print("Gold ratio rows:", len(ratios_df), "Expected:", num_source_sentences)

def prepend_ratios_to_source_text_all_7f(source_file_path, ratios_file_paths, processed_source_file_path):
    # order: #  W_0.92 C_0.90 DW_0.75 DTL_1.00 DTD_1.00 L_0.92 WR_1.03
    # Read the source sentences from the file
    with open(source_file_path, 'r') as file:
        source_sentences = file.readlines()

    # Read ratio file. IMPORTANT to give in this order.
    ratios_df_wc = pd.read_csv(ratios_file_paths[0])
    ratios_df_length = pd.read_csv(ratios_file_paths[1])
    ratios_df_dw = pd.read_csv(ratios_file_paths[2])
    ratios_df_dtl = pd.read_csv(ratios_file_paths[3])
    ratios_df_dtd = pd.read_csv(ratios_file_paths[4])
    ratios_df_leven = pd.read_csv(ratios_file_paths[5])
    ratios_df_freqrank = pd.read_csv(ratios_file_paths[6])

    # Create a new column 'current_line' which starts from 1 up to the length of the DataFrame
    ratios_df_wc['current_line'] = range(1, len(ratios_df_wc) + 1)
    ratios_df_length['current_line'] = range(1, len(ratios_df_length) + 1)
    ratios_df_dw['current_line'] = range(1, len(ratios_df_dw) + 1)
    ratios_df_dtl['current_line'] = range(1, len(ratios_df_dtl) + 1)
    ratios_df_leven['current_line'] = range(1, len(ratios_df_leven) + 1)
    ratios_df_freqrank['current_line'] = range(1, len(ratios_df_freqrank) + 1)
    ratios_df_dtd['current_line'] = range(1, len(ratios_df_dtd) + 1)

    # Get the number of lines in the source file
    num_source_sentences = len(source_sentences)
    # Compare the number of rows in each DataFrame to the number of source sentences
    all_match = (
            len(ratios_df_wc) >= num_source_sentences and
            len(ratios_df_length) >= num_source_sentences and
            len(ratios_df_leven) >= num_source_sentences and
            len(ratios_df_freqrank) >= num_source_sentences and
            len(ratios_df_dtd) >= num_source_sentences and
            len(ratios_df_dtl) >= num_source_sentences and
            len(ratios_df_dw) >= num_source_sentences
    )
    if all_match:
        print("All files have the same number of rows as the source sentences. Proceeding to include ratio prefix for complex source sentences.")

        # Open the output file for writing the new source lines
        with open(processed_source_file_path, 'w') as output_file:
            # Iterate over each source sentence and corresponding row in the DataFrame by line number
            for index, source_sentence in enumerate(source_sentences):
                # Adjust index to match 'current_line' which starts from 1
                line_number = index + 1

                # Retrieve the relevant ratios for the current sentence using 'current_line'
                if line_number in ratios_df_dtd['current_line'].values:
                    row_wc = ratios_df_wc[ratios_df_wc['current_line'] == line_number].iloc[0]
                    row_len = ratios_df_length[ratios_df_length['current_line'] == line_number].iloc[0]
                    row_dw = ratios_df_length[ratios_df_length['current_line'] == line_number].iloc[0]
                    row_dtl = ratios_df_length[ratios_df_length['current_line'] == line_number].iloc[0]
                    row_dtd = ratios_df_dtd[ratios_df_dtd['current_line'] == line_number].iloc[0]
                    row_leven = ratios_df_leven[ratios_df_leven['current_line'] == line_number].iloc[0]
                    row_freqrank = ratios_df_freqrank[ratios_df_freqrank['current_line'] == line_number].iloc[0]

                    word_count_ratio = "{:.2f}".format(float(row_wc['predicted_WordCount_ratio']))
                    length_ratio = "{:.2f}".format(float(row_len['predicted_Length_ratio']))
                    dw_ratio = "{:.2f}".format(float(row_dw['predicted_DiffWords_ratio']))
                    dtl_ratio = "{:.2f}".format(float(row_dtl['predicted_MaxDepLength_ratio']))
                    max_dep_depth_ratio = "{:.2f}".format(float(row_dtd['predicted_MaxDepDepth_ratio']))
                    leven_ratio = "{:.2f}".format(float(row_leven['predicted_Leven_ratio']))
                    freqrank_ratio = "{:.2f}".format(float(row_freqrank['predicted_FreqRank_ratio']))

                    # Create the new source line with ratios prepended
                    # W_0.92 C_0.90 DW_0.75 DTL_1.00 DTD_1.00 L_0.92 WR_1.03
                    new_source_line = f"W_{word_count_ratio} C_{length_ratio} DW_{dw_ratio} DTL_{dtl_ratio} " \
                                      f"DTD_{max_dep_depth_ratio} L_{leven_ratio} WR_{freqrank_ratio} {source_sentence}"

                    # Write the new source line to the output file
                    output_file.write(new_source_line)
                else:
                    print(f"Warning: No ratio data available for line {line_number}")
        print("Adding ratio prefix task completed!")
    else:
        print("Mismatch in row counts:")
        print("WC rows:", len(ratios_df_wc), "Expected:", num_source_sentences)
        print("Length rows:", len(ratios_df_length), "Expected:", num_source_sentences)
        print("Leven rows:", len(ratios_df_leven), "Expected:", num_source_sentences)
        print("FreqRank rows:", len(ratios_df_freqrank), "Expected:", num_source_sentences)
        print("DTD rows:", len(ratios_df_dtd), "Expected:", num_source_sentences)
        print("DTL rows:", len(ratios_df_dtl), "Expected:", num_source_sentences)
        print("DW rows:", len(ratios_df_dw), "Expected:", num_source_sentences)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_complex_src_path", required=False, help="Target sentence's dependency depth")
    parser.add_argument("--dataset_simple_tgt_path", required=False, help="Target sentence's dependency length")
    parser.add_argument("--ratio_files", required=False, help="Target sentence's no of difficult words")
    parser.add_argument("--fixed_ratios", required=False, help="Target sentence's no of difficult words")
    parser.add_argument("--gold_ref_ratios", required=False, help="Target sentence's no of difficult words")
    parser.add_argument("--processed_data_save_path", required=False, help="Target sentence's frequency")
    parser.add_argument("--feature_names", required=False, help="Target sentence's frequency")
    args = vars(parser.parse_args())

    if args["fixed_ratios"] is not None:
        preprocess_dataset(dataset_complex_src_path=args["dataset_complex_src_path"],
                           dataset_simple_tgt_path=args["dataset_simple_tgt_path"],
                           ratio_files=None,
                           processed_data_save_path=args["processed_data_save_path"],
                           fixed_ratios=args["fixed_ratios"],
                           gold_ref_ratios=None,
                           feature_names=args["feature_names"].split(","))
    elif args["gold_ref_ratios"] is not None:
        preprocess_dataset(dataset_complex_src_path=args["dataset_complex_src_path"],
                           dataset_simple_tgt_path=args["dataset_simple_tgt_path"],
                           ratio_files=None,
                           processed_data_save_path=args["processed_data_save_path"],
                           fixed_ratios=None,
                           gold_ref_ratios=args["gold_ref_ratios"],
                           feature_names=args["feature_names"].split(","))
    else:
        preprocess_dataset(dataset_complex_src_path=args["dataset_complex_src_path"],
                           dataset_simple_tgt_path=args["dataset_simple_tgt_path"],
                           ratio_files=args["ratio_files"],
                           processed_data_save_path=args["processed_data_save_path"],
                           fixed_ratios=None,
                           gold_ref_ratios=None,
                           feature_names=args["feature_names"].split(","))