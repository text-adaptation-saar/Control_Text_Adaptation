import argparse

import pandas as pd

from llm_based_control_rewrite.utils.helpers import load_yaml, yield_lines_in_parallel


def calculate_ratios(absolute_feature_value_src_PATH, absolute_feature_value_tgt_PATH, ratio_calculation_output_path):

    print("... Started to calculate Ratio values from absolute feature values Input and Output files ...")
    all_length_ratio = []
    all_max_dep_depth_ratio = []
    all_max_dep_length_ratio = []
    all_diffwords_rank_ratio = []
    all_freq_rank_ratio = []
    all_leven_ratio = []
    all_word_count_ratio = []
    all_grade_ratio = []

    with open(ratio_calculation_output_path+"/ratio_stats.csv", "a") as output_ratio_write_file:
        count = 0
        column_names = "Line,abs_src_MaxDepDepth,abs_src_MaxDepLength,abs_src_DiffWords,abs_src_WordCount," \
                       "abs_tgt_MaxDepDepth,abs_tgt_MaxDepLength,abs_tgt_DiffWords,abs_tgt_WordCount," \
                       "MaxDepDepth_ratio,MaxDepLength_ratio,DiffWords_ratio,WordCount_ratio"
        # column_names = "Line,abs_src_MaxDepDepth,abs_src_MaxDepLength,abs_src_DiffWords,abs_src_WordCount,abs_src_FreqRank," \
        #                "abs_src_Length,abs_src_Leven,abs_src_Grade," \
        #                "abs_tgt_MaxDepDepth,abs_tgt_MaxDepLength,abs_tgt_DiffWords,abs_tgt_WordCount,abs_tgt_FreqRank," \
        #                "abs_tgt_Length,abs_tgt_Leven,abs_tgt_Grade," \
        #                "MaxDepDepth_ratio,MaxDepLength_ratio,DiffWords_ratio,WordCount_ratio,FreqRank_ratio," \
        #                "Length_ratio,Leven_ratio,Grade_ratio"
        output_ratio_write_file.write(column_names + "\n")

        for src_sent, tgt_sent in yield_lines_in_parallel([absolute_feature_value_src_PATH, absolute_feature_value_tgt_PATH], strict=True):
            # MaxDepDepth, 5, MaxDepLength, 7, DiffWords, 3, WordCount, 11.0, Length, 54.5, Leven, 0.51, Grade, 4.0,
            # MaxDepDepth, 7, MaxDepLength, 2, DiffWords, 3, WordCount, 11.0, FreqRank, 10.55, Length, 52.0, Leven, 0.56, Grade, 4.0,

            src_value_split = src_sent.split(",")
            tgt_value_split = tgt_sent.split(",")
            print(src_value_split,  "\t", tgt_value_split)

            src_maxdepdepth = float(src_value_split[1].strip())
            src_maxdeplength = float(src_value_split[3].strip())
            src_diffwords = float(src_value_split[5].strip())
            src_wordcount = float(src_value_split[7].strip())
            # src_freqrank = float(src_value_split[9].strip())
            # src_length = float(src_value_split[11].strip())
            # src_leven = float(src_value_split[13].strip())
            # src_grade = float(src_value_split[15].strip())

            tgt_maxdepdepth = float(tgt_value_split[1].strip())
            tgt_maxdeplength = float(tgt_value_split[3].strip())
            tgt_diffwords = float(tgt_value_split[5].strip())
            tgt_wordcount = float(tgt_value_split[7].strip())
            # tgt_freqrank = float(tgt_value_split[9].strip())
            # tgt_length = float(tgt_value_split[11].strip())
            # tgt_leven = float(tgt_value_split[13].strip())
            # tgt_grade = float(tgt_value_split[15].strip())

            # ratio = tgt/src
            max_dep_depth_ratio = round((tgt_maxdepdepth if tgt_maxdepdepth != 0 else 0.5) / (src_maxdepdepth if src_maxdepdepth != 0 else 0.5), 2)
            max_dep_length_ratio = round((tgt_maxdeplength if tgt_maxdeplength != 0 else 0.5) / (src_maxdeplength if src_maxdeplength != 0 else 0.5), 2)
            diffwords_rank_ratio = round((tgt_diffwords if tgt_diffwords != 0 else 0.5) / (src_diffwords if src_diffwords != 0 else 0.5), 2)
            word_count_ratio = round(tgt_wordcount/src_wordcount, 2)
            # freq_rank_ratio = round((tgt_freqrank if tgt_freqrank != 0 else 0.5) / (src_freqrank if src_freqrank != 0 else 0.5), 2)
            # length_ratio = round(tgt_length/src_length, 2)
            # leven_ratio = round(tgt_leven, 2)
            # grade_ratio = round((tgt_grade if tgt_grade != 0 else 0.5) / (src_grade if src_grade != 0 else 0.5), 2)
            # # leven_ratio = round(float(tgt_value_split[9].strip())/float(src_value_split[9].strip()), 2)

            all_max_dep_depth_ratio.append(max_dep_depth_ratio)
            all_max_dep_length_ratio.append(max_dep_length_ratio)
            all_diffwords_rank_ratio.append(diffwords_rank_ratio)
            all_word_count_ratio.append(word_count_ratio)
            # all_freq_rank_ratio.append(freq_rank_ratio)
            # all_length_ratio.append(length_ratio)
            # all_leven_ratio.append(leven_ratio)
            # all_grade_ratio.append(grade_ratio)

            count += 1
            ratio_value_final_line = f"{count},{src_maxdepdepth},{src_maxdeplength},{src_diffwords},{src_wordcount}," \
                                     f"{tgt_maxdepdepth},{tgt_maxdeplength},{tgt_diffwords},{tgt_wordcount}," \
                                     f"{max_dep_depth_ratio},{max_dep_length_ratio},{diffwords_rank_ratio},{word_count_ratio}"
            # ratio_value_final_line = f"{count},{src_maxdepdepth},{src_maxdeplength},{src_diffwords},{src_wordcount},{src_freqrank},{src_length},{src_leven},{src_grade}," \
            #                                      f"{tgt_maxdepdepth},{tgt_maxdeplength},{tgt_diffwords},{tgt_wordcount},{tgt_freqrank},{tgt_length},{tgt_leven},{tgt_grade}," \
            #                                      f"{max_dep_depth_ratio},{max_dep_length_ratio},{diffwords_rank_ratio},{word_count_ratio},{freq_rank_ratio},{length_ratio},{leven_ratio},{grade_ratio}"

            # print("\n", ratio_value_final_line, "\n")
            output_ratio_write_file.write(ratio_value_final_line + "\n")

        avg_feature_dict = {}
        avg_feature_dict["avg_MaxDepDepth_ratio"] = sum(all_max_dep_depth_ratio) / float(len(all_max_dep_depth_ratio))
        avg_feature_dict["avg_MaxDepLength_ratio"] = sum(all_max_dep_length_ratio) / float(len(all_max_dep_length_ratio))
        avg_feature_dict["avg_DiffWords_ratio"] = sum(all_diffwords_rank_ratio) / float(len(all_diffwords_rank_ratio))
        avg_feature_dict["avg_WordCount_ratio"] = sum(all_word_count_ratio) / float(len(all_word_count_ratio))
        # avg_feature_dict["avg_FreqRank_ratio"] = sum(all_freq_rank_ratio) / float(len(all_freq_rank_ratio))
        # avg_feature_dict["avg_Length_ratio"] = sum(all_length_ratio) / float(len(all_length_ratio))
        # avg_feature_dict["avg_Leven_ratio"] = sum(all_leven_ratio) / float(len(all_leven_ratio))
        # avg_feature_dict["avg_Grade_ratio"] = sum(all_grade_ratio) / float(len(all_grade_ratio))

        print("... Generating Ratio values from absolute feature values Input and Output files ...\n"
              "absolute Input feature file:%s\nabsolute Output feature file:%s" % (absolute_feature_value_src_PATH, absolute_feature_value_tgt_PATH))
        for k, v in avg_feature_dict.items():
            print("... %s is: %s " % (
                k, v))
        return avg_feature_dict


def calculate_ratios_reverse_way_for_syn_gen_data(absolute_feature_value_src_PATH, absolute_feature_value_tgt_PATH, ratio_calculation_output_path):

    print("... Started to calculate Ratio values from absolute feature values Input and Output files ...")
    all_max_dep_depth_ratio = []
    all_max_dep_length_ratio = []
    all_diffwords_rank_ratio = []
    all_word_count_ratio = []
    all_grade_ratio = []

    # Read source CSV file into a DataFrame
    src_df = pd.read_csv(absolute_feature_value_src_PATH)

    with open(ratio_calculation_output_path + "/ratio_stats_reverse_way.csv", "a") as output_ratio_write_file, \
            open(absolute_feature_value_tgt_PATH, "r") as tgt_file:

        count = 0
        column_names = "Line,abs_src_MaxDepDepth,abs_src_MaxDepLength,abs_src_DiffWords,abs_src_WordCount," \
                       "abs_src_FKGL_Grade," \
                       "abs_tgt_MaxDepDepth,abs_tgt_MaxDepLength,abs_tgt_DiffWords,abs_tgt_WordCount," \
                       "abs_tgt_FKGL_Grade," \
                       "MaxDepDepth_ratio,MaxDepLength_ratio,DiffWords_ratio,WordCount_ratio," \
                        "FKGL_Grade_ratio"
        output_ratio_write_file.write(column_names + "\n")

        for tgt_line in tgt_file:
            # Extract target feature values for the source features from the CSV
            # Ensure we are using the correct column names from the CSV
            # in syntactic_train_subset.csv, abs_tgt_ are mono original target sentences, and src is generated syntactic values,
            # which here used to generate syn.sentences
            tgt_maxdepdepth = float(src_df["abs_tgt_MaxDepDepth"].iloc[count])
            tgt_maxdeplength = float(src_df["abs_tgt_MaxDepLength"].iloc[count])
            tgt_diffwords = float(src_df["abs_tgt_DiffWords"].iloc[count])
            tgt_wordcount = float(src_df["abs_tgt_WordCount"].iloc[count])
            tgt_fkgl_grade = float(src_df["abs_tgt_FKGL_Grade"].iloc[count])

            # MaxDepDepth, 4, MaxDepLength, 2, DiffWords, 2, WordCount, 6.5,
            # Extract target feature values from the text file
            print(f"tgt_line: {tgt_line}")
            tgt_values = tgt_line.strip().split(",")
            src_maxdepdepth = float(tgt_values[1].strip())
            src_maxdeplength = float(tgt_values[3].strip())
            src_diffwords = float(tgt_values[5].strip())
            src_wordcount = float(tgt_values[7].strip())
            src_fkgl_grade = float(tgt_values[9].strip())

            # ratio = tgt/src
            max_dep_depth_ratio = round((tgt_maxdepdepth if tgt_maxdepdepth != 0 else 0.5) / (src_maxdepdepth if src_maxdepdepth != 0 else 0.5), 2)
            max_dep_length_ratio = round((tgt_maxdeplength if tgt_maxdeplength != 0 else 0.5) / (src_maxdeplength if src_maxdeplength != 0 else 0.5), 2)
            diffwords_rank_ratio = round((tgt_diffwords if tgt_diffwords != 0 else 0.5) / (src_diffwords if src_diffwords != 0 else 0.5), 2)
            word_count_ratio = round(tgt_wordcount / src_wordcount, 2)
            # this is how fkgl grade ratio calcualted for phase-1 project. using
            # llm_based_control_rewrite/utils/ratio_file_with_grade_level.py
            fkgl_ratio = round(tgt_fkgl_grade / (0.5 if src_fkgl_grade == 0 else src_fkgl_grade), 2)

            all_max_dep_depth_ratio.append(max_dep_depth_ratio)
            all_max_dep_length_ratio.append(max_dep_length_ratio)
            all_diffwords_rank_ratio.append(diffwords_rank_ratio)
            all_word_count_ratio.append(word_count_ratio)
            all_grade_ratio.append(fkgl_ratio)

            count += 1
            ratio_value_final_line = f"{count},{src_maxdepdepth},{src_maxdeplength},{src_diffwords},{src_wordcount}," \
                                     f"{src_fkgl_grade}," \
                                     f"{tgt_maxdepdepth},{tgt_maxdeplength},{tgt_diffwords},{tgt_wordcount}," \
                                     f"{tgt_fkgl_grade}," \
                                     f"{max_dep_depth_ratio},{max_dep_length_ratio},{diffwords_rank_ratio},{word_count_ratio}," \
                                     f"{fkgl_ratio}"
            output_ratio_write_file.write(ratio_value_final_line + "\n")

        avg_feature_dict = {}
        avg_feature_dict["avg_MaxDepDepth_ratio"] = sum(all_max_dep_depth_ratio) / float(len(all_max_dep_depth_ratio))
        avg_feature_dict["avg_MaxDepLength_ratio"] = sum(all_max_dep_length_ratio) / float(len(all_max_dep_length_ratio))
        avg_feature_dict["avg_DiffWords_ratio"] = sum(all_diffwords_rank_ratio) / float(len(all_diffwords_rank_ratio))
        avg_feature_dict["avg_WordCount_ratio"] = sum(all_word_count_ratio) / float(len(all_word_count_ratio))
        avg_feature_dict["avg_FKGL_Grade_ratio"] = sum(all_grade_ratio) / float(len(all_grade_ratio))

        print("... Generating Ratio values from absolute feature values Input and Output files ...\n"
              "absolute Input feature file:%s\nabsolute Output feature file:%s" % (absolute_feature_value_src_PATH, absolute_feature_value_tgt_PATH))
        for k, v in avg_feature_dict.items():
            print("... %s is: %s " % (
                k, v))
        return avg_feature_dict


def calculate_ratios_reverse_way_for_syn_gen_data_both_feature_values_calcuated(absolute_feature_value_src_PATH, absolute_feature_value_tgt_PATH, ratio_calculation_output_path):

    print("... Started to calculate Ratio values from absolute feature values Input and Output files ...")
    all_max_dep_depth_ratio = []
    all_max_dep_length_ratio = []
    all_diffwords_rank_ratio = []
    all_word_count_ratio = []
    all_grade_ratio = []

    with open(ratio_calculation_output_path+"/ratio_stats_reverse_way.csv", "a") as output_ratio_write_file:
        count = 0
        column_names = "Line,abs_src_MaxDepDepth,abs_src_MaxDepLength,abs_src_DiffWords,abs_src_WordCount," \
                       "abs_src_FKGL_Grade," \
                       "abs_tgt_MaxDepDepth,abs_tgt_MaxDepLength,abs_tgt_DiffWords,abs_tgt_WordCount," \
                       "abs_tgt_FKGL_Grade," \
                       "MaxDepDepth_ratio,MaxDepLength_ratio,DiffWords_ratio,WordCount_ratio," \
                        "FKGL_Grade_ratio"
        output_ratio_write_file.write(column_names + "\n")

        for src_sent, tgt_sent in yield_lines_in_parallel([absolute_feature_value_src_PATH, absolute_feature_value_tgt_PATH], strict=True):
            # since it is synthetic data gen, assign output.txt feature values as abs_src_feature.
            # aassign input.txt features to abs_tgt_feature values.
            print(f"src_sent: {src_sent}")
            src_value_split = src_sent.split(",")
            tgt_maxdepdepth = float(src_value_split[1].strip())
            tgt_maxdeplength = float(src_value_split[3].strip())
            tgt_diffwords = float(src_value_split[5].strip())
            tgt_wordcount = float(src_value_split[7].strip())
            tgt_fkgl_grade = float(src_value_split[9].strip())

            # MaxDepDepth, 4, MaxDepLength, 2, DiffWords, 2, WordCount, 6.5,
            # Extract target feature values from the text file
            print(f"tgt_line: {tgt_sent}")
            tgt_values = tgt_sent.strip().split(",")
            src_maxdepdepth = float(tgt_values[1].strip())
            src_maxdeplength = float(tgt_values[3].strip())
            src_diffwords = float(tgt_values[5].strip())
            src_wordcount = float(tgt_values[7].strip())
            src_fkgl_grade = float(tgt_values[9].strip())

            # ratio = tgt/src
            max_dep_depth_ratio = round((tgt_maxdepdepth if tgt_maxdepdepth != 0 else 0.5) / (src_maxdepdepth if src_maxdepdepth != 0 else 0.5), 2)
            max_dep_length_ratio = round((tgt_maxdeplength if tgt_maxdeplength != 0 else 0.5) / (src_maxdeplength if src_maxdeplength != 0 else 0.5), 2)
            diffwords_rank_ratio = round((tgt_diffwords if tgt_diffwords != 0 else 0.5) / (src_diffwords if src_diffwords != 0 else 0.5), 2)
            word_count_ratio = round(tgt_wordcount / src_wordcount, 2)
            # this is how fkgl grade ratio calcualted for phase-1 project. using
            # llm_based_control_rewrite/utils/ratio_file_with_grade_level.py
            fkgl_ratio = round(tgt_fkgl_grade / (0.5 if src_fkgl_grade == 0 else src_fkgl_grade), 2)

            all_max_dep_depth_ratio.append(max_dep_depth_ratio)
            all_max_dep_length_ratio.append(max_dep_length_ratio)
            all_diffwords_rank_ratio.append(diffwords_rank_ratio)
            all_word_count_ratio.append(word_count_ratio)
            all_grade_ratio.append(fkgl_ratio)

            count += 1
            ratio_value_final_line = f"{count},{src_maxdepdepth},{src_maxdeplength},{src_diffwords},{src_wordcount}," \
                                     f"{src_fkgl_grade}," \
                                     f"{tgt_maxdepdepth},{tgt_maxdeplength},{tgt_diffwords},{tgt_wordcount}," \
                                     f"{tgt_fkgl_grade}," \
                                     f"{max_dep_depth_ratio},{max_dep_length_ratio},{diffwords_rank_ratio},{word_count_ratio}," \
                                     f"{fkgl_ratio}"
            output_ratio_write_file.write(ratio_value_final_line + "\n")

        avg_feature_dict = {}
        avg_feature_dict["avg_MaxDepDepth_ratio"] = sum(all_max_dep_depth_ratio) / float(len(all_max_dep_depth_ratio))
        avg_feature_dict["avg_MaxDepLength_ratio"] = sum(all_max_dep_length_ratio) / float(len(all_max_dep_length_ratio))
        avg_feature_dict["avg_DiffWords_ratio"] = sum(all_diffwords_rank_ratio) / float(len(all_diffwords_rank_ratio))
        avg_feature_dict["avg_WordCount_ratio"] = sum(all_word_count_ratio) / float(len(all_word_count_ratio))
        avg_feature_dict["avg_FKGL_Grade_ratio"] = sum(all_grade_ratio) / float(len(all_grade_ratio))

        print("... Generating Ratio values from absolute feature values Input and Output files ...\n"
              "absolute Input feature file:%s\nabsolute Output feature file:%s" % (absolute_feature_value_src_PATH, absolute_feature_value_tgt_PATH))
        for k, v in avg_feature_dict.items():
            print("... %s is: %s " % (
                k, v))
        return avg_feature_dict


def calculate_ratios_for_grade(absolute_feature_value_src_PATH, absolute_feature_value_tgt_PATH, ratio_calculation_output_path):

    print("... Started to calculate Ratio values from absolute feature values Input and Output files ...")
    all_grade_ratio = []

    with open(ratio_calculation_output_path+"/ratio_stats.csv", "a") as output_ratio_write_file:
        count = 0
        column_names = "Line,abs_src_Grade," \
                       "abs_tgt_Grade," \
                       "Grade_ratio"
        output_ratio_write_file.write(column_names + "\n")

        for src_sent, tgt_sent in yield_lines_in_parallel([absolute_feature_value_src_PATH, absolute_feature_value_tgt_PATH], strict=True):
            # Grade, 4.0

            src_value_split = src_sent.split(",")
            tgt_value_split = tgt_sent.split(",")
            print(src_value_split,  "\t", tgt_value_split)

            src_grade = float(src_value_split[1].strip())
            tgt_grade = float(tgt_value_split[1].strip())

            # ratio = tgt/src
            grade_ratio = round((tgt_grade if tgt_grade != 0 else 0.5) / (src_grade if src_grade != 0 else 0.5), 2)
            # leven_ratio = round(float(tgt_value_split[9].strip())/float(src_value_split[9].strip()), 2)

            all_grade_ratio.append(grade_ratio)

            count += 1
            ratio_value_final_line = f"{count},{src_grade}," \
                                     f"{tgt_grade}," \
                                     f"{grade_ratio}"

            # print("\n", ratio_value_final_line, "\n")
            output_ratio_write_file.write(ratio_value_final_line + "\n")

        avg_feature_dict = {}
        avg_feature_dict["avg_Grade_ratio"] = sum(all_grade_ratio) / float(len(all_grade_ratio))

        print("... Generating Ratio values from absolute feature values Input and Output files ...\n"
              "absolute Input feature file:%s\nabsolute Output feature file:%s" % (absolute_feature_value_src_PATH, absolute_feature_value_tgt_PATH))
        for k, v in avg_feature_dict.items():
            print("... %s is: %s " % (
                k, v))
        return avg_feature_dict

def calculate_ratios_for_single_feature(absolute_feature_value_src_PATH, absolute_feature_value_tgt_PATH, ratio_calculation_output_path, column_number):

    print("... Started to calculate Ratio values from absolute feature values Input and Output files ...")
    all_freq_rank_ratio = []

    with open(ratio_calculation_output_path+"/ratio_stats.csv", "a") as output_ratio_write_file:
        count = 0
        # column_names = "Line,abs_src_FreqRank,abs_tgt_FreqRank,FreqRank_ratio"
        column_names = "Line,src_difficult_words,tgt_difficult_words,difficult_words_ratio"
        output_ratio_write_file.write(column_names + "\n")

        for src_sent, tgt_sent in yield_lines_in_parallel([absolute_feature_value_src_PATH, absolute_feature_value_tgt_PATH], strict=True):
            # FreqRank, 9.03
            src_value_split = src_sent.split(",")
            tgt_value_split = tgt_sent.split(",")
            print(src_value_split,  "\t", tgt_value_split)

            src_freq = float(src_value_split[column_number].strip())
            # src_freq = float(src_value_split[1].strip())
            tgt_freq = float(tgt_value_split[column_number].strip())

            freq_rank_ratio = round((tgt_freq if tgt_freq != 0 else 0.5) / (src_freq if src_freq != 0 else 0.5), 2)
            all_freq_rank_ratio.append(freq_rank_ratio)


            count += 1
            ratio_value_final_line = "%s,%s,%s,%s" % (count,src_freq,tgt_freq,freq_rank_ratio)
            # print("\n", ratio_value_final_line, "\n")
            output_ratio_write_file.write(ratio_value_final_line + "\n")

        avg_feature_dict = {}
        # avg_feature_dict["avg_FreqRank_ratio"] = sum(all_freq_rank_ratio) / float(len(all_freq_rank_ratio))
        avg_feature_dict["avg_difficult_words_ratio"] = sum(all_freq_rank_ratio) / float(len(all_freq_rank_ratio))

        print("... Generating Ratio values from absolute feature values Input and Output files ...\n"
              "absolute Input feature file:%s\nabsolute Output feature file:%s" % (absolute_feature_value_src_PATH, absolute_feature_value_tgt_PATH))
        for k, v in avg_feature_dict.items():
            print("... %s is: %s " % (
                k, v))
        return avg_feature_dict


def calculate_ratios_for_single_feature_and_saveto_existing_ratio_csv(absolute_feature_value_src_PATH, absolute_feature_value_tgt_PATH,
                                        already_existing_ratio_file, column_number):

    print("... Started to calculate Ratio values from absolute feature values Input and Output files ...")
    all_freq_rank_ratio = []
    all_src_freq_rank = []
    all_tgt_freq_rank = []

    for src_sent, tgt_sent in yield_lines_in_parallel([absolute_feature_value_src_PATH, absolute_feature_value_tgt_PATH], strict=True):
        src_value_split = src_sent.split(",")
        tgt_value_split = tgt_sent.split(",")
        print(src_value_split, "\t", tgt_value_split)

        src_freq = float(src_value_split[column_number].strip())
        tgt_freq = float(tgt_value_split[column_number].strip())

        freq_rank_ratio = round((tgt_freq if tgt_freq != 0 else 0.5) / (src_freq if src_freq != 0 else 0.5), 2)
        all_freq_rank_ratio.append(freq_rank_ratio)
        all_src_freq_rank.append(src_freq)
        all_tgt_freq_rank.append(tgt_freq)

    # Loading existing ratio file
    df = pd.read_csv(already_existing_ratio_file)
    df['abs_src_FreqRank'] = all_src_freq_rank
    df['abs_tgt_FreqRank'] = all_tgt_freq_rank
    df['FreqRank_ratio'] = all_freq_rank_ratio

    # Saving updated DataFrame
    df.to_csv(already_existing_ratio_file, index=False)
    print("Updated DataFrame saved successfully.")

    avg_feature_dict = {}
    avg_feature_dict["avg_FreqRank_ratio"] = sum(all_freq_rank_ratio) / float(len(all_freq_rank_ratio))
    # avg_feature_dict["avg_difficult_words_ratio"] = sum(all_freq_rank_ratio) / float(len(all_freq_rank_ratio))

    print("... Generating Ratio values from absolute feature values Input and Output files ...\n"
          "absolute Input feature file:%s\nabsolute Output feature file:%s" % (absolute_feature_value_src_PATH, absolute_feature_value_tgt_PATH))
    for k, v in avg_feature_dict.items():
        print("... %s is: %s " % (k, v))
    return avg_feature_dict


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="yaml config file for feature based evaluation")
    args = vars(parser.parse_args())
    config = load_yaml(args["config"])

    absolute_feature_value_src_PATH = config["absolute_feature_value_src_PATH"]
    absolute_feature_value_tgt_PATH = config["absolute_feature_value_tgt_PATH"]
    ratio_calculation_output_path = config["ratio_calculation_output_path"]

    calculate_ratios(absolute_feature_value_src_PATH, absolute_feature_value_tgt_PATH, ratio_calculation_output_path)
    # calculate_ratios_for_single_feature_and_saveto_existing_ratio_csv(absolute_feature_value_src_PATH, absolute_feature_value_tgt_PATH,
    #                                                                   ratio_calculation_output_path, 1)

    # column_number = config["column_number"]
    # calculate_ratios_for_single_feature(absolute_feature_value_src_PATH, absolute_feature_value_tgt_PATH, ratio_calculation_output_path,column_number)

# python llm_based_control_rewrite/feature_value_calculation/ratio_calculation.py --config data_auxiliary/en/feature_distribution_analyse/Turkcorpus_for_prompt_experiments_TEST_50_dataset/tgt/feature_distribution_analyse.yaml