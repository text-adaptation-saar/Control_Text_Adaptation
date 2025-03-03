"""
THIS IS A STAND-ALONE SCRIPT TO CALCULATE CONTROL TOKEN FEATURE VALUES OF ACTUAL MODEL PREDICTED SENTENCES VS.
 INPUT SENTENCES.
This script takes command line arguments.
"""
import argparse
from llm_based_control_rewrite.utils.helpers import load_yaml, yield_lines_in_parallel
from llm_based_control_rewrite.utils.feature_extraction import feature_bins_bundle_sentence
from llm_based_control_rewrite.utils.prepare_word_embeddings_frequency_ranks import load_ranks
from llm_based_control_rewrite.utils.feature_bin_preparation import create_bins
import matplotlib.pyplot as plt

def calculate_actual_feature_values_from_output(FEATURES_REQUESTED, LANG, in_src_PATH, in_tgt_PATH,
                                                output_file_full_path, feature_bins,
                                                frequency_ranks, absolute=False, freq_rank_cal_property = "q3,log"):

    actual_feature_value_file = open(output_file_full_path, "w", encoding="utf-8")

    feature_dict_vals = {feat: [] for feat in FEATURES_REQUESTED}

    for src_sent, tgt_sent in yield_lines_in_parallel([in_src_PATH, in_tgt_PATH], strict=True):
        f_vals_bin, f_vals_exact = feature_bins_bundle_sentence(src_sent, tgt_sent, LANG, FEATURES_REQUESTED,
                                                                feature_bins, frequency_ranks, absolute, freq_rank_cal_property)

        # mapping: feature: special_token
        feature2spec_token = {"dependency_depth": "MaxDepDepth", "dependency_length": "MaxDepLength",
                              "difficult_words": "DiffWords", "word_count": "WordCount",
                              "frequency": "FreqRank", "length": "Length", "levenshtein": "Leven",
                              "grade": "Grade"}

        to_be_prepended = ""
        for f in FEATURES_REQUESTED:
            v = str(round(f_vals_exact[f], 2))
            prep = feature2spec_token[f] + ", " + v + ", "
            to_be_prepended += prep

        actual_feature_value_file.write(to_be_prepended + "\n")

        # Save feature exact values to calculate average.
        for f, v in f_vals_exact.items(): # bin or exact value
            feature_dict_vals[f].append(v)

    # close the output file.
    actual_feature_value_file.close()
    print("Actual feature values saved in: %s" %(output_file_full_path))

    return feature_dict_vals

def plot_histogram_for_output_actual_feature_values(x, feature_name, lang, actual_feature_value_file_path):
    n_bins = 80  # 50 in the binned version for NLG
    if feature_name in {"levenshtein"}:
        n_bins = 50  # 30 in the binned version for NLG
    plt.hist(x, density=False, bins=n_bins)  # density=False would make counts
    plt.ylabel('Count')
    plt.xlabel(feature_name)
    plt.title("Histogram for " + feature_name + " in " + lang)
    out_name = str(actual_feature_value_file_path + "_" + feature_name) + ".png"
    plt.savefig(out_name)
    plt.clf()

def calcuate_feature_values(LANG, in_src_PATH, in_tgt_PATH, actual_feature_value_file_path, analyze_features,
                            requested_dependency_depth=None, requested_dependency_length=None,
                            requested_difficult_words=None, requested_word_count=None,
                            requested_frequency=None, requested_length=None, requested_levenshtein=None,
                            requested_grade_level=None,
                            absolute=False, output_file_prefix="", freq_rank_cal_property = "q3,log"):

    lang_allowed = {"en": "English", "de": "German"}
    features_allowed = {"dependency_depth", "dependency_length", "difficult_words", "word_count",
                        "frequency", "length", "levenshtein", "grade"}

    requested_feature_dict_vals = {}
    if absolute:
        output_file_name = "absolute_" + output_file_prefix + "_"
    else:
        output_file_name = "ratio_"

    if requested_dependency_depth:
        requested_feature_dict_vals["dependency_depth"] = requested_dependency_depth
        output_file_name += "maxdepdepth_" + str(requested_dependency_depth) + "_"
    if requested_dependency_length:
        requested_feature_dict_vals["dependency_length"] = requested_dependency_length
        output_file_name += "maxdeplength_" + str(requested_dependency_length) + "_"
    if requested_difficult_words:
        requested_feature_dict_vals["difficult_words"] = requested_difficult_words
        output_file_name += "diffwordscount_" + str(requested_difficult_words) + "_"
    if requested_word_count:
        requested_feature_dict_vals["word_count"] = requested_word_count
        output_file_name += "avgwordcount_" + str(requested_word_count) + "_"
    if requested_frequency:
        requested_feature_dict_vals["frequency"] = requested_frequency
        output_file_name += "freqrank_" + str(requested_frequency) + "_"
    if requested_length:
        requested_feature_dict_vals["length"] = requested_length
        output_file_name += "length_" + str(requested_length) + "_"
    if requested_levenshtein:
        requested_feature_dict_vals["levenshtein"] = requested_levenshtein
        output_file_name += "leven_" + str(requested_levenshtein) + "_"
    if requested_grade_level:
        requested_feature_dict_vals["grade"] = requested_grade_level
        output_file_name += "grade_" + str(requested_grade_level)

    output_file_full_path = actual_feature_value_file_path + "/" + output_file_name + ".csv"  # output_actual_feature_value.csv

    FEATURES_REQUESTED = requested_feature_dict_vals.keys()

    # some checks
    assert LANG in lang_allowed
    assert set(FEATURES_REQUESTED).issubset(features_allowed)
    print("... Generating actual feature values from the model generated output sentences. "
          "Here we are comparing test input sentences vs model generated output sentences for the requested control features: %s " % (
              ", ".join(FEATURES_REQUESTED)))

    feature_bins = create_bins()
    frequency_ranks = load_ranks(LANG)

    all_phases_all_feature_values = {feat: [] for feat in FEATURES_REQUESTED}

    feature_d_values = calculate_actual_feature_values_from_output(FEATURES_REQUESTED, LANG, in_src_PATH, in_tgt_PATH,
                                                                                     output_file_full_path,
                                                                   feature_bins, frequency_ranks, absolute, freq_rank_cal_property)

    average_feature_value_dict={}
    for f, v in feature_d_values.items():
        """ f is a str, v is a list """
        average = 0
        average = sum(v) / float(len(v))
        print("... Requested %s is %s and actual obtained average value is: %s " % (
        f, requested_feature_dict_vals[f], average))
        average_feature_value_dict[f] = average
        all_phases_all_feature_values[f].extend(v)

    if analyze_features:
        for f, _x in all_phases_all_feature_values.items():
            plot_histogram_for_output_actual_feature_values(_x, f, LANG, actual_feature_value_file_path + "/" + output_file_prefix)

    print("Finished actual feature value calculation!")
    return average_feature_value_dict, output_file_full_path

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="yaml config file for feature based evaluation")
    parser.add_argument("--requested_dependency_depth", required=False, help="requested sentence length in testing")
    parser.add_argument("--requested_dependency_length", required=False, help="requested sentence length in testing")
    parser.add_argument("--requested_difficult_words", required=False, help="requested sentence length in testing")
    parser.add_argument("--requested_frequency", required=False, help="requested sentence length in testing")
    parser.add_argument("--requested_length", required=False, help="requested sentence length in testing")
    parser.add_argument("--requested_levenshtein", required=False, help="requested sentence length in testing")
    parser.add_argument("--requested_word_count", required=False, help="requested output sentence word count in testing")
    parser.add_argument("--requested_grade_level", required=False, help="requested output sentence fkgl grade level in testing")
    parser.add_argument("--requested_absolute_value", required=False, help="")
    parser.add_argument("--output_file_prefix", required=False, help="")
    args = vars(parser.parse_args())
    config = load_yaml(args["config"])
    LANG = config["lang"].lower()  # required to load freq rank
    in_src_PATH = config["path_to_input_test_data"]
    in_tgt_PATH = config["path_to_output_generation"]
    actual_feature_value_file_path = config["path_to_output_actual_feature_value"]
    analyze_features=config["analyze_features"]
    freq_rank_cal_property = None if "freq_rank_cal_property" not in config else config["freq_rank_cal_property"]

    output_file_prefix = ""
    if args["output_file_prefix"] is not None:
        output_file_prefix = args["output_file_prefix"]
    else:
        output_file_prefix = "" if "output_file_prefix" not in config else config["output_file_prefix"]

    calcuate_feature_values(LANG=LANG, in_src_PATH=in_src_PATH, in_tgt_PATH=in_tgt_PATH,
                            actual_feature_value_file_path=actual_feature_value_file_path,
                            analyze_features=analyze_features,
                            requested_dependency_depth=args["requested_dependency_depth"],
                            requested_dependency_length=args["requested_dependency_length"],
                            requested_difficult_words=args["requested_difficult_words"],
                            requested_word_count=args["requested_word_count"],
                            requested_frequency=args["requested_frequency"],
                            requested_length=args["requested_length"],
                            requested_levenshtein=args["requested_levenshtein"],
                            requested_grade_level=args["requested_grade_level"],
                            absolute=args["requested_absolute_value"] if args["requested_absolute_value"] is not None else False,
                            output_file_prefix=output_file_prefix,
                            freq_rank_cal_property=freq_rank_cal_property)

# python llm_based_control_rewrite/evaluation/feature_based_evaluation.py --config configs/feature_value_generation_en.yaml  \
#  --requested_dependency_depth -1 \
#  --requested_dependency_length -1 \
#  --requested_frequency -1  \
#  --requested_length -1 \
#  --requested_levenshtein -1 \
#  --requested_word_count -1
