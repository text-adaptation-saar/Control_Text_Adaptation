#cat data_auxiliary/en/feature_values_analyse/WikiLarge_V2/splits/wikilarge_train_split_1/tgt/length_-1_dependency_depth_-1_dependency_length_-1_frequency_-1_levenshtein_-1_word_count_-1_.csv data_auxiliary/en/feature_values_analyse/WikiLarge_V2/splits/wikilarge_train_split_2/tgt/length_-1_dependency_depth_-1_dependency_length_-1_frequency_-1_levenshtein_-1_word_count_-1_.csv data_auxiliary/en/feature_values_analyse/WikiLarge_V2/splits/wikilarge_train_split_3/tgt/length_-1_dependency_depth_-1_dependency_length_-1_frequency_-1_levenshtein_-1_word_count_-1_.csv data_auxiliary/en/feature_values_analyse/WikiLarge_V2/splits/wikilarge_train_split_4/tgt/length_-1_dependency_depth_-1_dependency_length_-1_frequency_-1_levenshtein_-1_word_count_-1_.csv data_auxiliary/en/feature_values_analyse/WikiLarge_V2/splits/wikilarge_train_split_5/tgt/length_-1_dependency_depth_-1_dependency_length_-1_frequency_-1_levenshtein_-1_word_count_-1_.csv data_auxiliary/en/feature_values_analyse/WikiLarge_V2/splits/wikilarge_train_split_6/tgt/length_-1_dependency_depth_-1_dependency_length_-1_frequency_-1_levenshtein_-1_word_count_-1_.csv > data_auxiliary/en/feature_values_analyse/WikiLarge_V2/wikilarge_all_v2/tgt/length_-1_dependency_depth_-1_dependency_length_-1_frequency_-1_levenshtein_-1_word_count_-1_.csv
#
#cat data_auxiliary/en/feature_values_analyse/WikiLarge_V2/splits/wikilarge_train_split_1/src/length_-1_dependency_depth_-1_dependency_length_-1_frequency_-1_levenshtein_-1_word_count_-1_.csv data_auxiliary/en/feature_values_analyse/WikiLarge_V2/splits/wikilarge_train_split_2/src/length_-1_dependency_depth_-1_dependency_length_-1_frequency_-1_levenshtein_-1_word_count_-1_.csv data_auxiliary/en/feature_values_analyse/WikiLarge_V2/splits/wikilarge_train_split_3/src/length_-1_dependency_depth_-1_dependency_length_-1_frequency_-1_levenshtein_-1_word_count_-1_.csv data_auxiliary/en/feature_values_analyse/WikiLarge_V2/splits/wikilarge_train_split_4/src/length_-1_dependency_depth_-1_dependency_length_-1_frequency_-1_levenshtein_-1_word_count_-1_.csv data_auxiliary/en/feature_values_analyse/WikiLarge_V2/splits/wikilarge_train_split_5/src/length_-1_dependency_depth_-1_dependency_length_-1_frequency_-1_levenshtein_-1_word_count_-1_.csv data_auxiliary/en/feature_values_analyse/WikiLarge_V2/splits/wikilarge_train_split_6/src/length_-1_dependency_depth_-1_dependency_length_-1_frequency_-1_levenshtein_-1_word_count_-1_.csv > data_auxiliary/en/feature_values_analyse/WikiLarge_V2/wikilarge_all_v2/src/length_-1_dependency_depth_-1_dependency_length_-1_frequency_-1_levenshtein_-1_word_count_-1_.csv

#merge the features
#python llm_based_control_rewrite/feature_value_calculation/megre_feature_cal.py

#Ratio calculation
#python llm_based_control_rewrite/feature_value_calculation/ratio_calculation.py --config data_auxiliary/en/feature_values_analyse/WikiLarge_V2/wikilarge_all_v2/tgt/feature_distribution_analyse.yaml


# Generate plots
#abs_src_feature_list="src_difficult_words"
#ratio_feature_list="difficult_words_ratio"
abs_src_feature_list="abs_src_MaxDepDepth,abs_src_MaxDepLength,abs_src_DiffWords,abs_src_WordCount"
ratio_feature_list="MaxDepDepth_ratio,MaxDepLength_ratio,DiffWords_ratio,WordCount_ratio"
#x_min_value="0"
#x_max_value="10"
#y_max_value=2

python llm_based_control_rewrite/feature_value_calculation/plot.py \
--ratio_file data_auxiliary/en/feature_values_analyse/WikiLarge_V2/wikilarge_all_v2/tgt/ratio_stats.csv \
--save_plot_path data_auxiliary/en/feature_values_analyse/WikiLarge_V2/wikilarge_all_v2/tgt/1_ratio_vs_abs \
--abs_src_feature_list "$abs_src_feature_list" \
--ratio_feature_list "$ratio_feature_list" > data_auxiliary/en/feature_values_analyse/WikiLarge_V2/wikilarge_all_v2/tgt/1_ratio_vs_abs/logs

## Generate plots
#abs_src_feature_list="abs_src_MaxDepDepth,abs_src_MaxDepLength,abs_src_FreqRank,abs_src_WordCount"
#ratio_feature_list="MaxDepDepth_ratio,MaxDepLength_ratio,FreqRank_ratio,WordCount_ratio"
#x_min_value="2,2,7,6"
#x_max_value="11,20,12,40"
#y_max_value=2

#python llm_based_control_rewrite/feature_value_calculation/plot.py \
#--ratio_file data_auxiliary/en/feature_distribution_analyse/WikiLarge_V2/wikilarge_all_v2/tgt/ratio_stats.csv \
#--save_plot_path data_auxiliary/en/feature_distribution_analyse/WikiLarge_V2/wikilarge_all_v2/tgt/1_ratio_vs_abs \
#--abs_src_feature_list "$abs_src_feature_list" \
#--ratio_feature_list "$ratio_feature_list" > data_auxiliary/en/feature_distribution_analyse/WikiLarge_V2/wikilarge_all_v2/tgt/1_ratio_vs_abs/logs


##filter only ratio>2  (comment out the code in plot.py)
#python llm_based_control_rewrite/feature_value_calculation/plot.py \
#--ratio_file data_auxiliary/en/feature_distribution_analyse/WikiLarge_V2/wikilarge_all_v2/tgt/ratio_stats.csv \
#--abs_src_feature_list "$abs_src_feature_list" \
#--ratio_feature_list "$ratio_feature_list" \
#--save_plot_path data_auxiliary/en/feature_distribution_analyse/WikiLarge_V2/wikilarge_all_v2/tgt/2_filter_ratio_above_2 \
#--do_filtering \
#--x_min_value "$x_min_value" \
#--x_max_value "$x_max_value" \
#--y_max_value $y_max_value > data_auxiliary/en/feature_distribution_analyse/WikiLarge_V2/wikilarge_all_v2/tgt/2_filter_ratio_above_2/logs


##filtered ratio plots
#python llm_based_control_rewrite/feature_value_calculation/plot.py \
#--ratio_file data_auxiliary/en/feature_distribution_analyse/WikiLarge_V2/wikilarge_all_v2/tgt/ratio_stats.csv \
#--abs_src_feature_list "$abs_src_feature_list" \
#--ratio_feature_list "$ratio_feature_list" \
#--save_plot_path data_auxiliary/en/feature_distribution_analyse/WikiLarge_V2/wikilarge_all_v2/tgt/3_filter_ratio_and_abs_src_values \
#--do_filtering \
#--x_min_value "$x_min_value" \
#--x_max_value "$x_max_value" \
#--y_max_value $y_max_value > data_auxiliary/en/feature_distribution_analyse/WikiLarge_V2/wikilarge_all_v2/tgt/3_filter_ratio_and_abs_src_values/logs


#create train/eval/test data
#python llm_based_control_rewrite/utils/create_train_eval_test_dataset.py                                                                                                                            ─╯


##mutually_filtered_rattio plots
#python llm_based_control_rewrite/feature_value_calculation/plot.py \
#--ratio_file data_filtered/en/test_sets_from_EASSE/turk_corpus/test/mutually_filtered_ratio_$i.csv \
#--save_plot_path data_auxiliary/en/feature_distribution_analyse/test_sets_from_EASSE/turkcorpus/test/tgt_$i/4_mutually_filtered_ratio_and_abs_src_values \
#--abs_src_feature_list "$abs_src_feature_list" \
#--ratio_feature_list "$ratio_feature_list" > data_auxiliary/en/feature_distribution_analyse/test_sets_from_EASSE/turkcorpus/test/tgt_$i/logs_of_plot_generation_4_mutually_filtered_ratio_and_abs_src_values

