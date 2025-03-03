#!/bin/bash


#./experiments/x_ablation_study/config_file_populate_gold_ratio_free_style.sh
#
#./experiments/x_ablation_study/config_file_populate_gold_ratio_no_sys_prompt.sh
#
#./experiments/x_ablation_study/config_file_populate_gold_ratio_level-2.1.sh
#
#./experiments/x_ablation_study/config_file_populate_gold_ratio_level-2.sh



./experiments/x_ablation_study/config_file_populate_gold_ratio_level-3.sh

./experiments/x_ablation_study/config_file_populate_gold_ratio_level-4.sh

#./experiments/x_ablation_study/config_file_populate_gold_ratio_level-4-feedbackloop.sh









#!/bin/bash

#file_paths=("free_style-gold-filtered_wiki.valid_v1.1.src-100_gpt-4o-2024-05-13_examples_0_temp_0_chain_False"
#"no_sys_prompt-gold-filtered_wiki.valid_v1.1.src-100_gpt-4o-2024-05-13_examples_0_temp_0_chain_False"
#"level-2.1_prompt-gold-filtered_wiki.valid_v1.1.src-100_gpt-4o-2024-05-13_examples_0_temp_0_chain_False"
#"level-2_prompt-gold-filtered_wiki.valid_v1.1.src-100_gpt-4o-2024-05-13_examples_0_temp_0_chain_False"
#"level-3_prompt-gold-filtered_wiki.valid_v1.1.src-100_gpt-4o-2024-05-13_examples_0_temp_0_chain_False"
#"level-3_prompt-gold-filtered_wiki.valid_v1.1.src-100_gpt-4o-2024-05-13_examples_5_temp_0_chain_False"
#"level-4_prompt-gold-filtered_wiki.valid_v1.1.src-100_gpt-4o-2024-05-13_examples_0_temp_0_chain_False"
#"level-4_prompt-gold-filtered_wiki.valid_v1.1.src-100_gpt-4o-2024-05-13_examples_5_temp_0_chain_False"
#"level-4_prompt_feedbackloop-gold-filtered_wiki.valid_v1.1.src-100_gpt-4o-2024-05-13_examples_5_temp_0_chain_True"
#)
#
#base="experiments/x_ablation_study"
#sufix="maxdepdepth_-1_maxdeplength_-1_diffwordscount_-1_avgwordcount_-1_length_-1_leven_-1_grade_-1/ratio_stats.csv"
#
#for prompt_level in "${file_paths[@]}"; do
#
#  python llm_based_control_rewrite/scores/calculate_success_rate.py \
#          --predicted_ratio_files  "${base}/WordCount/${prompt_level}/${sufix},${base}/MaxDepDepth/${prompt_level}/${sufix},${base}/MaxDepLength/${prompt_level}/${sufix},${base}/DiffWords/${prompt_level}/${sufix}" \
#          --obtained_ratio_file "data_filtered/en/wikilarge_train_val_test/val/v1.1_wo_line_46/grade_ratio_stats_filtered_wiki_val_v1.1_data.csv" \
#          --feature_names "WordCount,MaxDepDepth,MaxDepLength,DiffWords" \
#          --feature_range "1,1,1,1" \
#          --default_input_src "data_filtered/en/wikilarge_train_val_test/val/v1.1_wo_line_46/filtered_wiki.valid_v1.1.src" \
#          --tested_input_src  "data_filtered/en/wikilarge_train_val_test/val/v1.1_wo_line_46/filtered_wiki.valid_v1.1.src" \
#          --default_ref_tgt "data_filtered/en/wikilarge_train_val_test/val/v1.1_wo_line_46/filtered_wiki.valid_v1.1.tgt" \
#          --tested_ref_tgt  "data_filtered/en/wikilarge_train_val_test/val/v1.1_wo_line_46/filtered_wiki.valid_v1.1.tgt" \
#          --output_generation_path "experiments/x_ablation_study/run-1" \
#          >   experiments/x_ablation_study/run-1/logs_val_success_rate_val_v1.1_wo_line_46_prediction
#
#done