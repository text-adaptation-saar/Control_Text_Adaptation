#!/bin/bash

base_directory="experiments/x_ablation_study/baseline_T5ft/MaxDepLength"
mkdir -p ${base_directory}
mkdir -p ${base_directory}/maxdepdepth_-1_maxdeplength_-1_diffwordscount_-1_avgwordcount_-1_length_-1_leven_-1_grade_-1
base_config="experiments/train_v3_and_val_v1.1_wo_line_46/T5_ft/4f_a8_tokens/base_config_file.yaml"


head -n 100 ControlTS_T5/resources/datasets_with_dtl/filtered_wiki/filtered_wiki.valid_v1.1.src > "${base_directory}/maxdepdepth_-1_maxdeplength_-1_diffwordscount_-1_avgwordcount_-1_length_-1_leven_-1_grade_-1/input.processed.txt"

head -n 100 ControlTS_T5/resources/datasets_with_dtl/filtered_wiki/filtered_wiki.valid_v1.1.tgt > "${base_directory}/maxdepdepth_-1_maxdeplength_-1_diffwordscount_-1_avgwordcount_-1_length_-1_leven_-1_grade_-1/gold_ref.txt"

head -n 100 "data_filtered/en/wikilarge_train_val_test/val/v1.1_wo_line_46/filtered_wiki.valid_v1.1.src"  > "${base_directory}/maxdepdepth_-1_maxdeplength_-1_diffwordscount_-1_avgwordcount_-1_length_-1_leven_-1_grade_-1/input.txt"

#(controlTS_swetas) [sarubi@tony-1 ControlTS_T5]$ head -n 20   experiments/T5/logs_T5_ft_with_only_dtl
#'/nethome/sarubi/A8/tony_3/LLM_based_control_rewrite/ControlTS_T5/experiments/T5/exp_1719330922034451

test_data_files=("data_filtered/en/wikilarge_train_val_test/val/v1.1_wo_line_46/filtered_wiki.valid_v1.1.src")
test_gold_ref_files=("data_filtered/en/wikilarge_train_val_test/val/v1.1_wo_line_46/filtered_wiki.valid_v1.1.tgt")
test_gold_ref_ratio_files=("data_filtered/en/wikilarge_train_val_test/val/v1.1_wo_line_46/grade_ratio_stats_filtered_wiki_val_v1.1_data.csv")

number_of_lines_to_test=100
#MaxDepDepth,MaxDepLength,DiffWords,WordCount
feature_names="MaxDepLength"
abs_src_feature_list="abs_src_MaxDepLength"
ratio_feature_list="MaxDepLength_ratio"
#feature_range="0,0,0,0"
feature_range="1"
models=("ControlTS_T5/experiments/T5/exp_1719330922034451")

#W_0.71 C_0.63 L_0.72 WR_1.00 DTD_1.12
# Loop over each combination and create a new config file
for (( i=0; i<${#test_data_files[@]}; i++ )); do
    test_data=${test_data_files[$i]}
    test_data_name=$(basename "$test_data")

    for model in "${models[@]}"; do
                # Extract a name identifier for the test data
                test_data_name=$(basename "$test_data")

                # Create a new config file name
                new_config="config_${test_data_name}_exp_1719330922034451.yaml"

                # Directory to save the new config file
                config_dir="${base_directory}" # Adjust this path as needed
                mkdir -p "$config_dir"

                # Copy the base config to the new location
                cp "$base_config" "${config_dir}/${new_config}"

                # Modify the copied config file
                sed -i "s|path_to_output_generation: .*|path_to_output_generation: \"${config_dir}\"|" "${config_dir}/${new_config}"
                sed -i "s|path_to_input_test_data: .*|path_to_input_test_data: \"${test_data}\"|" "${config_dir}/${new_config}"
                sed -i "s|path_to_test_gold_ref_data: .*|path_to_test_gold_ref_data: \"${test_gold_ref_files[$i]}\"|" "${config_dir}/${new_config}"
                sed -i "s|model: .*|model: \"${model}\"|" "${config_dir}/${new_config}"
                sed -i "s/feature_names: .*/feature_names: \"${feature_names}\"/" "${config_dir}/${new_config}"
                sed -i "s/number_of_lines_to_test: .*/number_of_lines_to_test: ${number_of_lines_to_test}/" "${config_dir}/${new_config}"
                sed -i "s/feature_range: .*/feature_range: \"${feature_range}\"/" "${config_dir}/${new_config}"

                # Append the new entry for gold_ratio_file_given at the end of the config file
                echo "gold_ratio_file_given: \"${test_gold_ref_ratio_files[$i]}\"" >> "${config_dir}/${new_config}"

                echo $config_dir

                CUDA_VISIBLE_DEVICES=5  python only_calculate_scores.py --config ${config_dir}/${new_config} \
                --requested_dependency_depth -1 \
                --requested_dependency_length -1 \
                --requested_difficult_words -1 \
                --requested_length -1 \
                --requested_levenshtein -1 \
                --requested_word_count -1 \
                --requested_grade_level -1 \
                > ${config_dir}/logs_score_calculations
      done
done
echo "Configuration files created and experiments completed!"
