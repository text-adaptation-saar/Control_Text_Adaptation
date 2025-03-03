#!/bin/bash

base_directory="experiments/train_v3_and_val_v1.1_wo_line_46/T5_ft/5f_access_tokens/grade-level_using_0_catboost_swetas_fkgl_train_2_all_9input_7output"
base_config="experiments/train_v3_and_val_v1.1_wo_line_46/T5_ft/4f_a8_tokens/base_config_file.yaml"

test_data_files=("data_filtered/en/wikilarge_train_val_test/val/v1.1_wo_line_46/filtered_wiki.valid_v1.1.src")
test_gold_ref_files=("data_filtered/en/wikilarge_train_val_test/val/v1.1_wo_line_46/filtered_wiki.valid_v1.1.tgt")

number_of_lines_to_test=200
#MaxDepDepth,MaxDepLength,DiffWords,WordCount
feature_names="WordCount,Length,Leven,FreqRank,MaxDepDepth"
abs_src_feature_list="abs_src_WordCount,abs_src_Length,abs_src_Leven,abs_src_FreqRank,abs_src_MaxDepDepth"
ratio_feature_list="WordCount_ratio,Length_ratio,Leven_ratio,FreqRank_ratio,MaxDepDepth_ratio"
#feature_range="0,0,0,0"
feature_range="1,1,1,1,1"
models=("ControlTS_T5/experiments/T5/exp_1716772596450919_access_tokens") # "gpt-4-0125-preview" "gpt-4o-2024-05-13" "gpt-3.5-turbo" "gpt-4" "gpt-4-0125-preview"

#W_0.71 C_0.63 L_0.72 WR_1.00 DTD_1.12
# Loop over each combination and create a new config file
for (( i=0; i<${#test_data_files[@]}; i++ )); do
    test_data=${test_data_files[$i]}
    test_data_name=$(basename "$test_data")

    for model in "${models[@]}"; do
            for grade in {1..12}; do
                echo "Grade: $grade"
                predicted_ratio_feature_1_file_paths=("experiments/train_v3_and_val_v1.1_wo_line_46/0_catboost_swetas_fkgl/fkgl_train_2_all_9input_7output/grade_${grade}-val_v1.1_wo_line_46_prediction/ratio_stats_with_LR_predicted_ratio.csv")

                predicted_ratio_feature_1_file_path=${predicted_ratio_feature_1_file_paths[$i]}
                predicted_ratio_feature_2_file_path=${predicted_ratio_feature_1_file_paths[$i]}
                predicted_ratio_feature_3_file_path=${predicted_ratio_feature_1_file_paths[$i]}
                predicted_ratio_feature_4_file_path=${predicted_ratio_feature_1_file_paths[$i]}
                predicted_ratio_feature_5_file_path=${predicted_ratio_feature_1_file_paths[$i]}

                # Extract a name identifier for the test data
                test_data_name=$(basename "$test_data")

                # Create a new config file name
                new_config="config_grade-${grade}_${test_data_name}_exp_1716772596450919_access_tokens.yaml"

                # Directory to save the new config file
                config_dir="${base_directory}/catboost-${test_data_name}-${number_of_lines_to_test}_exp_1716772596450919_access_tokens" # Adjust this path as needed
                mkdir -p "$config_dir"

                # Copy the base config to the new location
                cp "$base_config" "${config_dir}/${new_config}"

                # Modify the copied config file
                sed -i "s|path_to_output_generation: .*|path_to_output_generation: \"${config_dir}\"|" "${config_dir}/${new_config}"
                sed -i "s|path_to_input_test_data: .*|path_to_input_test_data: \"${test_data}\"|" "${config_dir}/${new_config}"
                sed -i "s|path_to_test_gold_ref_data: .*|path_to_test_gold_ref_data: \"${test_gold_ref_files[$i]}\"|" "${config_dir}/${new_config}"
                sed -i "s|predicted_ratio_file_path: .*|predicted_ratio_file_path: \"${predicted_ratio_feature_1_file_path},${predicted_ratio_feature_2_file_path},${predicted_ratio_feature_3_file_path},${predicted_ratio_feature_4_file_path},${predicted_ratio_feature_5_file_path}\"|" "${config_dir}/${new_config}"
                sed -i "s|model: .*|model: \"${model}\"|" "${config_dir}/${new_config}"
                sed -i "s/feature_names: .*/feature_names: \"${feature_names}\"/" "${config_dir}/${new_config}"
                sed -i "s/number_of_lines_to_test: .*/number_of_lines_to_test: ${number_of_lines_to_test}/" "${config_dir}/${new_config}"
                sed -i "s/feature_range: .*/feature_range: \"${feature_range}\"/" "${config_dir}/${new_config}"

#                echo $config_dir
#                CUDA_VISIBLE_DEVICES=5 python llm_based_control_rewrite/t5_ft_scripts/generate_T5_ft.py --config ${config_dir}/${new_config} \
#                --requested_dependency_depth -1 \
#                --requested_dependency_length -1 \
#                --requested_difficult_words -1 \
#                --requested_length -1 \
#                --requested_levenshtein -1 \
#                --requested_word_count -1 \
#                --requested_grade_level ${grade} \
#                --predicted_ratio_file_given \
#                --requested_absolute_value False \
#                 > ${config_dir}/logs_grade_${grade}-${test_data_name}_exp_1716772596450919_access_tokens

                 echo $config_dir
                 CUDA_VISIBLE_DEVICES=5  python only_calculate_scores.py --config ${config_dir}/${new_config} \
                --requested_dependency_depth -1 \
                --requested_dependency_length -1 \
                --requested_difficult_words -1 \
                --requested_length -1 \
                --requested_levenshtein -1 \
                --requested_word_count -1 \
                --requested_grade_level ${grade} \
                --predicted_ratio_file_given \
                > ${config_dir}/logs_grade-${grade}-score_calculations
            done
      done
done
echo "Configuration files created and experiments completed!"
