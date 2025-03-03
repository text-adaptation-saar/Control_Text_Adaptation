#!/bin/bash

base_directory="experiments/x_ablation_study"
base_config="${base_directory}/base_config_file.yaml"
#base_config="experiments/data_filtered_regression_model/regression_3/f4_maxdepdepth_maxdeplength_diffwords_wc/base_config_file.yaml"

test_data_files=("data_filtered/en/wikilarge_train_val_test/val/v1.1_wo_line_46/filtered_wiki.valid_v1.1.src")
test_gold_ref_files=("data_filtered/en/wikilarge_train_val_test/val/v1.1_wo_line_46/filtered_wiki.valid_v1.1.tgt")
test_gold_ref_ratio_files=("data_filtered/en/wikilarge_train_val_test/val/v1.1_wo_line_46/grade_ratio_stats_filtered_wiki_val_v1.1_data.csv")

openai_key_name="OPEN_API_KEY_CSE"
number_of_lines_to_test=100
#MaxDepDepth,MaxDepLength,DiffWords,WordCount
#feature_names="MaxDepDepth,MaxDepLength,DiffWords,WordCount"
#abs_src_feature_list="abs_src_MaxDepDepth,abs_src_MaxDepLength,abs_src_DiffWords,abs_src_WordCount"
#ratio_feature_list="MaxDepDepth_ratio,MaxDepLength_ratio,DiffWords_ratio,WordCount_ratio"
#feature_range="0,0,0,0"
feature_range="1"
models=("gpt-4o-2024-05-13") # "gpt-4-0125-preview" "gpt-4o-2024-05-13" "gpt-3.5-turbo" "gpt-4" "gpt-4-0125-preview"
temperature_list=(0)
example_numbers=(5)
chain_of_thought_options=(False)
calibration=False
batch_request=True

#feature_list=( "MaxDepDepth" "MaxDepLength" "WordCount" "DiffWords") #
#feature_list=("WordCount")
feature_list=("Length")

# Loop over each combination and create a new config file
for (( i=0; i<${#test_data_files[@]}; i++ )); do
    test_data=${test_data_files[$i]}
    test_data_name=$(basename "$test_data")

    for model in "${models[@]}"; do
      for temperature in "${temperature_list[@]}"; do
        for num_examples in "${example_numbers[@]}"; do
          for chain in "${chain_of_thought_options[@]}"; do
            for feature in "${feature_list[@]}"; do
                echo "Feature: ${feature}"
                # Extract a name identifier for the test data
                test_data_name=$(basename "$test_data")

                # Create a new config file name
                new_config="config_${test_data_name}_${model}_examples_${num_examples}_temp_${temperature}_chain_${chain}.yaml"

                # Directory to save the new config file
                mkdir -p "${base_directory}/${feature}"
                config_dir="${base_directory}/${feature}/level-3_prompt-gold-${test_data_name}-${number_of_lines_to_test}_${model}_examples_${num_examples}_temp_${temperature}_chain_${chain}" # Adjust this path as needed
                mkdir -p "$config_dir"

                # Copy the base config to the new location
                cp "$base_config" "${config_dir}/${new_config}"

                # Modify the copied config file
                sed -i '' "s|path_to_output_generation: .*|path_to_output_generation: \"${config_dir}\"|" "${config_dir}/${new_config}"
                sed -i '' "s|path_to_input_test_data: .*|path_to_input_test_data: \"${test_data}\"|" "${config_dir}/${new_config}"
                sed -i '' "s|path_to_test_gold_ref_data: .*|path_to_test_gold_ref_data: \"${test_gold_ref_files[$i]}\"|" "${config_dir}/${new_config}"
#                sed -i '' "s|predicted_ratio_file_path: .*|predicted_ratio_file_path: \"${predicted_ratio_feature_1_file_path},${predicted_ratio_feature_2_file_path},${predicted_ratio_feature_3_file_path},${predicted_ratio_feature_4_file_path}\"|" "${config_dir}/${new_config}"
                sed -i '' "s/model: .*/model: \"${model}\"/" "${config_dir}/${new_config}"
                sed -i '' "s/temperature: .*/temperature: ${temperature}/" "${config_dir}/${new_config}"
                sed -i '' "s/number_of_examples_to_include_in_prompt: .*/number_of_examples_to_include_in_prompt: ${num_examples}/" "${config_dir}/${new_config}"
                sed -i '' "s/chain_of_thought: .*/chain_of_thought: ${chain}/" "${config_dir}/${new_config}"
                sed -i '' "s/feature_names: .*/feature_names: \"${feature}\"/" "${config_dir}/${new_config}"
                sed -i '' "s/number_of_lines_to_test: .*/number_of_lines_to_test: ${number_of_lines_to_test}/" "${config_dir}/${new_config}"
                sed -i '' "s/calibration: .*/calibration: ${calibration}/" "${config_dir}/${new_config}"
                sed -i '' "s/openai_key_name: .*/openai_key_name: \"${openai_key_name}\"/" "${config_dir}/${new_config}"
                sed -i '' "s/feature_range: .*/feature_range: \"${feature_range}\"/" "${config_dir}/${new_config}"

                sed -i '' "s|prompt_csv: .*|prompt_csv: \"${base_directory}/Prompts - ${feature}.csv\"|" "${config_dir}/${new_config}"
                sed -i '' "s/system_prompt: .*/system_prompt: 2/" "${config_dir}/${new_config}"
                sed -i '' "s/user_prompt: .*/user_prompt: 3/" "${config_dir}/${new_config}"
                sed -i '' "s/output_prompt: .*/output_prompt: 3/" "${config_dir}/${new_config}"

                # Append the new entry for gold_ratio_file_given at the end of the config file
                echo "gold_ratio_file_given: \"${test_gold_ref_ratio_files[$i]}\"" >> "${config_dir}/${new_config}"
                echo "batch_request: ${batch_request}" >> "${config_dir}/${new_config}"

                echo $config_dir
                python openai_preprocess_finetune_generate.py --config ${config_dir}/${new_config} \
                --requested_dependency_depth -1 \
                --requested_dependency_length -1 \
                --requested_difficult_words -1 \
                --requested_length -1 \
                --requested_levenshtein -1 \
                --requested_word_count -1 \
                --requested_grade_level -1 \
                --requested_absolute_value True \
                --do_eval > ${config_dir}/logs-${test_data_name}_${model}_examples_${num_examples}_temp_${temperature}_chain_${chain}
#
#                python llm_based_control_rewrite/generate/openai_gpt/batch_process.py \
#                --config ${config_dir}/${new_config} \
#                --batch_request_details_path "${config_dir}/grade_-1.0-batch_request_details.json" > ${config_dir}/logs-batch_retrieve
#
#                python only_calculate_scores.py --config ${config_dir}/${new_config} \
#                --requested_dependency_depth -1 \
#                --requested_dependency_length -1 \
#                --requested_difficult_words -1 \
#                --requested_length -1 \
#                --requested_levenshtein -1 \
#                --requested_word_count -1 \
#                --requested_grade_level -1 \
#                > ${config_dir}/logs-score_calculations
                done
              done
          done
      done
  done
done
echo "Configuration files created and experiments completed!"
