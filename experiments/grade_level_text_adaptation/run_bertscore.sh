#!/bin/bash

#
#base="experiments/train_v3_and_val_v1.1_wo_line_46/T5_ft/5f_access_tokens/grade-level_using_0_catboost_swetas_fkgl_train_2_all_9input_7output"
#
#sub_folders=("catboost-filtered_wiki.valid_v1.1.src-200_exp_1716772596450919_access_tokens")
#sufix="maxdepdepth_-1_maxdeplength_-1_diffwordscount_-1_avgwordcount_-1_length_-1_leven_-1_grade_"
#
#for sub_folder in "${sub_folders[@]}"; do
#  for grade in {1..12}; do
#      echo $grade
#      python llm_based_control_rewrite/scores/easse.py \
#      --orig_sents_path "${base}/${sub_folder}/${sufix}${grade}/input.txt" \
#      --refs_sents_paths "${base}/${sub_folder}/${sufix}${grade}/gold_ref.txt" \
#      --sys_sents_path "${base}/${sub_folder}/${sufix}${grade}/output.txt" \
#      --save_path "${base}/${sub_folder}"
#    done
#done
#
#
#
#base="experiments/train_v3_and_val_v1.1_wo_line_46/grade_level_eval_with_cot_feedback_prompt/0_catboost_swetas_fkgl_train_2_all_9input_7output/f4_maxdepdepth_maxdeplength_diffwords_wc"
#
#sub_folders=("cot_catboost_swetas-filtered_wiki.valid_v1.1.src-200_gpt-4o-2024-05-13_examples_5_temp_0_chain_True" "cot_feedback_with_catboost_swetas-filtered_wiki.valid_v1.1.src-200_gpt-4o-2024-05-13_examples_0_temp_0_chain_False" "cot_feedback_with_catboost_swetas-filtered_wiki.valid_v1.1.src-200_gpt-4o-2024-05-13_examples_5_temp_0_chain_True" "gpt-4o-mini/cot_feedback_with_catboost_swetas-filtered_wiki.valid_v1.1.src-200_gpt-4o-mini_examples_5_temp_0_chain_True" "llama_3_70b_instruct_sglang/cot_feedback_with_catboost_swetas-filtered_wiki.valid_v1.1.src-200_llama_3_70b_instruct_sglang_examples_5_temp_0_chain_True_seed_473829")
#sufix="maxdepdepth_-1_maxdeplength_-1_diffwordscount_-1_avgwordcount_-1_length_-1_leven_-1_grade_"
#
#for sub_folder in "${sub_folders[@]}"; do
#  for grade in {1..12}; do
#      echo $grade
#      python llm_based_control_rewrite/scores/easse.py \
#      --orig_sents_path "${base}/${sub_folder}/${sufix}${grade}/input.txt" \
#      --refs_sents_paths "${base}/${sub_folder}/${sufix}${grade}/gold_ref.txt" \
#      --sys_sents_path "${base}/${sub_folder}/${sufix}${grade}/output.txt" \
#      --save_path "${base}/${sub_folder}"
#    done
#done


#base="experiments/train_v3_and_val_v1.1_wo_line_46/grade_level_eval_with_cot_feedback_prompt/0_catboost_swetas_fkgl_train_2_all_9input_7output/f4_maxdepdepth_maxdeplength_diffwords_wc"
#
#sub_folders=("llama_3_70b_instruct_sglang/seeds/cot_feedback_with_catboost_swetas-filtered_wiki.valid_v1.1.src-200_llama_3_70b_instruct_sglang_examples_5_temp_0_chain_True_seed_184623" "llama_3_70b_instruct_sglang/seeds/cot_feedback_with_catboost_swetas-filtered_wiki.valid_v1.1.src-200_llama_3_70b_instruct_sglang_examples_5_temp_0_chain_True_seed_368914" "llama_3_70b_instruct_sglang/seeds/cot_feedback_with_catboost_swetas-filtered_wiki.valid_v1.1.src-200_llama_3_70b_instruct_sglang_examples_5_temp_0_chain_True_seed_756301" "llama_3_70b_instruct_sglang/seeds/cot_feedback_with_catboost_swetas-filtered_wiki.valid_v1.1.src-200_llama_3_70b_instruct_sglang_examples_5_temp_0_chain_True_seed_921405")
#
#sufix="maxdepdepth_-1_maxdeplength_-1_diffwordscount_-1_avgwordcount_-1_length_-1_leven_-1_grade_"
#
#for sub_folder in "${sub_folders[@]}"; do
#  for grade in {1..12}; do
#      echo $grade
#      python llm_based_control_rewrite/scores/easse.py \
#      --orig_sents_path "${base}/${sub_folder}/${sufix}${grade}/input.txt" \
#      --refs_sents_paths "${base}/${sub_folder}/${sufix}${grade}/gold_ref.txt" \
#      --sys_sents_path "${base}/${sub_folder}/${sufix}${grade}/output.txt" \
#      --save_path "${base}/${sub_folder}"
#    done
#done


#base="experiments/train_v3_and_val_v1.1_wo_line_46/grade_level/0_no_features/prompt_1_just_tgt_grade"
#
#sub_folders=("filtered_wiki.valid.src-200_gpt-4o-2024-05-13_examples_0_temp_0_chain_False"
#"filtered_wiki.valid.src-200_gpt-4o-2024-05-13_examples_5_temp_0_chain_False")
#
#sufix="maxdepdepth_-1_maxdeplength_-1_diffwordscount_-1_avgwordcount_-1_length_-1_leven_-1_grade_"
#
#for sub_folder in "${sub_folders[@]}"; do
#  for grade in {1..12}; do
#      echo $grade
#      python llm_based_control_rewrite/scores/easse.py \
#      --orig_sents_path "${base}/${sub_folder}/${sufix}${grade}/input.txt" \
#      --refs_sents_paths "${base}/${sub_folder}/${sufix}${grade}/gold_ref.txt" \
#      --sys_sents_path "${base}/${sub_folder}/${sufix}${grade}/output.txt" \
#      --save_path "${base}/${sub_folder}"
#    done
#done


#base="experiments/train_v3_and_val_v1.1_wo_line_46/grade_level_eval_with_cot_feedback_prompt/0_no_features/prompt_1_just_tgt_grade/llama_3_70b_instruct_sglang/seeds"
#
#sub_folders=("grade_only_prompt-filtered_wiki.valid_v1.1.src-200_llama_3_70b_instruct_sglang_examples_0_temp_0_chain_False_seed" "grade_only_prompt-filtered_wiki.valid_v1.1.src-200_llama_3_70b_instruct_sglang_examples_5_temp_0_chain_False_seed")
#sufix="maxdepdepth_-1_maxdeplength_-1_diffwordscount_-1_avgwordcount_-1_length_-1_leven_-1_grade_"
#
#seeds=(473829 921405 184623 756301 368914)
#
#for sub_folder in "${sub_folders[@]}"; do
#  for seed in "${seeds[@]}"; do
#    for grade in {1..12}; do
#        echo $grade
#        python llm_based_control_rewrite/scores/easse.py \
#        --orig_sents_path "${base}/${sub_folder}_${seed}/${sufix}${grade}/input.txt" \
#        --refs_sents_paths "${base}/${sub_folder}_${seed}/${sufix}${grade}/gold_ref.txt" \
#        --sys_sents_path "${base}/${sub_folder}_${seed}/${sufix}${grade}/output.txt" \
#        --save_path "${base}/${sub_folder}_${seed}"
#      done
#    done
#done



base="experiments/train_v3_and_val_v1.1_wo_line_46/grade_level_eval_with_cot_feedback_prompt/0_catboost_swetas_fkgl_train_2_all_9input_7output/f4_maxdepdepth_maxdeplength_diffwords_wc/llama_3_70b_instruct_sglang"

sub_folders=("seeds_cot_zs/cot_feedback_with_catboost_swetas-filtered_wiki.valid_v1.1.src-200_llama_3_70b_instruct_sglang_examples_0_temp_0_chain_False_seed"  "seeds_cot_fs/cot_feedback_with_catboost_swetas-filtered_wiki.valid_v1.1.src-200_llama_3_70b_instruct_sglang_examples_5_temp_0_chain_False_seed")

sufix="maxdepdepth_-1_maxdeplength_-1_diffwordscount_-1_avgwordcount_-1_length_-1_leven_-1_grade_"

seeds=(921405 184623 756301 368914 473829)

for sub_folder in "${sub_folders[@]}"; do
  for seed in "${seeds[@]}"; do
    for grade in {1..12}; do
        echo $grade
        python llm_based_control_rewrite/scores/easse.py \
        --orig_sents_path "${base}/${sub_folder}_${seed}/${sufix}${grade}/input.txt" \
        --refs_sents_paths "${base}/${sub_folder}_${seed}/${sufix}${grade}/gold_ref.txt" \
        --sys_sents_path "${base}/${sub_folder}_${seed}/${sufix}${grade}/output.txt" \
        --save_path "${base}/${sub_folder}_${seed}"
      done
    done
done


#CUDA_VISIBLE_DEVICES=3,4,5 bash