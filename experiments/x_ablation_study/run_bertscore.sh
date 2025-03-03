#!/bin/bash


#easse evaluate -m bertscore -t custom -tok 13a --orig_sents_path experiments/x_ablation_study/MaxDepDepth/level-2.1_prompt-gold-filtered_wiki.valid_v1.1.src-100_gpt-4o-2024-05-13_examples_0_temp_0_chain_False/maxdepdepth_-1_maxdeplength_-1_diffwordscount_-1_avgwordcount_-1_length_-1_leven_-1_grade_-1/input.txt \
#--refs_sents_paths experiments/x_ablation_study/MaxDepDepth/level-2.1_prompt-gold-filtered_wiki.valid_v1.1.src-100_gpt-4o-2024-05-13_examples_0_temp_0_chain_False/maxdepdepth_-1_maxdeplength_-1_diffwordscount_-1_avgwordcount_-1_length_-1_leven_-1_grade_-1/gold_ref.txt  \
#--sys_sents_path experiments/x_ablation_study/MaxDepDepth/level-2.1_prompt-gold-filtered_wiki.valid_v1.1.src-100_gpt-4o-2024-05-13_examples_0_temp_0_chain_False/maxdepdepth_-1_maxdeplength_-1_diffwordscount_-1_avgwordcount_-1_length_-1_leven_-1_grade_-1/input.txt

#Some weights of RobertaModel were not initialized from the model checkpoint at roberta-large and are newly initialized: ['roberta.pooler.dense.bias', 'roberta.pooler.dense.weight']
#You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
#{'bertscore_precision': 0.413, 'bertscore_recall': 0.559, 'bertscore_f1': 0.484}


easse evaluate -m bertscore -t custom -tok 13a --orig_sents_path experiments/x_ablation_study/MaxDepDepth/level-2.1_prompt-gold-filtered_wiki.valid_v1.1.src-100_gpt-4o-2024-05-13_examples_0_temp_0_chain_False/maxdepdepth_-1_maxdeplength_-1_diffwordscount_-1_avgwordcount_-1_length_-1_leven_-1_grade_-1/input.txt \
--refs_sents_paths experiments/x_ablation_study/MaxDepDepth/level-2.1_prompt-gold-filtered_wiki.valid_v1.1.src-100_gpt-4o-2024-05-13_examples_0_temp_0_chain_False/maxdepdepth_-1_maxdeplength_-1_diffwordscount_-1_avgwordcount_-1_length_-1_leven_-1_grade_-1/gold_ref.txt  \
--sys_sents_path experiments/x_ablation_study/MaxDepDepth/level-2.1_prompt-gold-filtered_wiki.valid_v1.1.src-100_gpt-4o-2024-05-13_examples_0_temp_0_chain_False/maxdepdepth_-1_maxdeplength_-1_diffwordscount_-1_avgwordcount_-1_length_-1_leven_-1_grade_-1/input.txt


feature_list=( "MaxDepDepth" "MaxDepLength" "WordCount" "DiffWords")
#feature_list=( "MaxDepDepth" )

#file_paths=(
#"baseline_T5ft"
#"free_style-gold-filtered_wiki.valid_v1.1.src-100_gpt-4o-2024-05-13_examples_0_temp_0_chain_False"
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
#sufix="maxdepdepth_-1_maxdeplength_-1_diffwordscount_-1_avgwordcount_-1_length_-1_leven_-1_grade_-1"
#
#for feature in "${feature_list[@]}"; do
#  for prompt_level in "${file_paths[@]}"; do
#    echo $feature $prompt_level
#    python llm_based_control_rewrite/scores/easse.py \
#    --orig_sents_path ${base}/${feature}/${prompt_level}/${sufix}/input.txt \
#    --refs_sents_paths ${base}/${feature}/${prompt_level}/${sufix}/gold_ref.txt \
#    --sys_sents_path ${base}/${feature}/${prompt_level}/${sufix}/output.txt \
#    --save_path ${base}
#
##    easse evaluate -m bertscore -t custom -tok 13a --orig_sents_path ${base}/${feature}/${prompt_level}/${sufix}/input.txt \
##    --refs_sents_paths ${base}/${feature}/${prompt_level}/${sufix}/gold_ref.txt  \
##    --sys_sents_path ${base}/${feature}/${prompt_level}/${sufix}/output.txt
#  done
#done


base="experiments/x_ablation_study/baseline_T5ft"
sufix="maxdepdepth_-1_maxdeplength_-1_diffwordscount_-1_avgwordcount_-1_length_-1_leven_-1_grade_-1"

for feature in "${feature_list[@]}"; do
    echo $feature $prompt_level
    python llm_based_control_rewrite/scores/easse.py \
    --orig_sents_path ${base}/${feature}/${sufix}/input.txt \
    --refs_sents_paths ${base}/${feature}/${sufix}/gold_ref.txt \
    --sys_sents_path ${base}/${feature}/${sufix}/output.txt \
    --save_path ${base}

#    easse evaluate -m bertscore -t custom -tok 13a --orig_sents_path ${base}/${feature}/${prompt_level}/${sufix}/input.txt \
#    --refs_sents_paths ${base}/${feature}/${prompt_level}/${sufix}/gold_ref.txt  \
#    --sys_sents_path ${base}/${feature}/${prompt_level}/${sufix}/output.txt
done



#easse evaluate -m bertscore -t custom -tok 13a --orig_sents_path experiments/x_ablation_study/MaxDepDepth/level-2.1_prompt-gold-filtered_wiki.valid_v1.1.src-100_gpt-4o-2024-05-13_examples_0_temp_0_chain_False/maxdepdepth_-1_maxdeplength_-1_diffwordscount_-1_avgwordcount_-1_length_-1_leven_-1_grade_-1/input.txt \
#--refs_sents_paths experiments/x_ablation_study/MaxDepDepth/level-2.1_prompt-gold-filtered_wiki.valid_v1.1.src-100_gpt-4o-2024-05-13_examples_0_temp_0_chain_False/maxdepdepth_-1_maxdeplength_-1_diffwordscount_-1_avgwordcount_-1_length_-1_leven_-1_grade_-1/gold_ref.txt  \
#--sys_sents_path experiments/x_ablation_study/MaxDepDepth/level-2.1_prompt-gold-filtered_wiki.valid_v1.1.src-100_gpt-4o-2024-05-13_examples_0_temp_0_chain_False/maxdepdepth_-1_maxdeplength_-1_diffwordscount_-1_avgwordcount_-1_length_-1_leven_-1_grade_-1/output.txt

#easse evaluate -m bertscore -t custom -tok 13a --orig_sents_path experiments/x_ablation_study/MaxDepDepth/baseline_T5ft/maxdepdepth_-1_maxdeplength_-1_diffwordscount_-1_avgwordcount_-1_length_-1_leven_-1_grade_-1/input.txt --refs_sents_paths experiments/x_ablation_study/MaxDepDepth/baseline_T5ft/maxdepdepth_-1_maxdeplength_-1_diffwordscount_-1_avgwordcount_-1_length_-1_leven_-1_grade_-1/gold_ref.txt  --sys_sents_path experiments/x_ablation_study/MaxDepDepth/baseline_T5ft/maxdepdepth_-1_maxdeplength_-1_diffwordscount_-1_avgwordcount_-1_length_-1_leven_-1_grade_-1/output.txt



easse evaluate -m bertscore -t custom -tok 13a --orig_sents_path   experiments/x_ablation_study/baseline_T5ft/MaxDepDepth/maxdepdepth_-1_maxdeplength_-1_diffwordscount_-1_avgwordcount_-1_length_-1_leven_-1_grade_-1/input.txt \
--refs_sents_paths   experiments/x_ablation_study/baseline_T5ft/MaxDepDepth/maxdepdepth_-1_maxdeplength_-1_diffwordscount_-1_avgwordcount_-1_length_-1_leven_-1_grade_-1/gold_ref.txt  \
--sys_sents_path experiments/x_ablation_study/baseline_T5ft/MaxDepDepth/maxdepdepth_-1_maxdeplength_-1_diffwordscount_-1_avgwordcount_-1_length_-1_leven_-1_grade_-1/input.txt

Some weights of RobertaModel were not initialized from the model checkpoint at roberta-large and are newly initialized: ['roberta.pooler.dense.bias', 'roberta.pooler.dense.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
{'bertscore_precision': 0.413, 'bertscore_recall': 0.559, 'bertscore_f1': 0.484}

easse evaluate -m bertscore -t custom -tok 13a --orig_sents_path   experiments/x_ablation_study/MaxDepDepth/level-4_prompt_feedbackloop-gold-filtered_wiki.valid_v1.1.src-100_gpt-4o-2024-05-13_examples_5_temp_0_chain_True/maxdepdepth_-1_maxdeplength_-1_diffwordscount_-1_avgwordcount_-1_length_-1_leven_-1_grade_-1/input.txt \
--refs_sents_paths   experiments/x_ablation_study/MaxDepDepth/level-4_prompt_feedbackloop-gold-filtered_wiki.valid_v1.1.src-100_gpt-4o-2024-05-13_examples_5_temp_0_chain_True/maxdepdepth_-1_maxdeplength_-1_diffwordscount_-1_avgwordcount_-1_length_-1_leven_-1_grade_-1/gold_ref.txt  \
--sys_sents_path experiments/x_ablation_study/MaxDepDepth/level-4_prompt_feedbackloop-gold-filtered_wiki.valid_v1.1.src-100_gpt-4o-2024-05-13_examples_5_temp_0_chain_True/maxdepdepth_-1_maxdeplength_-1_diffwordscount_-1_avgwordcount_-1_length_-1_leven_-1_grade_-1/input.txt

Some weights of RobertaModel were not initialized from the model checkpoint at roberta-large and are newly initialized: ['roberta.pooler.dense.bias', 'roberta.pooler.dense.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
{'bertscore_precision': 0.413, 'bertscore_recall': 0.559, 'bertscore_f1': 0.484}