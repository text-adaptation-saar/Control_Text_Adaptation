#!/bin/bash

base_directory="experiments/x_ablation_study/baseline_T5ft/MaxDepDepth"

head -n 200 ControlTS_T5/resources/datasets_with_dtd/filtered_wiki/filtered_wiki.valid_v1.1.src > "${base_directory}/maxdepdepth_-1_maxdeplength_-1_diffwordscount_-1_avgwordcount_-1_length_-1_leven_-1_grade_-1/input.processed.txt"

head -n 200 ControlTS_T5/resources/datasets_with_dtd/filtered_wiki/filtered_wiki.valid_v1.1.tgt > "${base_directory}/maxdepdepth_-1_maxdeplength_-1_diffwordscount_-1_avgwordcount_-1_length_-1_leven_-1_grade_-1/gold_ref.txt"

head -n 200 "data_filtered/en/wikilarge_train_val_test/val/v1.1_wo_line_46/filtered_wiki.valid_v1.1.src"  > "${base_directory}/maxdepdepth_-1_maxdeplength_-1_diffwordscount_-1_avgwordcount_-1_length_-1_leven_-1_grade_-1/input.txt"



base_directory="experiments/x_ablation_study/baseline_T5ft/MaxDepLength"

head -n 200 ControlTS_T5/resources/datasets_with_dtd/filtered_wiki/filtered_wiki.valid_v1.1.src > "${base_directory}/maxdepdepth_-1_maxdeplength_-1_diffwordscount_-1_avgwordcount_-1_length_-1_leven_-1_grade_-1/input.processed.txt"

head -n 200 ControlTS_T5/resources/datasets_with_dtd/filtered_wiki/filtered_wiki.valid_v1.1.tgt > "${base_directory}/maxdepdepth_-1_maxdeplength_-1_diffwordscount_-1_avgwordcount_-1_length_-1_leven_-1_grade_-1/gold_ref.txt"

head -n 200 "data_filtered/en/wikilarge_train_val_test/val/v1.1_wo_line_46/filtered_wiki.valid_v1.1.src"  > "${base_directory}/maxdepdepth_-1_maxdeplength_-1_diffwordscount_-1_avgwordcount_-1_length_-1_leven_-1_grade_-1/input.txt"
