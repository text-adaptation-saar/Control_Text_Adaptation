# [Controllable Text Adaptation Using In-context Learning with Linguistic Features](https://drive.google.com/file/d/1wpChlJPVdCJEG_O5VMTWh0C0tSbbfTQy/view?usp=sharing)  

This repository accompanies the paper **"Controllable Text Adaptation Using In-context Learning with Linguistic Features"** by **Sarubi Thillainathan** and **Alexander Koller**, Saarland University.  

The codebase is built upon the [rewrite_text](https://github.com/coli-saar/rewrite_text.git) repository and has been extended to enable **controllable text rewriting** based on various linguistic features. It supports text adaptation using **OpenAI APIs** as well as **other open models** that adhere to the OpenAI API format.  

## Abstract  

The diversity in readers' cognitive abilities, including working memory capacity and prior knowledge, necessitates texts that align with individual comprehension levels. We address the challenge of rewriting text to match readers' unique needs, approximating readers to specific grade levels. Unlike prior approaches that rely on fine-tuned models and large training datasets, our method leverages in-context learning (ICL), making it effective in data-sparse scenarios. By precisely controlling linguistic features such as syntactic depth, our approach delivers tailored rewrites aligned with specific grade levels. We demonstrate state-of-the-art performance in generating grade-specific adaptations, highlighting the potential of ICL-based methods to enhance text accessibility and inclusivity.

## Setup  

1. Clone the repository and run the `run_setup.sh` script.  
2. Set up the codebase for the **[ControlTS_T5](https://github.com/text-adaptation-saar/ControlTS.git)** repository, where we train our **CatBoost Regressor** model.  
3. Run experiments using **OpenAI models** or **LLaMA** as a server with OpenAI-compatible APIs.
   sample experiment file given in: `experiments/grade_level_text_adaptation/grade_level_eval_with_cot_feedback_prompt/0_catboost_swetas_fkgl_train_2_all_9input_7output/f4_maxdepdepth_maxdeplength_diffwords_wc/config_file_populate_catboost_FS.sh`

## Contact
{sarubi, koller}@coli.uni-saarland.de

## Citation  

If you use this work, please cite:  

```bibtex
@inproceedings{thillainathan2025finegrained,
      title={Controllable Text Adaptation Using In-context Learning with Linguistic Features},
      author={Sarubi Thillainathan and Alexander Koller},
      booktitle = {AAAI Workshop on AI for Education -- Tools, Opportunities, and Risks in the Generative AI Era},
      year={2025},
      keywords = {workshop},
      url={https://coli-saar.github.io/ctgthroughiclwithfeedbacks},
}
