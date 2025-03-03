import pandas as pd
from textstat import textstat
from easse.fkgl import corpus_fkgl

def analysis(ratio, src, tgt, save, update_and_save_2=None):
    df = pd.read_csv(ratio)
    # df['current_train_v3_line'] = range(1, len(df) + 1)
    df['current_line'] = range(1, len(df) + 1)

    # Load the sentences from the src and tgt files
    with open(src, 'r') as file:
        src_sentences = file.readlines()

    with open(tgt, 'r') as file:
        tgt_sentences = file.readlines()

    # Lists to store scores and ratios
    fkgl_score_src = []
    fkgl_score_tgt = []
    fkgl_score_ratio = []

    # Lists to store scores and ratios
    ari_score_src = []
    ari_score_tgt = []
    ari_score_ratio = []

    # Step 3: Calculate the scores for each selected sentence pair
    # for idx in df['current_train_v3_line']:
    for idx in df['current_line']:
        src_sentence = src_sentences[idx-1].strip()
        tgt_sentence = tgt_sentences[idx - 1].strip()

        src_fkgl_grade = min(13, max(0, round(textstat.flesch_kincaid_grade(src_sentence))))
        tgt_fkgl_grade = min(13, max(0, round(textstat.flesch_kincaid_grade(tgt_sentence))))
        # src_grade = round(corpus_fkgl(src_sentence,  tokenizer="13a")) #courpus_fkgl return 0 for single sentence.
        # tgt_grade = round(corpus_fkgl(tgt_sentence,  tokenizer="13a"))
        fkgl_ratio = tgt_fkgl_grade / (0.5 if src_fkgl_grade == 0 else src_fkgl_grade)
        fkgl_score_src.append(src_fkgl_grade)
        fkgl_score_tgt.append(tgt_fkgl_grade)
        fkgl_score_ratio.append(round(fkgl_ratio,2) )

        src_ari_grade=min(14, max(0, round(textstat.automated_readability_index(src_sentence))))
        tgt_ari_grade=min(14, max(0, round(textstat.automated_readability_index(tgt_sentence))))
        ari_ratio= tgt_ari_grade / (0.5 if src_ari_grade == 0 else src_ari_grade)
        ari_score_src.append(src_ari_grade)
        ari_score_tgt.append(tgt_ari_grade)
        ari_score_ratio.append(round(ari_ratio,2) )

        # if src_fkgl_grade != src_ari_grade:
        #     print(f"idx: {idx}\tsrc_fkgl_grade: {src_fkgl_grade}, \t src_ari_grade:{src_ari_grade}")
        # if tgt_fkgl_grade != tgt_ari_grade:
        #     print(f"idx: {idx}\ttgt_fkgl_grade: {tgt_fkgl_grade}, \t tgt_ari_grade:{tgt_ari_grade}")

    df['abs_src_FKGL_Grade'] = fkgl_score_src
    df['abs_tgt_FKGL_Grade'] = fkgl_score_tgt
    df['FKGL_Grade_ratio'] = fkgl_score_ratio

    df['abs_src_ARI_Grade'] = ari_score_src
    df['abs_tgt_ARI_Grade'] = ari_score_tgt
    df['ARI_Grade_ratio'] = ari_score_ratio
    # replace leven ratio (i.e 1) to leven score.
    df['Leven_ratio'] = df['abs_tgt_Leven']

    # Rearrange columns to make 'current_line' the first column
    cols = ['current_line'] + [col for col in df.columns if col != 'current_line']
    df = df[cols]

    df.to_csv(save, index=False)

    # df_2 = pd.read_csv(update_and_save_2)
    # df_2['abs_src_FKGL_Grade'] = df['abs_src_FKGL_Grade']
    # df_2['abs_tgt_FKGL_Grade'] = df['abs_tgt_FKGL_Grade']
    # df_2['FKGL_Grade_ratio'] = df['FKGL_Grade_ratio']
    #
    # df_2['abs_src_ARI_Grade'] = df['abs_src_ARI_Grade']
    # df_2['abs_tgt_ARI_Grade'] = df['abs_tgt_ARI_Grade']
    # df_2['ARI_Grade_ratio'] = df['ARI_Grade_ratio']
    #
    # df_2['Leven_ratio'] = df['Leven_ratio']
    #
    # df_2.to_csv(update_and_save_2, index=False)

if __name__=="__main__":
    # analysis(ratio="data_filtered/en/wikilarge_train_val_test/val/v1.1_wo_line_46/ratio_stats_filtered_wiki_val_v1.1_data.csv",
    #          src="data_filtered/en/wikilarge_train_val_test/val/v1.1_wo_line_46/filtered_wiki.valid_v1.1.src",
    #          tgt="data_filtered/en/wikilarge_train_val_test/val/v1.1_wo_line_46/filtered_wiki.valid_v1.1.tgt",
    #          save="data_filtered/en/wikilarge_train_val_test/val/v1.1_wo_line_46/grade_ratio_stats_filtered_wiki_val_v1.1_data.csv",
    #          update_and_save_2="ControlTS_T5/resources/datasets/filtered_wiki/CP_features_filtered_wiki_val_v1.1_data.csv")
    #
    # analysis(
    #     ratio="data_filtered/en/wikilarge_train_val_test/train_v3_500_complex_eliminated-duplicated_removed_from_v1v2_val_and_test/ratio_stats_filtered_wiki_train_v3_data.csv",
    #     src="data_filtered/en/wikilarge_train_val_test/train_v3_500_complex_eliminated-duplicated_removed_from_v1v2_val_and_test/filtered_wiki.train_v3.src",
    #     tgt="data_filtered/en/wikilarge_train_val_test/train_v3_500_complex_eliminated-duplicated_removed_from_v1v2_val_and_test/filtered_wiki.train_v3.tgt",
    #     save="data_filtered/en/wikilarge_train_val_test/train_v3_500_complex_eliminated-duplicated_removed_from_v1v2_val_and_test/grade_ratio_stats_filtered_wiki_train_v3_data.csv",
    #     update_and_save_2="ControlTS_T5/resources/datasets/filtered_wiki/CP_features_filtered_wiki_train_v3_data.csv")
    #
    # analysis(ratio="data_filtered/en/wikilarge_train_val_test/test/v1.1-duplicated_removed_from_val_and_val_v2_only_complex_extracted_from_train/ratio_stats_filtered_wiki_test_v1.1_duplicated_removed.csv",
    #          src="data_filtered/en/wikilarge_train_val_test/test/v1.1-duplicated_removed_from_val_and_val_v2_only_complex_extracted_from_train/filtered_wiki.test_v1.1_duplicated_removed.src",
    #          tgt="data_filtered/en/wikilarge_train_val_test/test/v1.1-duplicated_removed_from_val_and_val_v2_only_complex_extracted_from_train/filtered_wiki.test_v1.1_duplicated_removed.tgt",
    #          save="data_filtered/en/wikilarge_train_val_test/test/v1.1-duplicated_removed_from_val_and_val_v2_only_complex_extracted_from_train/grade_ratio_stats_filtered_wiki_test_v1.1_duplicated_removed.csv",
    #          update_and_save_2="ControlTS_T5/resources/datasets/filtered_wiki/CP_features_filtered_wiki_test_v1.1_duplicated_removed.csv")

    analysis(ratio="data/en/test_sets_from_EASSE/turkcorpus/feature_cal/tgt_0/ratio_stats.csv",
             src="data/en/test_sets_from_EASSE/turkcorpus/test.truecase.detok.orig",
             tgt="data/en/test_sets_from_EASSE/turkcorpus/test.truecase.detok.simp.0",
             save="data/en/test_sets_from_EASSE/turkcorpus/feature_cal/tgt_0/grade_ratio_stats_turk_test_simp0.csv")
