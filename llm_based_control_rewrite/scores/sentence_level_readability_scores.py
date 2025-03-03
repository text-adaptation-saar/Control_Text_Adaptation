"""
THIS IS A STAND-ALONE SCRIPT TO CALCULATE CONTROL TOKEN FEATURE VALUES OF ACTUAL MODEL PREDICTED SENTENCES VS.
 INPUT SENTENCES.
This script takes command line arguments.
"""
import numpy as np
import argparse

from llm_based_control_rewrite.utils.feature_extraction import load_spacy_model
from llm_based_control_rewrite.utils.helpers import load_yaml
import spacy
from textstat import textstat

def write_to_file(final_scores_string, final_score_file_path):
    with open(final_score_file_path, "a") as fp:
        fp.write(final_scores_string + "\n")
def calculate_readability_scores_doc_level(input_sent_file, doc_readability_score_file_path, total_line_number=None):

    # Read the text file
    with open(input_sent_file, 'r', encoding='utf-8') as file:
        text = file.read()

    # Calculate Flesch Reading Ease score
    all_scores = doc_level_calculation(text)
    # Final avg.
    final_string_to_print = ""
    for key, value in all_scores.items():
        final_string_to_print += "doc_level_" + key + "," + str(value) + ","

    final_string_to_print += "Input_file_path," + input_sent_file + "\n"
    write_to_file(final_string_to_print, doc_readability_score_file_path+"/doc_level_scores.csv")
    return final_string_to_print.strip()


def doc_level_calculation(input_text):
    dict_all_scores = {}
    # it is averaged, need to cal abs.
    dict_all_scores["difficult_words"] = textstat.difficult_words(input_text)
    # calculate readability score in doc level.
    dict_all_scores["FRE"] = textstat.flesch_reading_ease(input_text)
    dict_all_scores["FKGL"] = textstat.flesch_kincaid_grade(input_text)
    dict_all_scores["CLI"] = textstat.coleman_liau_index(input_text)
    dict_all_scores["DCR"] = textstat.dale_chall_readability_score(input_text)
    dict_all_scores["ARI"] = textstat.automated_readability_index(input_text)
    dict_all_scores["SI"]  = textstat.smog_index(input_text)
    dict_all_scores["LWF"] = textstat.linsear_write_formula(input_text)
    dict_all_scores["GF"] = textstat.gunning_fog(input_text)
    dict_all_scores["TS"] = textstat.text_standard(input_text)
    dict_all_scores["TS_score"] = textstat.text_standard(input_text,float_output=True)
    return dict_all_scores

def calculate_readability_scores_sentence_level(input_sent_file, readability_scores_file_path, avg_score_file_path, total_line_number=None):
    all_scores = {}
    count=0
    with open(input_sent_file, "r") as fp, open(readability_scores_file_path+"/readability_scores.csv", 'w') as save_fp:
        # save_fp.write("FRE, CLI, DCR, ARI\n")
        for line in fp.readlines():
            count += 1
            if total_line_number is not None:
                if count > total_line_number:
                    break
            scores_dict = sentence_level_calculation(line)
            string_to_print = ""
            for key, value in scores_dict.items():
                string_to_print += key + "," + str(value) + ","
                # Check if the key exists in all_scores, and if not, initialize it
                if key not in all_scores:
                    all_scores[key] = []
                all_scores[key].append(value)
            string_to_print += "\n"
            save_fp.write(string_to_print)

    # Final avg.
    final_string_to_print = ""
    for key, value in all_scores.items():
        final_string_to_print += "avg_" + key + "," + str(np.average(value)) + "," + "std_" + key + ","+ str(np.std(value)) + ","

    final_string_to_print += "Input_file_path," + input_sent_file + "\n"
    write_to_file(final_string_to_print, avg_score_file_path+"/avg_scores.csv")

    return final_string_to_print.strip()

# Sentence level score cal. if the target sentence have more than one sentences, we are going to get the average score.
# Strickly measure it for each sentence in the given (target/source line) text.
def sentence_level_calculation(input_text):
    # Initialize spaCy
    nlp = load_spacy_model("en")
    doc = nlp(input_text)
    # can use this maybe?
    # sentences = nltk.sent_tokenize(text)
    # Iterate over each sentence in the document
    dict_all_scores = {}
    # calculate readability score in doc level.
    dict_all_scores["difficult_words"] = textstat.difficult_words(input_text)
    # dict_all_scores["FRE"] = round(np.average([textstat.flesch_reading_ease(sentence.text) for sentence in doc.sents]), 2)
    # dict_all_scores["FKGL"] = round(np.average([textstat.flesch_kincaid_grade(sentence.text) for sentence in doc.sents]), 2)
    # dict_all_scores["CLI"] = round(np.average([textstat.coleman_liau_index(sentence.text)  for sentence in doc.sents]), 2)
    # dict_all_scores["DCR"] = round(np.average([textstat.dale_chall_readability_score(sentence.text)  for sentence in doc.sents]), 2)
    # dict_all_scores["ARI"] = round(np.average([textstat.automated_readability_index(sentence.text)  for sentence in doc.sents]), 2)
    # dict_all_scores["SI"]  = round(np.average([textstat.smog_index(sentence.text)  for sentence in doc.sents]), 2)
    # dict_all_scores["LWF"] = round(np.average([textstat.linsear_write_formula(sentence.text)  for sentence in doc.sents]), 2)
    # dict_all_scores["GF"] = round(np.average([textstat.gunning_fog(sentence.text)  for sentence in doc.sents]), 2)
    # dict_all_scores["TS"] = textstat.text_standard(input_text)

    dict_all_scores["FRE"] = textstat.flesch_reading_ease(input_text)
    dict_all_scores["FKGL"] = textstat.flesch_kincaid_grade(input_text)
    dict_all_scores["CLI"] =textstat.coleman_liau_index(input_text)
    dict_all_scores["DCR"] = textstat.dale_chall_readability_score(input_text)
    dict_all_scores["ARI"] = textstat.automated_readability_index(input_text)
    dict_all_scores["SI"] = textstat.smog_index(input_text)
    dict_all_scores["LWF"] = textstat.linsear_write_formula(input_text)
    dict_all_scores["GF"] = textstat.gunning_fog(input_text)
    return dict_all_scores

# Function to calculate depth of dependency tree
def calculate_depth(token, current_depth=0):
    return current_depth if token.head == token else calculate_depth(token.head, current_depth + 1)

def sentence_level_calculation_all_features(input_text):
    # Initialize spaCy
    nlp = load_spacy_model("en")
    # Process the text with spaCy
    doc = nlp(input_text)
    # Iterate over each sentence in the document
    dict_all_scores = {}
    max_depth_for_all_list, avg_modifiers_per_np_all_list, lexical_density_all_list, ttr_all_list = [], [], [], []
    for sentence in doc.sents:
        alpha_tokens = [token for token in sentence if token.is_alpha]
        types = set(token.text.lower() for token in alpha_tokens)

        if len(alpha_tokens) > 0:
            ttr = len(types) / len(alpha_tokens)
        else:
            ttr = 0  # Or you could use np.nan or another placeholder to indicate no data

        ttr_all_list.append(ttr)

        # Lexical Density
        lexical_words = [token for token in sentence if token.pos_ in ["NOUN", "VERB", "ADJ", "ADV"]]
        lexical_density_all_list.append(len(lexical_words) / len(sentence) if sentence else 0)

        # Modifiers per Noun Phrase
        modifiers = sum(len([w for w in nounp if w.pos_ in ['ADJ', 'ADV']]) for nounp in sentence.noun_chunks)
        avg_modifiers_per_np_all_list.append(modifiers / len(list(sentence.noun_chunks)) if list(sentence.noun_chunks) else 0)

        # Calculate the depth of the dependency tree for each token and find the maximum
        max_depth_for_all_list.append(max(calculate_depth(token) for token in sentence))

    dict_all_scores["ttr"] = round(np.average(ttr_all_list), 2)
    dict_all_scores["lexcial_density"] = round(np.average(lexical_density_all_list), 2)
    dict_all_scores["avg_modifiers_per_noun_phrase"] = round(np.average(avg_modifiers_per_np_all_list), 2)
    dict_all_scores["max_depth"] = round(np.average(max_depth_for_all_list), 2)
    # it is averaged, need to cal abs.
    dict_all_scores["difficult_words"] = textstat.difficult_words(input_text)
    # calculate readability score in doc level.
    # dict_all_scores["FRE"] = round(np.average([textstat.flesch_reading_ease(sentence.text) for sentence in doc.sents]), 2)
    # dict_all_scores["FKGL"] = round(np.average([textstat.flesch_kincaid_grade(sentence.text) for sentence in doc.sents]), 2)
    # dict_all_scores["CLI"] = round(np.average([textstat.coleman_liau_index(sentence.text)  for sentence in doc.sents]), 2)
    # dict_all_scores["DCR"] = round(np.average([textstat.dale_chall_readability_score(sentence.text)  for sentence in doc.sents]), 2)
    # dict_all_scores["ARI"] = round(np.average([textstat.automated_readability_index(sentence.text)  for sentence in doc.sents]), 2)
    # dict_all_scores["SI"]  = round(np.average([textstat.smog_index(sentence.text)  for sentence in doc.sents]), 2)
    # dict_all_scores["LWF"] = round(np.average([textstat.linsear_write_formula(sentence.text)  for sentence in doc.sents]), 2)
    # dict_all_scores["GF"] = round(np.average([textstat.gunning_fog(sentence.text)  for sentence in doc.sents]), 2)
    # dict_all_scores["TS"] = textstat.text_standard(input_text)

    return dict_all_scores
def sentence_level_calculation_diffcult_words(input_text):

    dict_all_scores = {}
    print("Text:%s\nAbsolute textstat.difficult_words(_tgt): %s, " % (input_text, textstat.difficult_words(input_text)))
    dict_all_scores["difficult_words"] = textstat.difficult_words(input_text)
    return dict_all_scores



if __name__=="__main__":
    # use default textstat setting where lang="en". I don't explicitly set the language using set_lang.
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="yaml config file for feature based evaluation")
    parser.add_argument("--doc_or_sent", required=True, help="yaml config file for feature based evaluation")
    args = vars(parser.parse_args())
    config = load_yaml(args["config"])
    input_file = config["path_to_input_file"]
    output_scores_save = config["save_scores_path"]
    total_line_number = None if "total_line_number" not in config else config["total_line_number"]
    doc_or_sent = "doc" if args["doc_or_sent"] is None else args["doc_or_sent"]

    if "doc" == doc_or_sent:
        calculate_readability_scores_doc_level(input_sent_file=input_file, doc_readability_score_file_path = output_scores_save,
                                           total_line_number = total_line_number)
    else:
        calculate_readability_scores_sentence_level(input_sent_file=input_file, readability_scores_file_path = output_scores_save,
                         avg_score_file_path = output_scores_save, total_line_number = total_line_number)


# python llm_based_control_rewrite/scores/sentence_level_readabilty_scores.py --config experiments/gpt_openai/4_freq/old_exp/readabiltiy_scores.yaml

