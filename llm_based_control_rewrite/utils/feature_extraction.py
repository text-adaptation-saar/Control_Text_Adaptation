"""
List of features supported:
1) maximum dependency depth  ratio
2) target/source ratio in terms of character count  LENGTH RATIO
3) Levenshtein edit ratio: between 0 and 1  SURFACE FORM SIMILARITY
4) word frequency in the target: third quantile of the word frequencies ratio

"""

# If sentences are empty or none, return feature value in negative number (-100).
# if not sentence or all(not sentence.strip() for sentence in sentence):
#     return -100
# else:

import spacy
import numpy as np
import string
import re
import Levenshtein
from nltk.corpus import stopwords
import nltk
from textstat import textstat

from llm_based_control_rewrite.utils.prepare_word_embeddings_frequency_ranks import load_ranks


def load_spacy_model(lang):
    lang_model = {"en": "en_core_web_sm", "de": "de_core_news_sm"}
    if lang.lower() not in lang_model:
        print("Language choice not supported, defaulting to English (other option: German)")
        lang = "en"
    nlp_model = spacy.load(lang_model[lang])
    return nlp_model

def walk_tree(node, depth):
    """ Pass a spacy root of a sentence, return the maximum depth """
    if node.n_lefts + node.n_rights > 0:
        return max(walk_tree(child, depth + 1) for child in node.children)
    else:
        return depth


def maximum_dependency_depth(s, t, lang, absolute):
    """ Language-dependent, language model loaded from spacy, parse
    Calculate the maximum dependency depth of the SOURCE (complex) and divide it by that of TARGET (simple)
    """
    # empty target sentences:
    if not t or all(not t.strip() for t in t):
          return -100

    nlp_model = load_spacy_model(lang)
    doc = nlp_model(t)  # target
    #all_depths = [walk_tree(sent.root, 0) for sent in doc.sents]
    #print(all_depths)
    if len(t) == 0:
        max_depth_target = 0
    else:
        max_depth_target = max([walk_tree(sent.root, 0) for sent in doc.sents])

    if absolute:
        return max_depth_target

    # only when ratio is needed to calcaute and source is empty.
    if not s or all(not s.strip() for s in s):
          return -100

    doc = nlp_model(s)  # source
    #all_depths = [walk_tree(sent.root, 0) for sent in doc.sents]
    #print(all_depths)
    max_depth_source = max([walk_tree(sent.root, 0) for sent in doc.sents])

    if max_depth_source == 0:  # single word sentences...
        #print("source depth 0??")
        #import pdb; pdb.set_trace()
        max_depth_source = 0.5
    if max_depth_target == 0:
        #print("target depth 0??")
        #import pdb; pdb.set_trace()
        max_depth_target = 0.5
    depth_ratio = max_depth_target / max_depth_source

    return depth_ratio


def maximum_dependency_length(s, t, lang, absolute):
    """ Language-dependent, language model loaded from spacy, parse
    Calculate the maximum dependency length of the SOURCE (complex) and divide it by that of TARGET (simple)
    """
    # lang_model = {"en": "en_core_web_sm", "de": "de_core_news_sm"}
    # if lang.lower() not in lang_model:
    #     print("Language choice not supported, defaulting to English (other option: German)")
    #     lang = "en"

    # empty target sentences:
    if not t or all(not t.strip() for t in t):
        return -100

    nlp_model = load_spacy_model(lang)

    doc = nlp_model(t)
    if len(t) == 0:
        max_dep_length_target = 0
    else:
        max_dep_length_target = max([calc_max_dependency_length(sent) for sent in doc.sents])
    if absolute:
        return max_dep_length_target

    # only when ratio is needed to calcaute and source is empty.
    if not s or all(not s.strip() for s in s):
        return -100

    doc = nlp_model(s)
    max_dep_length_source = max([calc_max_dependency_length(sent) for sent in doc.sents])

    if max_dep_length_source == 0:  # single word sentences...
        max_dep_length_source = 0.5
    if max_dep_length_target == 0:
        max_dep_length_target = 0.5
    dep_length_ratio = max_dep_length_target / max_dep_length_source

    return dep_length_ratio

def calc_max_dependency_length(doc):
    # Initialize max dependency length variable
    max_dep_length = 0

    # Calculate dependency length and get max of it.
    for token in doc:
        # Skip any punctuation marks dependency length.
        if str(token) not in string.punctuation:
            dep_length = abs(token.i - token.head.i)
            max_dep_length = max(max_dep_length, dep_length)

    return max_dep_length

def calc_no_of_diffcult_words(_src, _tgt, absolute):
    """ Calculate no.of difficult words between target and source.
    """
    # empty target sentences:
    if not _tgt or all(not _tgt.strip() for _tgt in _tgt):
        return -100

    if absolute:
        print("Text:%s\nAbsolute textstat.difficult_words(_tgt): %s, " % (_tgt,textstat.difficult_words(_tgt)))
        return textstat.difficult_words(_tgt)

    # only when ratio is needed to calcaute and source is empty.
    if not _src or all(not _src.strip() for _src in _src):
        return -100

    return textstat.difficult_words(_tgt)/textstat.difficult_words(_src)

def cal_grade_level(_src, _tgt, absolute):
    """ Calculate fkgl grade between target and source.
        """
    # empty target sentences:
    if not _tgt or all(not _tgt.strip() for _tgt in _tgt):
        return -100

    if absolute:
        print("Text:%s\nAbsolute textstat.flesch_kincaid_grade(_tgt): %s, sentence: %s" % (_tgt, textstat.flesch_kincaid_grade(_tgt), _tgt))
        return round(textstat.flesch_kincaid_grade(_tgt))

    # only when ratio is needed to calcaute and source is empty.
    if not _src or all(not _src.strip() for _src in _src):
        return -100
    src_grade = round(textstat.flesch_kincaid_grade(_src))
    ratio = round(textstat.flesch_kincaid_grade(_tgt)) / (0.5 if src_grade == 0 else src_grade)
    return round(ratio,2)

def character_length_ratio(_src, _tgt, absolute):
    """ Calculate the ratio between target and source length in characters
    len(simple) / len(complex)  OR len(target) / len(source)
    """
    # empty target sentences:
    if not _tgt or all(not _tgt.strip() for _tgt in _tgt):
        return -100

    if absolute:
        return average_character_length(_tgt)

    # only when ratio is needed to calcaute and source is empty.
    if not _src or all(not _src.strip() for _src in _src):
        return -100

    return average_character_length(_tgt)/average_character_length(_src)

def average_character_length(text):
    total_words = 0
    sentences = split_sentences(text)
    for sentence in sentences:
        total_words += len(sentence)

    average_character_length = total_words / len(sentences)
    return average_character_length

def word_count_ratio(_src, _tgt, absolute):
    """ Calculate the ratio between target and source word count
    word_count(simple) / word_count(complex)  OR word_count(target) / word_count(source)
    """
    # empty target sentences:
    if not _tgt or all(not _tgt.strip() for _tgt in _tgt):
        return -100

    if absolute:
        return calculate_average_word_count(_tgt)

    # only when ratio is needed to calcaute and source is empty.
    if not _src or all(not _src.strip() for _src in _src):
        return -100

    return calculate_average_word_count(_tgt)/calculate_average_word_count(_src)


def calculate_average_word_count(text):
    total_words = 0
    sentences = split_sentences(text)
    for sentence in sentences:
        words = sentence.split()
        total_words += len(words)

    average_word_count = total_words / len(sentences)
    return average_word_count

def split_sentences(text):
    # Download the Punkt tokenizer models. It is needed for sentence tokenization.
    # This line should be run once to download necessary data.
    # nltk.download('punkt')
    sentences = nltk.sent_tokenize(text)
    return sentences


def Levenshtein_ratio(c, s):
    """ Calculate the Levenshtein ratio between the source and target: the ratio is normalized, 0-1
    Note that this ratio is symmetric: c->s and s->c gives the same score
    Higher score = higher similarity
    """
    return Levenshtein.ratio(c, s)

# -------------------Iza's code for freq rank cal START
# Only difference here is return lowest rank which is 1 for EMPTY LIST (this situation comes when there is no words after stopword, punctuation removals.)
# so that we can distinguish rare words log-rank (i.e max) and stop/punctuation words (min rank1).
# calculated Freq Rank for entire given text. (not an avg. rank if there is multiple sentences)
# -------------------------------------------------------------
def get_freq_rank_word(word, ranks_dict):
    """ If the words in is vocabulary, return its rank (int), else return the very final rank """
    if word in ranks_dict:
        return ranks_dict[word] + 1
    return len(ranks_dict) + 1


def get_log_freq_rank_word(word, ranks):
    """ If the words in is vocabulary, return its log rank (int) with natural log as base,
    else return the log of the very final rank """
    if word in ranks:
        # print("word:%s\trank:%s\tlog-rank:%s\tlog-rank+1:%s " %(word, ranks[word], np.log(ranks[word]), np.log(ranks[word] + 1) ))
        return np.log(ranks[word] + 1)
    if word == "-NO-WORDS-": # this situation comes when there is no words after stopword removals.
        # return np.log(len(ranks) + 1)
        # if so return lowest rank which is 1, so that we can distinguish rare words log-rank (i.e max) and stop words.
        print(f"Empty list after stopword, punctuation removals. hence return lowest freq rank 1 (np.log(1 + 1))")
        return [np.log(1 + 1)]  # adding another +1 ensures that the argument to np.log is never zero, thereby avoiding a math error.
    return np.log(len(ranks) + 1)


def word_checks(token):
    """
    :param token: a string of the token in sentence
    :return: Boolean
    This function check if the token is a punctuation symbol or a number. If any of the two, return False
    """
    is_int_or_float = re.compile(r"^[+-]?((\d+(\.\d+)?)|(\.\d+))$")  # should match with "1", "3.68", but NOT "3s1"
    if token in string.punctuation:
        return False
    if is_int_or_float.match(token):
        return False
    return True


def stop_word_check(token, lang):
    if lang == "de":
        _stopwords = stopwords.words("german")
    else:
        _stopwords = stopwords.words("english")
    if token in _stopwords:
        return False
    return True


def properties_word_freq_in_sentence(frequency_ranks, all_ranks):
    """
    :param frequency_ranks: a list of float/int frequency ranks from selected words in sentence
    :param all_ranks: a dictionary of ranks of word frequencies
    :return: third quantile (the value right between the median and the max)
    """
    #mu = round(np.mean(frequency_ranks), 2)
    #mode = round(np.max(frequency_ranks), 2)
    #sd = np.std(frequency_ranks)
    #medi = np.median(frequency_ranks)
    if not frequency_ranks:  # if the list of ranks is emtpy because all words were numbers or stopwords or punct.
        frequency_ranks = [get_log_freq_rank_word("-NO-WORDS-", all_ranks)]

    #first_quantile = np.quantile(frequency_ranks, 0.25)
    third_quantile = np.quantile(frequency_ranks, 0.75)
    #q3q1 = third_quantile - first_quantile

    # print("Mean %f, Mode %f, Median %f, SD %f, Q1 %f, Q3 %f, Q3-Q1 %f" % (mu, mode, medi, sd, first_quantile,
    #                                                                       third_quantile, q3q1))
    return third_quantile


def word_frequency_rank(source, target, lang, _ranks, absolute):
    """
    Each word is associated with a frequency, For a sentence we get a distribution of frequencies.
    Properties of a distribution: mean, standard dev, quartiles
    Think about using log of the frequency rank
    """
    # print("Start Freq-rank calculation process!.........")
    # step 0: load the rank dictionary
    #ranks = load_ranks(lang)

    # step 1: tokenize
    # lang_model = {"en": "en_core_web_sm", "de": "de_core_news_sm"}
    # if lang.lower() not in lang_model:
    #     print("Language choice not supported, defaulting to English (other option: German (de))")
    #     lang = "en"
    #
    # nlp_model = spacy.load(lang_model[lang])


    # empty target sentences:
    if not target or all(not target.strip() for target in target):
        return -100
    nlp_model = load_spacy_model("en")
    doc_target = nlp_model(target)

    # step 2: get the (log) frequency rank for each token, observe for entire sentence
    # but ignore punctuation and numbers as well as stopwords
    considered_tokens = [w.text for w in doc_target if word_checks(w.text) and stop_word_check(w.text, lang)]
    # target_ranks = [get_freq_rank_word(w.text, ranks) for w in doc if word_checks(w.text) and
    #                 stop_word_check(w.text, lang)]
    target_ranks = [get_log_freq_rank_word(w.text, _ranks) for w in doc_target if word_checks(w.text) and
                    stop_word_check(w.text, lang)]
    third_quantile_target = properties_word_freq_in_sentence(target_ranks, _ranks)

    if absolute:
        return third_quantile_target

    # only when ratio is needed to calcaute and source is empty.
    if not source or all(not source.strip() for source in source):
        return -100
    # repeat the process for the source and return the ratio
    doc_source = nlp_model(source)
    source_ranks = [get_log_freq_rank_word(w.text, _ranks) for w in doc_source if word_checks(w.text) and
                    stop_word_check(w.text, lang)]
    third_quantile_source = properties_word_freq_in_sentence(source_ranks, _ranks)

    return third_quantile_target / third_quantile_source

# -------------------Iza's code for freq rank cal END -------------------------------------------------------------
def get_bin_value(x_val, this_feature_bins):
    # using numpy digitize
    idx = np.digitize([x_val], this_feature_bins, right=True).item()
    if idx == len(this_feature_bins):  # the idx is out of range if the value is bigger than the last bin
        idx -= 1
    return this_feature_bins[idx]


def feature_bins_bundle_sentence(source, target, lang, list_of_desired_features, feature_bin_dictionary, _freq_ranks,
                                 absolute=False, freq_rank_cal_property = "q3,log"):
    bundle = {}
    bundle_exact = {}

    if "dependency_depth" in list_of_desired_features:
        v = maximum_dependency_depth(source, target, lang, absolute)
        v_bin = get_bin_value(v, feature_bin_dictionary["dependency_depth"])
        bundle["dependency_depth"] = v_bin
        bundle_exact["dependency_depth"] = v
    if "dependency_length" in list_of_desired_features:
        v = maximum_dependency_length(source, target, lang, absolute)
        v_bin = get_bin_value(v, feature_bin_dictionary["dependency_length"])
        bundle["dependency_length"] = v_bin
        bundle_exact["dependency_length"] = v
    if "difficult_words" in list_of_desired_features:
        print("hit difficult_words")
        v = calc_no_of_diffcult_words(source, target, absolute)
        v_bin = get_bin_value(v, feature_bin_dictionary["difficult_words"])
        bundle["difficult_words"] = v_bin
        bundle_exact["difficult_words"] = v
    if "word_count" in list_of_desired_features:
        v = word_count_ratio(source, target, absolute)
        v_bin = get_bin_value(v, feature_bin_dictionary["word_count"])
        bundle["word_count"] = v_bin
        bundle_exact["word_count"] = v
    # low priority features.
    if "frequency" in list_of_desired_features:
        v = word_frequency_rank(source, target, lang, _freq_ranks, absolute)
        print(f"Final Frequency Rank value: {v}, Type: {type(v)}\n")  # Debugging line
        v_bin = get_bin_value(v, feature_bin_dictionary["frequency"])
        bundle["frequency"] = v_bin
        bundle_exact["frequency"] = v
    if "length" in list_of_desired_features:
        v = character_length_ratio(source, target, absolute)
        v_bin = get_bin_value(v, feature_bin_dictionary["length"])
        bundle["length"] = v_bin
        bundle_exact["length"] = v
    if "levenshtein" in list_of_desired_features:
        v = Levenshtein_ratio(source, target)
        v_bin = get_bin_value(v, feature_bin_dictionary["levenshtein"])
        bundle["levenshtein"] = v_bin
        bundle_exact["levenshtein"] = v
    if "grade" in list_of_desired_features:
        print("hit grade-level")
        v = cal_grade_level(source, target, absolute)
        v_bin = get_bin_value(v, feature_bin_dictionary["grade"])
        bundle["grade"] = v_bin
        bundle_exact["grade"] = v
    return bundle, bundle_exact

def get_max_dep_depth_of_given_sent(sentence):
    # lang_model = {"en": "en_core_web_sm", "de": "de_core_news_sm"}
    if not sentence or all(not sentence.strip() for sentence in sentence):
        return -100
    else:
        nlp_model = load_spacy_model("en")
        doc = nlp_model(sentence)
        max_dep_depth = max([walk_tree(sent.root, 0) for sent in doc.sents])
        return max_dep_depth

def get_max_dep_length_of_given_sent(sentence):
    # lang_model = {"en": "en_core_web_sm", "de": "de_core_news_sm"}
    if not sentence or all(not sentence.strip() for sentence in sentence):
        return -100
    else:
        nlp_model = load_spacy_model("en")
        doc = nlp_model(sentence)
        max_dep_length = max([calc_max_dependency_length(sent) for sent in doc.sents])
        return max_dep_length

def get_freq_rank_of_given_sent(sentence):
    frequency_ranks = load_ranks("en")
    freq_rank_of_given_sentence = word_frequency_rank("", sentence, "en", frequency_ranks , True)
    return freq_rank_of_given_sentence
def get_no_of_difficult_words_of_given_sent(sentence):
    if not sentence or all(not sentence.strip() for sentence in sentence):
        return -100
    else:
        return textstat.difficult_words(sentence)

def get_word_count_of_given_sent(sentence):
    if not sentence or all(not sentence.strip() for sentence in sentence):
        return -100
    else:
        return calculate_average_word_count(sentence)

def get_length_of_given_sent(sentence):
    return average_character_length(sentence)

def get_grade_level_of_given_sent(sentence):
    return min(13, max(0, round(textstat.flesch_kincaid_grade(sentence))))

def get_target_feature_value(src_sent_control_token_value, requested_control_token_value, absolute=False):

    if absolute:
        return round(requested_control_token_value)
    else:
        tgt = round((src_sent_control_token_value * requested_control_token_value))
        print("absolute value calculation: (src=%s),\t*\t(ratio=%s)\t=\t(tgt=%s)" % (src_sent_control_token_value, requested_control_token_value, tgt ))
    # requested feature value is given in ratio.
        return tgt


def depth_indexed_tree(token, depth=0):
    return f"({depth}, {token.text})" + " ".join([depth_indexed_tree(child, depth + 1) for child in token.children])
def dependency_tree_print_with_depth(lang, sentence):

    # Load Spacy model
    nlp = load_spacy_model(lang)
    doc = nlp(sentence)

    trees = []
    for sent in doc.sents:
        root = [tok for tok in sent if tok.head == tok][0]
        trees.append(depth_indexed_tree(root))
    return trees

    # # Function to calculate the depth of a token in the dependency tree
    # def token_depth(token):
    #     depth = 0
    #     while token.head != token:
    #         depth += 1
    #         token = token.head
    #     return depth
    #
    # # Create a formatted string representation of the dependency tree
    # dependency_tree = []
    # for token in doc:
    #     depth = token_depth(token)
    #     # dependency_tree.append(f"({token.text}, {token.dep_}, {token.head.text}, Dependency Depth:{depth})")
    #     dependency_tree.append(f"({token.text}, {token.dep_}, Dependency Depth:{depth})")
    #
    # # Joining the formatted strings and printing them in a single line
    # dependency_tree_line = " ".join(dependency_tree)
    # return dependency_tree_line


def length_indexed_tree(token):
    """
    Recursive function to create a length-indexed linearized tree, excluding punctuation.
    """
    if token.is_punct:
        return ""

    length = abs(token.i - token.head.i) if token.dep_ != "ROOT" else 0
    children_repr = " ".join(filter(None, [length_indexed_tree(child) for child in token.children]))

    return f"({length}, '{token.text}')" if children_repr == "" else f"({length}, '{token.text}') [{children_repr}]"

def dependency_tree_print_with_length(lang, sentence):
    # Load Spacy model
    nlp = load_spacy_model(lang)
    doc = nlp(sentence)

    length_trees = []
    # Generate and print the length-indexed linearized tree for each sentence
    for sent in doc.sents:
        root = [tok for tok in sent if tok.head == tok][0]
        length_trees.append(length_indexed_tree(root))
    return length_trees

def print_difficult_words(src):
    difficult_words_list = []
    for word in src.split(" "):
        if textstat.difficult_words(word):
            difficult_words_list.append(word)

    return difficult_words_list


def print_word_count(src):
    return  src.split(" ")

def print_char_list(src):
    # Convert the input string to a list of characters
    char_list = list(src)
    # Return the list of characters
    return char_list


