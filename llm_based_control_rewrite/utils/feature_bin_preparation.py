"""
Pool feature extractors into a single function and bin their values
"""

# prepare the bins in advance
# this will circumvent the problem of first loading all instances, calculating their features, binning them and
# then iterating over the instances again to assign feature bins and write into file

# BINS
import numpy as np


def create_bins():
    dependency_depth_bins = np.arange(0.05, 2.5, 0.05)
    dependency_length_bins = np.arange(0.05, 2.5, 0.05)
    difficult_words = np.arange(0.05, 2.5, 0.05)
    word_count_bins = np.arange(0.05, 2.5, 0.05)
    frequency_bins = np.arange(0.05, 2.5, 0.05)  # min 0.05, max 2.0
    length_bins = np.arange(0.05, 2.5, 0.05)
    levenshtein_bins = np.arange(0.05, 1.5, 0.05)  # min 0.05, max 1.0
    grade_bins = np.arange(0.05, 2.5, 0.05)  # min 0.05, max 2.0
    return {"dependency_depth": dependency_depth_bins, "dependency_length": dependency_length_bins,
            "difficult_words":difficult_words, "word_count": word_count_bins,
            "frequency": frequency_bins, "length": length_bins, "levenshtein": levenshtein_bins,
            "grade": grade_bins}

