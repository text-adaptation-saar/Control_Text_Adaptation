from abc import ABC, abstractmethod


class LLModels(ABC):
    def __init__(self,
                requested_dependency_depth,
                requested_dependency_length,
                requested_difficult_words,
                requested_word_count,
                requested_frequency,
                requested_length,
                requested_levenshtein,
                requested_absolute_feature_value = False,
                requested_grade=None
                ):

        self.dependency_depth =  requested_dependency_depth
        self.dependency_length = requested_dependency_length
        self.difficult_words = requested_difficult_words
        self.word_count = requested_word_count
        # frequency sometime categorical.
        self.frequency = requested_frequency
        self.length = requested_length
        self.levenshtein = requested_levenshtein
        self.requested_absolute_feature_value = requested_absolute_feature_value
        self.grade = requested_grade

    # @abstractmethod
    # def generate(self):
    #     pass

    @abstractmethod
    def fine_tune(self):
        pass
