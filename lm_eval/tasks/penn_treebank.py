"""
TODO
"""
import re
from lm_eval.base import PerplexityTask


_CITATION = """TODO"""


def penn_treebank_detokenizer(string):
    string = string.replace(" '", "'")
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" n't", "n't")
    string = string.replace(" N ", " 1 ")
    string = string.replace("$ 1", "$1")
    string = string.replace("# 1", "#1")
    return string


class PennTreebank(PerplexityTask):
    VERSION = 0
    DATASET_PATH = "ptb_text_only"
    DATASET_NAME = None

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        pass

    def validation_docs(self):
        pass

    def test_docs(self):
        return [" ".join(map(self._process_doc, self.dataset["test"]))]

    def _process_doc(self, doc):
        return doc["sentence"]

    def doc_to_target(self, doc):
        return penn_treebank_detokenizer(doc)

    def should_decontaminate(self):
        return True

    def count_words(self, doc):
        # count number of words in *original doc before detokenization*
        return len(re.split(r"\s+", doc))
