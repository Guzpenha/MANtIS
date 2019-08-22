import numpy as np
import spacy
from gensim.summarization.bm25 import BM25
from math import floor


class BM25Helper:
    def __init__(self, raw_corpus: list, processed_corpus: list = None):
        self.__pipeline = spacy.load('en_core_web_sm', disable=["parser", "tagger", "ner"])
        self.raw_corpus = raw_corpus
        self.processed_corpus = self.__pre_process_corpus() if processed_corpus is None else processed_corpus
        self.raw_corpus = np.array(self.raw_corpus)
        self.model = BM25(self.processed_corpus)

    def __bm25_pre_process_utterance(self, query: str) -> list:
        """
        Tokenizes a utterance and removes stopwords and punctuation
        :param query:
        :return:
        """
        doc = self.__pipeline(query)
        return [token.text.lower() for token in doc if not token.is_stop and not token.is_punct]

    def __pre_process_corpus(self) -> list:
        """
        Prepares each utterance in the corpus to be fed to BM25
        :return:
        """
        processed_corpus = []

        print('Pre-processing the agent corpus in order to apply BM25...')

        corpus_size = len(self.raw_corpus)
        progress_increment = floor(corpus_size / 100)
        i = 0

        for doc in self.__pipeline.pipe(self.raw_corpus):
            if i % progress_increment == 0:
                print('Progress: ' + str(floor(i / progress_increment)) + '%')

            processed_corpus.append([token.text.lower() for token in doc if not token.is_stop and not token.is_punct])
            i += 1

        return processed_corpus

    def get_negative_samples(self, query: str,  n: int) -> list:
        """
        Given a query, this function returns a sample of n responses from the top 1000 potential responses obtained
        by applying BM25
        :param query:
        :param n:
        :return:
        """
        processed_query = self.__bm25_pre_process_utterance(query)

        scores = np.array(self.model.get_scores(processed_query))
        subset_length = min(1000, len(scores))
        top_queries = np.argpartition(scores, -subset_length)[-subset_length:]

        top_indexes = np.random.choice(top_queries, n, replace=False)
        unique_responses = list(set(self.raw_corpus[top_indexes]))

        while len(unique_responses) < n:
            candidate_element = np.random.choice(top_queries, 1)[0]
            if candidate_element not in unique_responses:
                unique_responses.append(candidate_element)

        return unique_responses
