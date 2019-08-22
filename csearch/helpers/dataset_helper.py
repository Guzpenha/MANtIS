import os
import json
from csearch.helpers.bm25_helper import BM25Helper


class DatasetHelper:
    @classmethod
    def __get_index_split(cls, dataset: dict, dataset_split: dict) -> dict:
        """
        Given a train/dev/test distribution, this function returns the indexes of the chronological split. The
        resulting split can deviate from the proposed percentages in case the index falls between dialogs occuring
        at the same time
        NOTE: The dataset is assumed to be already ordered, as the previous processes take care of that
        :param dataset
        :param dataset_split: Should be a dict in the form
        {
            'train': 0.8,
            'dev': 0.1,
            'test': 0.1
        }
        :return:
        """
        dataset_length = len(dataset)
        dev_start_index = int(dataset_split['train'] * dataset_length)

        while (dataset[dev_start_index]['dialog_time']
               == dataset[dev_start_index + 1]['dialog_time']):
            dev_start_index += 1

        test_start_index = dev_start_index + int(dataset_split['dev'] * dataset_length)
        while (dataset[test_start_index]['dialog_time']
               == dataset[test_start_index + 1]['dialog_time']):
            test_start_index += 1

        return {
            'dev_start_index': dev_start_index,
            'test_start_index': test_start_index,
        }

    @classmethod
    def get_split_dataset(cls, dataset: dict, dataset_split: dict) -> dict:
        index_split = DatasetHelper.__get_index_split(dataset, dataset_split)
        split_dataset = {
            'train': {},
            'dev': {},
            'test': {}
        }

        current_dataset_allocation = 'train'
        for (key, entry) in dataset.items():
            if key == index_split['dev_start_index']:
                current_dataset_allocation = 'dev'

            elif key == index_split['test_start_index']:
                current_dataset_allocation = 'test'

            split_dataset[current_dataset_allocation].update({key: entry})

        return split_dataset

    @classmethod
    def __add_topic_to_dataset(cls, topic, allocation, dataset, current_id) -> tuple:
        root_folder = os.path.dirname(__file__) + '/../../stackexchange_dump/'
        dataset_file = root_folder + topic + '/data_' + allocation + '.json'

        if not os.path.isfile(dataset_file):
            raise Exception('Could not find json dataset for topic: ' + topic)

        with open(dataset_file, 'r') as f:
            json_data = json.load(f)

        for i, entry in json_data.items():
            dataset[current_id] = entry
            current_id += 1

        return dataset, current_id

    @classmethod
    def merge_topics(cls, topics: list):

        merged_dataset = {
            'train': {},
            'dev': {},
            'test': {}
        }

        current_ids = {
            'train': 0,
            'dev': 0,
            'test': 0
        }

        for allocation in merged_dataset.keys():
            for topic in topics:
                merged_dataset[allocation], current_ids[allocation] = DatasetHelper.__add_topic_to_dataset(
                    topic, allocation, merged_dataset[allocation], current_ids[allocation]
                )

        for allocation in merged_dataset.keys():
            with open(os.path.dirname(__file__) + '/../../stackexchange_dump/merged_' + allocation + '.json', 'w') as fp:
                json.dump(merged_dataset[allocation], fp)

    @classmethod
    def __build_raw_agent_corpus(cls, json_data: dict) -> list:
        """
        Builds the agent corpus, which is used to generate additional dialogues using BM25
        :return:
        """
        corpus = []

        for (key, dialogue) in json_data.items():
            corpus += (DatasetHelper.__process_agent_responses(dialogue))

        return corpus

    @classmethod
    def __build_multi_topic_raw_agent_corpus(cls, json_data: dict) -> dict:
        """
        Builds the agent corpus, which is used to generate additional dialogues using BM25
        :return:
        """
        corpus = {}

        for (key, dialogue) in json_data.items():
            topic = dialogue['category']
            if topic not in corpus:
                corpus[topic] = []

            corpus[topic] += (DatasetHelper.__process_agent_responses(dialogue))

        return corpus

    @classmethod
    def build_bm25_helper(cls, json_data: dict) -> BM25Helper:
        """
        Build the bm25 helper, that will be used to perform negative sampling
        :return:
        """
        return BM25Helper(DatasetHelper.__build_raw_agent_corpus(json_data))

    @classmethod
    def build_multi_topic_bm25_helper(cls, json_data: dict) -> dict:
        bm25_helper = {}
        raw_agent_corpus = DatasetHelper.__build_multi_topic_raw_agent_corpus(json_data)

        for topic, entry in raw_agent_corpus.items():
            print('Building BM25 corpus for topic: ' + topic)
            bm25_helper[topic] = BM25Helper(entry)

        return bm25_helper

    @classmethod
    def __process_agent_responses(cls, dialogue: dict) -> list:
        """
        For a given dialogue, generates a list of pre-processed agent responses (ready for BM25)
        :param dialogue:
        :return:
        """
        utterances = dialogue['utterances']
        agent_utterances = list(
            filter(
                lambda utterance: utterance['actor_type'] == 'agent', utterances
            )
        )

        return [utterance['utterance'] for utterance in agent_utterances]

