import json
import csv
from csearch.converters.json2training import JSON2Training
from csearch.converters.json2training import Json2EasyTraining
from csearch.helpers.bm25_helper import BM25Helper
from csearch.helpers.dataset_helper import DatasetHelper


class TrainingSetBuilder:
    def __init__(self, json_location):
        self.__json_location = json_location

    def __write_tsv(self, file_name: str, data: list) -> None:
        """
        Given a filename and a list, this function writes the list in tsv format
        :param file_name:
        :param data:
        :return:
        """
        with open(self.__json_location + '/' + file_name, 'w',) as tsv_file:
            writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
            for entry in data:
                writer.writerow(entry)

    def __write_array(self, file_name: str, data:list) -> None:
        with open(self.__json_location + '/' + file_name, 'w') as f:
            for entry in data:
                f.write('%s\n' % entry)

    def __build_bm25_helper(self, is_easy):
        allocation = ['train', 'dev']
        json_data_for_bm25 = {}
        current_index = 0

        for entry in allocation:
            with open(self.__json_location + '/merged_' + entry + '.json', 'r') as f:
                json_data = json.load(f)

            for i, dialogue in json_data.items():
                json_data_for_bm25[current_index] = dialogue
                current_index += 1

        if is_easy:
            return DatasetHelper.build_multi_topic_bm25_helper(json_data_for_bm25)

        return DatasetHelper.build_bm25_helper(json_data_for_bm25)

    def build(self, is_easy=False) -> None:
        """
        Given a json structure, this function builds a tsv containing all possible (label, context, response) triples
        that can be obtained from the dialogues
        :return:
        """
        allocation = ['train', 'dev', 'test']

        bm25_helper = self.__build_bm25_helper(is_easy)

        for entry in allocation:
            with open(self.__json_location + '/merged_' + entry + '.json', 'r') as f:
                json_data = json.load(f)

            output_file_name = 'data_' + entry
            if is_easy:
                json2training_converter = Json2EasyTraining(json_data, bm25_helper)
                output_file_name += '_easy'
            else:
                json2training_converter = JSON2Training(json_data, bm25_helper)

            training_set = json2training_converter.convert()
            dialog_lookup_table = json2training_converter.get_dialog_lookup_table()

            self.__write_tsv(output_file_name + '.tsv', training_set)
            self.__write_array(output_file_name + '_lookup' '.txt', dialog_lookup_table)
