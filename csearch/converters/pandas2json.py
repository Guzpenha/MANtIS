from math import floor
from pandas import DataFrame
from csearch.models.json_dialogue import JsonDialogue
from nltk.sentiment.vader import SentimentIntensityAnalyzer


class Pandas2JSON:
    """
    This class handles the conversion of a Pandas dataframe to a JSON dataset
    """

    def __init__(self, df: DataFrame, topic: str):
        self.df = df
        self.topic = topic
        self.__output = {}
        self.__global_index = 0
        self.sid = SentimentIntensityAnalyzer()

    def __init_entry(self, utterance: dict, usernames: tuple) -> JsonDialogue:
        """
        Format the standard entry before adding the utterances
        :param utterance:
        :return: An dialogue "stub"
        """
        return JsonDialogue(self.topic, utterance['Title'], utterance['CreationDate_post'], usernames, self.sid)

    def __add_to_output(self, entry: JsonDialogue) -> None:
        if len(entry.utterances) == 0:
            return

        entry.concat_consecutive_same_person_comments()

        if not entry.is_grounded():
            return

        if not entry.is_multiturn():
            return

        if not entry.is_feedback_final_response():
            return

        if not entry.is_two_way():
            return

        if entry.is_deprecated():
            return

        self.__output[self.__global_index] = entry.as_dict()
        self.__global_index += 1

    @classmethod
    def __get_usernames(cls, df: DataFrame) -> list:
        users_comments = df.loc[~df['DisplayName_comment'].isnull()].drop_duplicates('DisplayName_comment')[
            'DisplayName_comment'].tolist()
        users_posts = df.drop_duplicates('DisplayName_post')['DisplayName_post'].tolist()

        return users_comments + list(set(users_posts) - set(users_comments))

    def __generate_dialogues_from_responses(self, original_post, responses) -> None:
        """
        Given a question, this function processes all the responses and turns them into separate dialogues
        :param original_post: The original question on the thread
        :param responses: A list of all the responses
        :return:
        """
        if responses.shape[0] == 0:
            return

        user_username = original_post['DisplayName_post']
        usernames = tuple(set(self.__get_usernames(responses)) - set(user_username))
        # current_agent_usernames = []

        current_post_id = responses.iloc[0]['Id_post']
        accepted_answer_id = original_post['AcceptedAnswerId']
        original_user_id = original_post['OwnerUserId']

        entry = self.__init_entry(original_post, usernames)
        for index, response in responses.iterrows():
            if response['OwnerUserId'] == original_user_id:
                continue

            current_response_id = response['Id_post']
            is_accepted = (current_response_id == accepted_answer_id)

            if current_response_id == current_post_id:
                entry.append_utterance(original_post, response, is_accepted)
            else:
                self.__add_to_output(entry)

                entry = self.__init_entry(original_post, usernames)
                entry.append_utterance(original_post, response, is_accepted)

                current_post_id = current_response_id

        self.__add_to_output(entry)

    def convert(self) -> dict:
        """
        Given a dataframe, this function turns it into a JSON array with dialogues and utterances
        :return: JSON dataset
        """
        original_posts_df = self.df.loc[self.df['PostTypeId'] == '1'].drop_duplicates('Id_post')
        original_posts_df = original_posts_df.reset_index(drop=True)

        responses_df = self.df.loc[
            (self.df['PostTypeId'] == '2') &
            (self.df['Text'].isnull() | (~self.df['Text'].isnull() & ~self.df['UserId'].isnull()))
        ]

        responses_df = responses_df.reset_index(drop=True)

        total_progress_increment = floor(original_posts_df.shape[0] / 100)

        for progress_index, original_post in original_posts_df.iterrows():
            if progress_index % total_progress_increment == 0:
                print('Progress: ' + str(floor(progress_index / total_progress_increment)) + '%')

            original_post_id = original_post['Id_post']
            responses_df_current = responses_df.loc[responses_df['ParentId'] == original_post_id]

            self.__generate_dialogues_from_responses(original_post, responses_df_current)

        return self.__output
