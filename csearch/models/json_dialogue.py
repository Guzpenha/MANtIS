import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer


class JsonDialogue:
    """
    This class is a representation of a dialogue in the JSON dataset. Each dialogue is described by
        - A category (domain)
        - The time the dialogue started
        - The users involved
        - The utterances that were generated throughout the dialogue
        - Whether or not the dialogue had concatenated utterances (when the same users generated consecutive utterances)
    """

    def __init__(self, category: str, title: str, dialog_time, usernames, sid: SentimentIntensityAnalyzer):
        self.category = category
        self.title = title
        self.dialog_time = dialog_time
        self.usernames = usernames
        self.utterances = []
        self.__agent_utterances = []
        self.has_concatenated_utterances = 0
        self.sid = sid
        self.polarity_threshold = {
            'apple': 0.3,
            'askubuntu': 0.37,
            'dba': 0.35,
            'diy': 0,
            'electronics': 0.06,
            'english': 0.23,
            'gaming': 0.07,
            'gis': 0.35,
            'physics': 0.54,
            'scifi': 0,
            'security': 0.28,
            'stats': 0.48,
            'travel': 0.17,
            'worldbuilding': 0.58,
        }

    def __build_agent_utterances(self):
        self.__agent_utterances = list(
            filter(
                lambda utterance: utterance['actor_type'] == 'agent', self.utterances
            )
        )

    def as_dict(self) -> dict:
        """
        Converts a JsonDialogue instance to a dictionary
        Useful to "hide" the usernames attribute
        :return:
        """
        return {
            'category': self.category,
            'title': self.title,
            'dialog_time': self.dialog_time,
            'utterances': self.utterances,
            'has_concatenated_utterances': self.has_concatenated_utterances,
        }

    def __is_other_mention(self, text: str) -> bool:
        """
        Checks whether a comment contains a mention referring some other users than the ones participating in the
        dialogue.
        :param text:
        :return:
        """

        normalized_text = ''
        try:
            normalized_text = text[1:] if text[0] == '@' else text
        except IndexError:
            print(text)

        return normalized_text.startswith(self.usernames)

    @classmethod
    def __process_text(cls, text: str) -> str:
        """
        Processes the utterance by stripping all the HTML tags (except links, quotes, code) and removing
        all the newlines.
        :param text:
        :return:
        """
        clean_text = re.sub('<(?!a)(?!br)(?!blockquote)(?!pre)(?!code)(?!/blockquote)(?!/pre)(?!/code).*?>', '', text)\
            .replace('\n', '').replace('\r', '').replace('\t', '')

        return clean_text if clean_text else '<Placeholder Response>'

    @classmethod
    def __format_utterance(cls,
                           utterance: dict,
                           current_position: int,
                           is_agent: bool = False,
                           is_comment: bool = False,
                           is_accepted: bool = False) -> dict:
        """
        Given an utterance, this function formats it similar to this https://ciir.cs.umass.edu/downloads/msdialog/
        :param utterance: The current utterance to be formatted
        :param current_position: The position of the current utterance
        :param is_agent: True if the utterance has been emitted by an agent
        :param is_comment: True if the utterance is a comment
        :param is_accepted: True if the current response is the accepted one
        :return: A formatted utterance
        """

        actor_type = 'agent' if is_agent else 'user'

        if is_comment:
            return {
                'utterance': JsonDialogue.__process_text(utterance['Text']),
                'utterance_time': utterance['CreationDate_comment'],
                'utterance_pos': current_position,
                'actor_type': actor_type,
                'user_name': utterance['DisplayName_comment'],
                'user_id': utterance['UserId'],
                'votes': utterance['Score_comment'],
                'id': str(utterance['Id_post']) + '-' + str(utterance['Id_comment']),
                'is_answer': 0
            }

        return {
            'utterance': JsonDialogue.__process_text(utterance['Body']),
            'utterance_time': utterance['CreationDate_post'],
            'utterance_pos': current_position,
            'actor_type': actor_type,
            'user_name': utterance['DisplayName_post'],
            'user_id': utterance['OwnerUserId'],
            'votes': utterance['Score_post'],
            'id': str(utterance['Id_post']),
            'is_answer': 1 if is_accepted else 0
        }

    def append_utterance(self, original_post: dict, response: dict, is_accepted: bool=False) -> None:
        """
        Appends the utterance(s) to the dialog
        :param original_post: The original question
        :param response: The current response under analysis
        :param is_accepted: Is True if the current response is the accepted one
        """

        current_position = 1

        # Start of the dialogue. Add the original utterance and the first response
        if len(self.utterances) == 0:
            self.utterances.append(self.__format_utterance(original_post, current_position))
            current_position += 1

            self.utterances.append(
                self.__format_utterance(
                    response, current_position, is_agent=True, is_comment=False, is_accepted=is_accepted
                )
            )
            current_position += 1
        else:
            current_position = len(self.utterances) + 1

        # The original user posted a comment
        if response['UserId'] == original_post['OwnerUserId']:
                # and ~self.__is_other_mention(response['Text']):
            self.utterances.append(
                self.__format_utterance(response, current_position, is_agent=False, is_comment=True)
            )
            current_position += 1
        # Other users commented. If the response['Text'] is a float, that means it's not a comment
        elif not isinstance(response['Text'], float):
            self.utterances.append(
                self.__format_utterance(response, current_position, is_agent=True, is_comment=True)
            )

    @classmethod
    def renumber_utterances(cls, utterances: list):
        """
        In case utterances are appended, the counter for the utterances position need to be reset
        :param utterances:
        :return:
        """
        current_position = 1
        for utterance in utterances:
            utterance['utterance_pos'] = current_position
            current_position += 1

    def concat_consecutive_same_person_comments(self) -> None:
        """
        Concatenates consecutive utterances coming from the same user
        :return:
        """
        i = 0
        indexes_to_remove = []
        are_concatenated_comments = False

        utterances = self.utterances

        while i < len(utterances) - 1:
            last_user_id = utterances[i]['user_id']
            utterance = utterances[i]['utterance']

            j = i + 1
            current_user_id = utterances[j]['user_id']
            while last_user_id == current_user_id:
                utterance += utterances[j]['utterance']
                are_concatenated_comments = True
                indexes_to_remove.append(j)

                j += 1

                if j >= len(utterances):
                    break

                current_user_id = utterances[j]['user_id']

            utterances[i]['utterance'] = utterance
            i = j

        for i in sorted(indexes_to_remove, reverse=True):
            del utterances[i]

        if are_concatenated_comments:
            self.renumber_utterances(utterances)
            self.utterances = utterances
            self.has_concatenated_utterances = 1

    def is_feedback_final_response(self) -> bool:
        last_utterance = self.utterances[-1]
        polarity_score = self.sid.polarity_scores(last_utterance['utterance'])['compound']

        if last_utterance['actor_type'] == 'user':
            return polarity_score >= self.polarity_threshold[self.category]

        return True

    def is_two_way(self) -> bool:
        agents = set([utterance['user_id'] for utterance in self.utterances])

        if len(agents) > 2:
            return False

        return True

    def is_grounded(self) -> bool:
        if not self.__agent_utterances:
            self.__build_agent_utterances()

        # Check if at least one agent utterance is grounded (contains link)
        for agent_utterance in self.__agent_utterances:
            urls = re.findall(
                'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                agent_utterance['utterance']
            )
            if 'href' in agent_utterance['utterance'] or len(urls) > 0:
                return True

        return False

    def is_multiturn(self) -> bool:
        if not self.__agent_utterances:
            self.__build_agent_utterances()

        # Discard conversations with just 1 turn
        if len(self.__agent_utterances) <= 1:
            return False

        return True

    def is_deprecated(self) -> bool:
        if not self.__agent_utterances:
            self.__build_agent_utterances()

        for utterance in self.__agent_utterances:
            lower_utt = utterance['utterance'].lower()
            if 'edit:' in lower_utt or 'deprecated:' in lower_utt:
                return True

        return False
