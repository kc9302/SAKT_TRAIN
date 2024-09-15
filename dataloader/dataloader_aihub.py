import numpy as np
from torch.utils.data import Dataset
import logging
import polars as pl

# 패키지 초기화 함수 불러오기
from common.utils import match_sequence_length

final_dict = {
    'user': [],
    'question_sequences': [],
    'response_sequences': []
}


class AIHUB(Dataset):

    def __init__(self, sequence_length) -> None:
        super().__init__()        
        self.dataset_path = None
        self.proc_user_list = None
        self.number_question = None
        self.question_sequences, \
        self.response_sequences, \
        self.question_list, \
        self.user_list, \
        self.question_to_index, \
        self.user_to_index = self.preprocess()

        self.number_user = self.user_list.shape[0]

        if sequence_length:
            self.proc_user_list, \
            self.question_sequences, \
            self.response_sequences = match_sequence_length(user_list=self.user_list, \
                                                            question_sequences=self.question_sequences, \
                                                            response_sequences=self.response_sequences, \
                                                            sequence_length=sequence_length)

        self.length = len(self.question_sequences)

        """
        전처리한 모든 데이터 저장.
        """
        for user, question, response in zip(self.proc_user_list, self.question_sequences, self.response_sequences):
            final_dict["user"].append(user)
            final_dict["question_sequences"].append(question)
            final_dict["response_sequences"].append(response)

    def __getitem__(self, index):
        return self.question_sequences[index], self.response_sequences[index]

    def __len__(self):
        return self.length

    def preprocess(self):
        logging.debug(
            "\n" + "\n" + " ###################### " + \
            "\n" + " #### get raw data #### " + \
            "\n" + " ###################### " + "\n"
        )

        aihub_df = pl.read_csv(".\dataset\\ai_hub_sample.csv")

        logging.debug("target grade record count : {}".format(str(len(aihub_df))))

        # check record data question id
        record_question_id = aihub_df.select(pl.col("assessmentItemID").unique())
        logging.debug("record_question_id_count : {}".format(str(len(record_question_id))))

        self.number_question = len(record_question_id)

        user_list = aihub_df.select(pl.col("learnerID").unique()).to_pandas()["learnerID"]
        question_list = aihub_df.select(pl.col("assessmentItemID").unique()).to_pandas()["assessmentItemID"]

        logging.debug("user_list : {}".format(str(len(user_list))))
        logging.debug("question_list : {}".format(str(len(question_list))))
        user_to_index = {user: index for index, user in enumerate(user_list)}
        question_to_index = {question: index for index, question in enumerate(question_list)}

        question_sequences = []
        response_sequences = []
        user_sequences = []

        for user in user_list:
            df_u = aihub_df.filter(pl.col("learnerID") == user)
            response_sequence = df_u.select(pl.col("answerCode")).to_pandas()["answerCode"]
            question_sequence = np.array([question_to_index[question] for question in df_u["assessmentItemID"]])
            user_sequences.append(df_u.select(pl.col("learnerID")).to_pandas()["learnerID"])
            question_sequences.append(question_sequence)
            response_sequences.append(response_sequence)

        # save file
        np.save('question_sequences.npy', np.array(question_sequences, dtype=object))
        np.save('response_sequences.npy', np.array(response_sequences, dtype=object))
        np.save('question_list.npy', np.array(question_list, dtype=object))
        np.save('user_list.npy', np.array(user_list, dtype=object))
        np.save('question_to_index.npy', np.array(question_to_index, dtype=object))
        np.save('user_to_index.npy', np.array(user_to_index, dtype=object))

        return question_sequences, response_sequences, question_list, user_list, question_to_index, user_to_index
