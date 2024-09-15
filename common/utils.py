import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.optim import SGD, Adam
import datetime
import logging
import os

if torch.cuda.is_available():
    from torch.cuda import FloatTensor
    torch.torch.set_default_dtype(torch.float64)
else:
    from torch import FloatTensor


def match_sequence_length(
        user_list: list, 
        question_sequences: list,
        response_sequences: list,
        sequence_length: int,
        padding_value=-1
):
    """Function that matches the length of question_sequences and response_sequences to the length of sequence_length.

    Args:
        question_sequences: A list of question solutions for each student.
        response_sequences: A list of questions and answers for each student.
        sequence_length: Length of sequence.
        padding_value: Value of padding.

    Returns:
        length-matched parameters.

    Note:
        Return detail.

        - proc_question_sequences : length-matched question_sequences.
        - proc_response_sequences : length-matched response_sequences.

    Examples:
        >>> match_sequence_length(question_sequences=[[67 18 67 18 67 18 18 67 67 18 18 35 18 32 18]...],
        >>>                       response_sequence=[[0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0]...],
        >>>                       sequence_length=50,
        >>>                       padding_value=-1)
        ([[67 18 67 18 67 18 18 67 67 18 18 35 18 32 18 -1 -1 -1 ... -1 -1 -1]...],
        [[0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 -1 -1 -1 ... -1 -1 -1]...])
    """
    proc_question_sequences = []
    proc_response_sequences = []
    proc_user_list = []
    
    for user, question_sequence, response_sequence in zip(user_list, question_sequences, response_sequences):

        i = 0

        while i + sequence_length + 1 < len(question_sequence):
            proc_user_list.append(user)
            proc_question_sequences.append(question_sequence[i:i + sequence_length + 1])
            proc_response_sequences.append(response_sequence[i:i + sequence_length + 1])
            i += sequence_length + 1
        
        proc_user_list.append(user)
        proc_question_sequences.append(
            np.concatenate(
                [
                    question_sequence[i:],
                    np.array([padding_value] * (i + sequence_length + 1 - len(question_sequence)))
                ]
            )
        )

        proc_response_sequences.append(
            np.concatenate(
                [
                    response_sequence[i:],
                    np.array([padding_value] * (i + sequence_length + 1 - len(question_sequence)))
                ]
            )
        )

    return proc_user_list, proc_question_sequences, proc_response_sequences


def collate_fn(
        batch,
        padding_value=-1
):
    """The collate function for torch.utils.data.DataLoader

    Args:
        batch: data batch.
        padding_value: Value of padding.

    Returns:
        Dataloader elements for model training.

    Note:
        Return detail.

        - question_sequences: the question(KC) sequences.
            - question_sequences_shape: [batch_size, maximum_sequence_length_in_the_batch].

        - response_sequences: the response sequences.
            - response_sequences_shape: [batch_size, maximum_sequence_length_in_the_batch].

        - question_shift_sequences: the question(KC) sequences which were shifted one step to the right.
            - question_shift_sequences_shape: [batch_size, maximum_sequence_length_in_the_batch].

        - response_shift_sequences: the response sequences which were shifted one step to the right.
            - response_shift_sequences_shape: [batch_size, maximum_sequence_length_in_the_batch].

        - mask_sequences: the mask sequences indicating where the padded entry.
            - mask_sequences_shape: [batch_size, maximum_sequence_length_in_the_batch].
    """
    question_sequences = []
    response_sequences = []
    question_shift_sequences = []
    response_shift_sequences = []

    for q_seq, r_seq in batch:
        q_seq = list(q_seq)
        r_seq = list(r_seq)
        question_sequences.append(torch.tensor(q_seq[:-1], dtype=torch.float64, device='cuda'))
        response_sequences.append(torch.tensor(r_seq[:-1], dtype=torch.float64, device='cuda'))
        question_shift_sequences.append(torch.tensor(q_seq[1:], dtype=torch.float64, device='cuda'))
        response_shift_sequences.append(torch.tensor(r_seq[1:], dtype=torch.float64, device='cuda'))

    question_sequences = pad_sequence(
        question_sequences, batch_first=True, padding_value=padding_value
    )

    response_sequences = pad_sequence(
        response_sequences, batch_first=True, padding_value=padding_value
    )
    question_shift_sequences = pad_sequence(
        question_shift_sequences, batch_first=True, padding_value=padding_value
    )
    response_shift_sequences = pad_sequence(
        response_shift_sequences, batch_first=True, padding_value=padding_value
    )

    mask_sequences = (question_sequences != padding_value) * (question_shift_sequences != padding_value)

    question_sequences, response_sequences, question_shift_sequences, response_shift_sequences = \
        question_sequences * mask_sequences, response_sequences * mask_sequences, question_shift_sequences * mask_sequences, \
        response_shift_sequences * mask_sequences

    return question_sequences, response_sequences, question_shift_sequences, response_shift_sequences, mask_sequences


def collate_fn_cpu(
        batch,
        padding_value=-1
):
    """The collate function for torch.utils.data.DataLoader

    Args:
        batch: data batch.
        padding_value: Value of padding.

    Returns:
        Dataloader elements for model training.

    Note:
        Return detail.

        - question_sequences: the question(KC) sequences.
            - question_sequences_shape: [batch_size, maximum_sequence_length_in_the_batch].

        - response_sequences: the response sequences.
            - response_sequences_shape: [batch_size, maximum_sequence_length_in_the_batch].

        - question_shift_sequences: the question(KC) sequences which were shifted one step to the right.
            - question_shift_sequences_shape: [batch_size, maximum_sequence_length_in_the_batch].

        - response_shift_sequences: the response sequences which were shifted one step to the right.
            - response_shift_sequences_shape: [batch_size, maximum_sequence_length_in_the_batch].

        - mask_sequences: the mask sequences indicating where the padded entry.
            - mask_sequences_shape: [batch_size, maximum_sequence_length_in_the_batch].
    """
    question_sequences = []
    response_sequences = []
    question_shift_sequences = []
    response_shift_sequences = []

    for q_seq, r_seq in batch:
        question_sequences.append(FloatTensor(q_seq[:-1]))
        response_sequences.append(FloatTensor(r_seq[:-1]))
        question_shift_sequences.append(FloatTensor(q_seq[1:]))
        response_shift_sequences.append(FloatTensor(r_seq[1:]))

    question_sequences = pad_sequence(
        question_sequences, batch_first=True, padding_value=padding_value
    )

    response_sequences = pad_sequence(
        response_sequences, batch_first=True, padding_value=padding_value
    )
    question_shift_sequences = pad_sequence(
        question_shift_sequences, batch_first=True, padding_value=padding_value
    )
    response_shift_sequences = pad_sequence(
        response_shift_sequences, batch_first=True, padding_value=padding_value
    )

    mask_sequences = (question_sequences != padding_value) * (question_shift_sequences != padding_value)

    question_sequences, response_sequences, question_shift_sequences, response_shift_sequences = \
        question_sequences * mask_sequences, response_sequences * mask_sequences, question_shift_sequences * mask_sequences, \
        response_shift_sequences * mask_sequences

    return question_sequences, response_sequences, question_shift_sequences, response_shift_sequences, mask_sequences


def set_optimizer(
        optimizer: str,
        model_parameters=torch.nn.Parameter,
        learning_rate=float
):
    """A function that creates an optimizer.

    Args:
        optimizer: Optimization function you want to set.
        model_parameters: Parameters of the model.
        learning_rate: learning rate.

    Returns:
        Optimization function.

    Note:
        Return detail.

        - Optimization_function : Optimization function.

    Examples:
        >>> set_optimizer(optimizer="adam",
        >>>               model_parameters=model_parameters,
        >>>               learning_rate=float)
        torch.optim.adam.Adam
    """
    if optimizer == "sgd":
        optimization_function = SGD(model_parameters, float(learning_rate), momentum=0.9)
        return optimization_function
    elif optimizer == "adam":
        optimization_function = Adam(model_parameters, float(learning_rate))
        return optimization_function


def set_logging(model_name: str,
                dataset_name: str):
    """set lgging.

    Args:
        model_name: The model name.
        dataset_name: The dataset name.

    Returns:
        logging and date time.
    """
    if not os.path.isdir("./logs"):
        # logs 폴더 생성
        os.mkdir('logs')

    # 현재 시간을 저장
    date_info = datetime.datetime.now()

    # 파일 이름
    log_file_name = "./logs/" + model_name + "_" + dataset_name + "_log_{}.log".format(
        date_info.today().strftime("%y%m%d_%H_%M_%S")
    )

    logging.basicConfig(filename=log_file_name,
                        datefmt="%Y-%m-%d %H-%M-%S",
                        level=logging.DEBUG,
                        format="%(asctime)s %(levelname)s %(filename)s :%(message)s")
    logging.basicConfig(filename=log_file_name,
                        datefmt="%Y-%m-%d %H-%M-%S",
                        level=logging.ERROR,
                        format="%(asctime)s %(levelname)s %(filename)s :%(message)s")
    logging.getLogger().setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(filename)s :%(message)s")
    )
    logging.getLogger().addHandler(console_handler)
    return logging, date_info
