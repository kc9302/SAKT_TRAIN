import torch
import configparser
import logging

# Operation flow sequence 1.
try:
    # setting device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # read config.ini
    config = configparser.ConfigParser()
    config.read("./config.ini")
except Exception as err:
    logging.error(err)
