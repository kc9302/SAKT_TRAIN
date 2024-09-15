from train import RunModel
from common.utils import set_logging


def run(dataset_name):
    model_name = "sakt"

    # Operation flow sequence 2.
    _, date_info = set_logging(model_name=model_name, dataset_name=dataset_name)
    # Operation flow sequence 3.
    if dataset_name == "AI_HUB":
        RunModel(model_name=model_name,
                 dataset_name=dataset_name,
                 date_info=date_info).run_model()


if __name__ == "__main__":
    run(dataset_name="AI_HUB")