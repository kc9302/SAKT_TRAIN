from typing import Type
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch
from torch.utils.data import DataLoader, random_split

# 사용자 함수 불러오기
from dataloader.dataloader_aihub import AIHUB
from common.utils import collate_fn
from model.sakt import LITNING_SAKT
from __init__ import config

# torch 설정
torch._dynamo.config.suppress_errors = True
torch.set_default_dtype(torch.float64)
torch.set_float32_matmul_precision('high')


class RunModel:

    # Operation flow sequence 3-1.
    def __init__(
            self,
            model_name=Type[str],
            dataset_name=Type[str],
            date_info=None
    ) -> None:

        self.model = None
        self.dataset_name = dataset_name
        self.date_info = date_info
        self.model_name = str(model_name)

        if dataset_name == "AI_HUB":
            config.set("train_config", "batch_size", "8")
            config.set("train_config", "number_epochs", "1")
            config.set("train_config", "train_ratio", "0.8")
            config.set("train_config", "learning_rate", "0.001")
            config.set("train_config", "optimizer", "adam")
            config.set("train_config", "sequence_length", "50")
            config.set("sakt", "n", "50")
            config.set("sakt", "d", "50")
            config.set("sakt", "number_attention_heads", "5")
            config.set("sakt", "dropout", "0.5")

        self.model_config = dict(config.items(self.model_name))
        self.train_config = dict(config.items("train_config"))
        self.batch_size = int(self.train_config["batch_size"])
        self.number_epochs = int(self.train_config["number_epochs"])
        self.train_ratio = float(self.train_config["train_ratio"])
        self.validate_rate = float(1 - self.train_ratio)
        self.learning_rate = float(self.train_config["learning_rate"])
        self.optimizer = self.train_config["optimizer"]
        self.sequence_length = int(self.train_config["sequence_length"])
        self.n = int(self.model_config["n"])
        self.d = int(self.model_config["d"])
        self.number_attention_heads = int(self.model_config["number_attention_heads"])
        self.dropout = float(self.model_config["dropout"])

        # 데이터 불러오기
        self.dataset = AIHUB(sequence_length=self.sequence_length)
        self.model_config["number_questions"] = str(self.dataset.number_question)

        # 총 데이터 수
        dataset_size = len(self.dataset)

        # 훈련 데이터 수
        train_size = int(dataset_size * self.train_ratio)

        # 검증 데이터 수
        validation_size = int(dataset_size * self.validate_rate)

        # 데스트 데이터 수 (일반화 성능 측정)
        test_size = dataset_size - train_size - validation_size

        # Dataset setting.
        train_dataset, validation_dataset, test_dataset = random_split(
            self.dataset,
            [train_size, validation_size, test_size],
            generator=torch.Generator(device=torch.get_default_device()))

        self.train_loader = DataLoader(train_dataset, collate_fn=collate_fn)

        self.validation_loader = DataLoader(validation_dataset, collate_fn=collate_fn)

        self.test_loader = DataLoader(test_dataset, collate_fn=collate_fn)

    def run_model(self):
        # model
        sakt_model = LITNING_SAKT(
            number_questions=int(self.dataset.number_question),
            n=int(self.n),
            d=int(self.d),
            number_attention_heads=int(self.number_attention_heads),
            dropout=float(self.dropout)
        )

        trainer = ""

        # pytorch_lightning
        trainer = Trainer(accelerator="auto",
                          devices="auto",
                          strategy="auto",
                          max_epochs=self.number_epochs,
                          callbacks=[EarlyStopping(monitor="val_loss", patience=3, mode="min")])

        # 얼리스탑 사용 안할 시
        # trainer = Trainer(strategy='auto',
        #                   max_epochs=int)

        # 훈련
        trainer.fit(sakt_model,
                    self.train_loader,
                    self.validation_loader)
        # 평가
        trainer.test(sakt_model, self.test_loader)

        # 모델 저장
        trainer.save_checkpoint("example.pth")
