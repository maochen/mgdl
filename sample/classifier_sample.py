import logging
import os
import sys
import tempfile
import time

import numpy as np
import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset

from core.abstract_classifier import AbstractClassifier


class SampleModel(nn.Module):
    def __init__(self, input_size: int, num_classes: int):
        super(SampleModel, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        y = nn.functional.relu(self.fc(x))
        return y


class SampleClassifier(AbstractClassifier):

    def __init__(self, input_size: int):
        super(SampleClassifier, self).__init__()
        self.input_size = input_size

    def get_model(self, num_classes: int, **kwargs):
        self.model = SampleModel(self.input_size, num_classes)
        self.criterion = nn.CrossEntropyLoss()

        lr = 1.0e-4
        self.optimizer = SGD(self.model.parameters(), lr=lr)
        return self.model, self.criterion, self.optimizer

    def load_data(self) -> (DataLoader, {}):
        label2idx = {"yes": 1, "no": 0}

        input_size = self.input_size

        class SampleDataset(Dataset):
            def __init__(self):
                self.count = 0

            def __len__(self):
                return 1000

            def __getitem__(self, idx):
                start = min(label2idx.values())
                end = max(label2idx.values()) + 1
                y = np.random.randint(start, end)

                self.count += 1
                return torch.rand(input_size), y, f"id_{self.count}"

        ds = SampleDataset()
        return DataLoader(ds, batch_size=1000), label2idx


if __name__ == "__main__":
    # INIT LOGGER
    root_logger = logging.root
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(logging.StreamHandler(sys.stdout))

    cf = SampleClassifier(input_size=1000)
    cf.get_model(2)
    dl, label2idx = cf.load_data()
    cf.label2idx = label2idx

    temp_dir = os.path.join(tempfile.gettempdir(), "cf_test_" + str(time.strftime("%Y%m%d%H%M%S", time.gmtime())))
    print(f"Generate output to {temp_dir}")

    cf.fit(dl, val_dl=dl, epochs=3, saved_model_dir=temp_dir, model_id="1", tensorboard_dir=temp_dir)
    pred, metadata = cf.predict(dl)
    print(pred)
    print(metadata)
