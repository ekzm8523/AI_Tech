import datasets
from datasets import load_dataset, DatasetDict
from typing import Optional, List, Dict
import random
import pandas as pd


class KLUEDataset:
    dataset: Optional[DatasetDict]

    def __init__(self, task: str):
        dataset = load_dataset('klue', task)
        dataset = dataset.flatten()  # KLUE는 label이 labels로 되어있어 real-label, binary-label, label로 이루어져있음
        dataset = dataset.rename_column('labels.real-label', 'label')
        self.dataset = dataset.remove_columns(['labels.label', 'labels.binary-label'])

    def get_dataset(self) -> DatasetDict:
        return self.dataset

    def get_random_samples(self, count: int = 5) -> List[Dict]:
        return random.choices(self.dataset['train'], k=count)  # noqa
    
