from transformers import AutoTokenizer
from data import KLUEDataset

if __name__ == '__main__':
    task = "sts"
    pretrain_model_name = "klue/bert-base"
    batch_size = 64

    tokenizer = AutoTokenizer.from_pretrained(pretrain_model_name, use_fast=True)
    dataset = KLUEDataset(task)
    print()
    dataset.get_dataset().map()