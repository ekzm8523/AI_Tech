import pickle as pickle
import os
import pandas as pd
import torch
# from pororo import Pororo
# Dataset 구성.
class RE_Dataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_dataset, labels):
        self.tokenized_dataset = tokenized_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.tokenized_dataset.items()}
        item['labels'] = torch.tensor(self.labels[idx])

        return item # input_ids, token_type_ids, attention_mask, label return

    def __len__(self):
        return len(self.labels)

# 처음 불러온 tsv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다.
# 변경한 DataFrame 형태는 baseline code description 이미지를 참고해주세요.
def preprocessing_dataset(dataset, label_type):
    label = []
    for i in dataset[8]:
        if i == 'blind':
            label.append(100)
        else:
            label.append(label_type[i])
    out_dataset = pd.DataFrame({'sentence':dataset[1],'entity_01':dataset[2],'entity_02':dataset[5],'label':label,})
    return out_dataset
    
# tsv 파일을 불러옵니다.
def load_data(dataset_dir):
    with open('/opt/ml/input/data/label_type.pkl', 'rb') as f:
        label_type = pickle.load(f)
    dataset = pd.read_csv(dataset_dir, delimiter='\t', header=None)
    dataset = preprocessing_dataset(dataset, label_type)
    return dataset

def binary_load_data(dataset_dir):
    with open('/opt/ml/input/data/label_type.pkl', 'rb') as f:
        label_type = pickle.load(f)
    dataset = pd.read_csv(dataset_dir, delimiter='\t', header=None)
    dataset = preprocessing_dataset(dataset, label_type)
    dataset['label'] = dataset['label'] > 0
    return dataset

# bert input을 위한 tokenizing.
# tip! 다양한 종류의 tokenizer와 special token들을 활용하는 것으로도 새로운 시도를 해볼 수 있습니다.
# baseline code에서는 2가지 부분을 활용했습니다.
def tokenized_dataset(dataset, tokenizer):
    concat_entity = []
    for e01, e02 in zip(dataset['entity_01'], dataset['entity_02']):
        temp = ''
        temp = e01 + '</s></s>' + e02
        concat_entity.append(temp)
    tokenized_sentences = tokenizer(
        concat_entity,
        list(dataset['sentence']),
        return_tensors="pt",
        padding=True,
        truncation='longest_first',
        max_length=128,
        add_special_tokens=True,
        
        )
    return tokenized_sentences


# def pororo_tokenized_dataset(dataset, tokenizer):
#     concat_entity = []
#     ner = Pororo(task="ner", lang="ko")
#     for sent, e1, e2, start1, end1, start2, end2,  in zip(dataset['sentence'], dataset['entity_01'], dataset['entity_02'], dataset['entity_01_start'],
#                                                           dataset['entity_01_end'], dataset['entity_02_start'], dataset['entity_02_end']):
#         ner_01 = ' α ' + ner(e1)[0][1].lower() + ' α '
#         ner_02 = ' β ' + ner(e2)[0][1].lower() + ' β '
#
#         start1, end1 = int(start1), int(end1)
#         start2, end2 = int(start2), int(end2)
#
#         if start1 < start2:
#             sent = sent[:start1] + '@' + ner_01 + sent[start1:end1+1] + ' @ ' + sent[end1+1:start2] + \
#                 '#' + ner_02 + sent[start2:end2+1] + ' # ' + sent[end2+1:]
#         else:
#             sent = sent[:start2] + '#' + ner_02 + sent[start2:end2+1] + ' # ' + sent[end2+1:start1] + \
#                 '@' + ner_01 + sent[start1:end1+1] + ' @ ' + sent[end1:]
#
#         concat_entity.append(sent)
#     tokenized_sentences = tokenizer(
#         concat_entity,
#         return_tensors="pt",
#         padding=True,
#         truncation='longest_first',
#         max_length=128,
#         add_special_tokens=True,
#
#     )
#     return tokenized_sentences
