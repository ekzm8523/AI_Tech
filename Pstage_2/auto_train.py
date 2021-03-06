import pickle as pickle
import os
import pandas as pd
import torch
import re
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback
from load_data import *
import argparse
import glob
from importlib import import_module
from pathlib import Path

# 평가를 위한 metrics function.
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # calculate accuracy using sklearn's function
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
    }
def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"
    
def train(args):
    
    for fold in range(5):
        MODEL_NAME = args.pretrained_model
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
        # load dataset
        train_dataset = load_data(f'/opt/ml/input/data/train/k_fold/train_fold{fold}.tsv')
        dev_dataset = load_data(f'/opt/ml/input/data/train/k_fold/val_fold{fold}.tsv')
        
        train_label = train_dataset['label'].values
        dev_label = dev_dataset['label'].values
        
        # tokenizing dataset
        tokenized_train = tokenized_dataset(train_dataset, tokenizer)
        tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)
    
        # make dataset for pytorch.
        RE_train_dataset = RE_Dataset(tokenized_train, train_label)
        RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)
    
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
        # setting model hyperparameter
        config_module = getattr(import_module("transformers"), args.model + "Config")
        print(args.model + "Config")
        model_config = config_module.from_pretrained(MODEL_NAME)
        model_config.num_labels = 42
        
        model_module = getattr(import_module("transformers"), args.model + "ForSequenceClassification")
        model = model_module.from_pretrained(MODEL_NAME, config=model_config)
        model.to(device)
    
        output_dir = increment_path(args.output_dir + args.model)
        output_dir += f'fold{fold}'
        # 사용한 option 외에도 다양한 option들이 있습니다.
        # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
        training_args = TrainingArguments(
            output_dir=output_dir,          # output directory
            save_total_limit=args.save_total_limit,              # number of total save model.
            save_steps=args.save_steps,                 # model saving step.
            num_train_epochs=args.epochs,              # total number of training epochs
            learning_rate=args.lr,               # learning_rate
            per_device_train_batch_size=args.batch_size,  # batch size per device during training
            per_device_eval_batch_size=args.batch_size,   # batch size for evaluation
            warmup_steps=args.warmup_steps,                # number of warmup steps for learning rate scheduler
            weight_decay=args.weight_decay,               # strength of weight decay
            logging_dir=args.logging_dir,            # directory for storing logs
            logging_steps=args.logging_steps,              # log saving step.
            evaluation_strategy='steps', # evaluation strategy to adopt during training
                                        # `no`: No evaluation during training.
                                        # `steps`: Evaluate every `eval_steps`.
                                        # `epoch`: Evaluate every end of epoch.
            eval_steps = 500,            # evaluation step.
            dataloader_num_workers=4,
            label_smoothing_factor=0.5,
            load_best_model_at_end = True,
            metric_for_best_model="accuracy",
        )
        early_stopping = EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.00005)
        trainer = Trainer(
            model=model,                         # the instantiated 🤗 Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=RE_train_dataset,         # training dataset
            eval_dataset=RE_dev_dataset,             # evaluation dataset
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,         # define metrics function
            callbacks=[early_stopping],
        )
        trainer.train()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="XLMRoberta", help='transform model choice (default : XLMRoberta) ')
    parser.add_argument('--pretrained_model', type=str, default="xlm-roberta-base", help='Which pretrained model will you bring? (default : bert-base-multilingual-cased)')

    parser.add_argument('--epochs', type=int, default=6, help='number of epochs to train (default: 1)')
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate (default: 5e-5)')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay (default: 0.01)')
    parser.add_argument('--warmup_steps', type=int, default=500, help='number of warmup steps for learning rate scheduler (default : 500)')
    parser.add_argument('--save_steps', type=int, default=500, help='save steps (default : 500) ')
    parser.add_argument('--save_total_limit', type=int, default=1, help='save total limit (default : 3)')
    parser.add_argument('--logging_steps', type=int, default=100, help='logging steps (default : 100)')
    parser.add_argument('--logging_dir', type=str, default='./logs', help='directory for storing logs (default : ./logs)')
    parser.add_argument('--output_dir', type=str, default='./results', help='save checkpoint (default : ./results/expr)')
    parser.add_argument('--train_dir', type=str, default="/opt/ml/input/data/train/train.tsv")
    args = parser.parse_args()
    train(args)
