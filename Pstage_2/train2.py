"""
trainer를 사용하지 않은 train.py
"""
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, XLMRobertaConfig, get_cosine_schedule_with_warmup, AdamW
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from load_data import *
import torch
from torch import nn
import time
import re
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback
from load_data import *


def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc


# 평가를 위한 metrics function.
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # calculate accuracy using sklearn's function
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
    }


def train():
	##############
	max_len = 128
	batch_size = 32
	warmup_ratio = 0.01
	num_epochs = 30
	max_grad_norm = 1
	learning_rate = 5e-6
	stop_count = 5
	num_folds = 15
	PATH = './model/model_state_dict_init'
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	##############
	print(device)
	
	df_train = load_data('/opt/ml/input/data/train/train.tsv')
	df_train = df_train[~df_train['label'].isin((40, 37, 29, 41, 19, 18, 26, 28, 39))]
	MODEL_NAME = "xlm-roberta-large"
	tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)
	
	model_config = XLMRobertaConfig.from_pretrained(MODEL_NAME)
	model_config.num_labels = 42
	model = XLMRobertaForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config).to(device)
	
	# 모델 저장 (fold마다 모델을 불러오기 위해)
	torch.save(model.state_dict(), PATH)
	
	# dataset making
	t_dataset = tokenized_dataset(df_train, tokenizer)
	t_label = df_train['label'].values
	dataset = RE_Dataset(t_dataset, t_label)
	# kfold = KFold(n_splits=num_folds, random_state=0, shuffle=True)
	kfold = StratifiedKFold(n_splits=num_folds, random_state=0, shuffle=True)
	criterion = nn.CrossEntropyLoss()
	# criterion = LabelSmoothingLoss()
	results = {}
	
	for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset, dataset.labels)):
		print(f'FOLD {fold}')
		print('=' * 10)
		
		# Sample elements randomly from a given list of ids, no replacement.
		train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
		val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
		
		# Define data loaders for training and testing data in this fold
		train_loader = torch.utils.data.DataLoader(
			dataset,
			batch_size=batch_size,
			sampler=train_subsampler,
		)
		
		val_loader = torch.utils.data.DataLoader(
			dataset,
			batch_size=batch_size,
			sampler=val_subsampler,
		)
		
		# 모델을 불러온다. (huggingface로 계속 불러오면 메모리 초과 발생!)
		model.load_state_dict(torch.load(PATH))
		
		no_decay = ['bias', 'LayerNorm.weight']
		optimizer_grouped_parameters = [
			{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
			 'weight_decay': 0.01},
			{'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
		]
		optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
		
		t_total = len(train_loader) * num_epochs
		warmup_step = int(t_total * warmup_ratio)
		
		scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)
		
		best_acc = 0.0
		early_stopping_count = 0
		
		for epoch in range(num_epochs):
			print(f'Epoch {epoch}/{num_epochs - 1}')
			print('-' * 10)
			
			train_acc = 0.0
			val_acc = 0.0
			
			since = time.time()
			
			#################### Train ####################
			train_loss = 0.0
			
			model.train()
			for batch_id, batch in enumerate(train_loader):
				optimizer.zero_grad()
				input_ids = batch['input_ids'].to(device)
				attention_mask = batch['attention_mask'].to(device)
				labels = batch['labels'].to(device)
				outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
				loss = outputs[0]
				loss.backward()
				torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
				optimizer.step()
				scheduler.step()
				train_acc += calc_accuracy(outputs[1], labels)
				train_loss = loss.data.cpu().numpy()
			
			print(f"train Loss: {train_loss:.4f} Acc: {train_acc / (batch_id + 1):.4f}")
			
			#################### Validation ####################
			val_loss = 0.0
			
			model.eval()
			for batch_id, batch in enumerate(val_loader):
				input_ids = batch['input_ids'].to(device)
				attention_mask = batch['attention_mask'].to(device)
				labels = batch['labels'].to(device)
				outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
				val_acc += calc_accuracy(outputs[1], labels)
				val_loss = outputs[0].data.cpu().numpy()
			
			print(f"val Loss: {val_loss:.4f} Acc: {val_acc / (batch_id + 1):.4f}")
			
			#################### model save ####################
			if (val_acc / (batch_id + 1)) > best_acc:
				early_stopping_count = 0
				print(f"epochs_val acc: {val_acc / (batch_id + 1):.4f}")
				print(f"epochs_before_best acc: {best_acc:.4f}")
				best_acc = (val_acc / (batch_id + 1))
				print(f"epochs_after_best acc: {best_acc:.4f}")
				torch.save(model.state_dict(), f"/opt/ml/model/model_state_dict{fold}.pt")
			
			else:
				early_stopping_count += 1
			
			#################### running time check ####################
			time_elapsed = time.time() - since
			print(f'Training complete in {time_elapsed // 60}m {time_elapsed % 60}s')
			
			#################### Early Stopping ####################
			if early_stopping_count >= stop_count:
				print('...Early Stopping...')
				print()
				break
			
			print()
		
		results[fold] = best_acc
		print(f'Best val Acc: {best_acc}')
		print()

if __name__ == "__main__":
	train()
