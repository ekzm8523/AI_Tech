import torch
import argparse
from kobert.pytorch_kobert import get_pytorch_kobert_model
from sklearn.model_selection import train_test_split
from dataset import *
from model import *
from loss import *
from transformers import AdamW
from transformers import ElectraModel, ElectraTokenizer
import random
import gluonnlp as nlp
from kobert.utils import get_tokenizer
from transformers.optimization import get_cosine_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import glob
import re
from sklearn.model_selection import StratifiedKFold
import wandb

def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc


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


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

import os

def train():
	# 1. Start a new run
	wandb.init(project='Pstage2', entity='ekzm8523')
	
	# 2. Save model inputs and hyperparameters
	config = wandb.config
	config.update(args)
	
	###################################
	# 한번만 실행하면 되는 데이터 처리
	# dataset_path = r"/opt/ml/input/data/train/train.tsv"
	# dataset = load_data(dataset_path)
	# dataset['sentence'] = dataset['entity_01'] + ' [SEP] ' + dataset['entity_02'] + ' [SEP] ' + dataset['sentence']
	# dataset[['sentence','label']].to_csv("/opt/ml/input/data/train/train_train.txt", sep='\t', index=False)
	
	# dataset_train = pd.read_csv("/opt/ml/input/data/train/train_train.txt", delimiter='\t')
	# train_sentence = dataset_train['sentence'].tolist()
	# train_labels = dataset_train['label'].tolist()
	# skf = StratifiedKFold(n_splits=5, shuffle=True)
	#
	# for fold, (train_idx, val_idx) in enumerate(skf.split(dataset_train, dataset_train.label)):
	# 	train_dataset = [[train_sentence[i], train_labels[i]] for i in train_idx]
	# 	val_dataset = [[train_sentence[i], train_labels[i]] for i in val_idx]
	# 	train_dataset = pd.DataFrame(train_dataset)
	# 	train_dataset[[0, 1]].to_csv(f"/opt/ml/input/data/train/train_data{fold}.txt", sep='\t', index=False)
	# 	val_dataset = pd.DataFrame(val_dataset)
	# 	val_dataset[[0, 1]].to_csv(f"/opt/ml/input/data/train/val_data{fold}.txt", sep='\t', index=False)
	######################################
	
	seed_everything(args.seed)
	
	################
	max_len = args.max_len
	batch_size = args.batch_size
	warmup_ratio = args.warmup_ratio
	epochs = args.epochs
	max_grad_norm = args.max_grad_norm
	logging_steps = args.logging_steps
	learning_rate = args.lr
	device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
	################
	
	
	for fold in range(5):
		train_dataset = nlp.data.TSVDataset(f"/opt/ml/input/data/train/train_data{fold}.txt", field_indices=[0, 1],
											num_discard_samples=1)
		val_dataset = nlp.data.TSVDataset(f"/opt/ml/input/data/train/val_data{fold}.txt", field_indices=[0, 1],
										  num_discard_samples=1)
		
		bertmodel, vocab = get_pytorch_kobert_model()
		tokenizer = get_tokenizer()
		tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
		model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)
		
		no_decay = ['bias', 'LayerNorm.weight']
		optimizer_grouped_parameters = [
			{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
			 'weight_decay': 0.01},
			{'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
		]
		optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
		
		train_dataset = BERTDataset(train_dataset, 0, 1, tok, max_len, True, False)
		val_dataset = BERTDataset(val_dataset, 0, 1, tok, max_len, True, False)
		
		train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=5)
		val_iter = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=5)
		
		loss_fn = LabelSmoothingLoss()
		
		t_total = len(train_iter) * epochs
		warmup_step = int(t_total * warmup_ratio)
		print(f"t_total : {t_total}\t warmup_step : {warmup_step}")
		
		scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)
		wandb.watch(model)
		
		for e in range(epochs):
			train_acc_sum = 0.0
			test_acc = 0.0
			best_acc = 0.0
			model.train()
			for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(train_iter):
				optimizer.zero_grad()
				token_ids = token_ids.long().to(device)
				segment_ids = segment_ids.long().to(device)
				valid_length = valid_length
				label = label.long().to(device)
				out = model(token_ids, valid_length, segment_ids)
				loss = loss_fn(out, label)
				loss.backward()
				torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
				optimizer.step()
				scheduler.step()
				train_acc_sum += calc_accuracy(out, label)
				
				if batch_id % logging_steps == 0:
					train_loss = loss.data.cpu().numpy()
					print("fold {} epoch {} batch id {} loss {} train acc {}".format(fold, e + 1, batch_id + 1,
																					 train_loss,
																					 train_acc_sum / (batch_id + 1)))
					wandb.log({f"train_loss": train_loss, "train_accuracy": train_acc_sum / (batch_id + 1)})
			
			print("epoch {} train acc {}".format(e + 1, train_acc_sum / (batch_id + 1)))
			model.eval()
			for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(val_iter):
				token_ids = token_ids.long().to(device)
				segment_ids = segment_ids.long().to(device)
				valid_length = valid_length
				label = label.long().to(device)
				out = model(token_ids, valid_length, segment_ids)
				test_acc += calc_accuracy(out, label)
			print("epoch {} test acc {}".format(e + 1, test_acc / (batch_id + 1)))
			if test_acc >= best_acc:
				best_acc = test_acc
				torch.save(model.state_dict(), f"/opt/ml/model/model_state_dict{fold}.pt")
			wandb.log({"val accuracy": test_acc/(len(val_dataset)/batch_size)})
if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()
	# parser.add_argument('--model', type=str, default="Bert", help="transform model choice (default : Bert) ")
	# parser.add_argument('--pretrained_model', type=str, default="bert-base-multilingual-cased", help='Which pretrained model will you bring? (default : bert-base-multilingual-cased)')
	parser.add_argument('--seed', type=int, default=42)
	parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 1)')
	parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 64)')
	parser.add_argument('--lr', type=float, default=5e-5, help='learning rate (default: 5e-5)')
	parser.add_argument('--warmup_ratio', type=float, default=0.01, help='weight ratio (default: 0.01)')
	parser.add_argument('--warmup_steps', type=int, default=500, help='number of warmup steps for learning rate scheduler (default : 500)')
	parser.add_argument('--max_len', type=int, default=128)
	parser.add_argument('--max_grad_norm', type=int, default=1)
	
	parser.add_argument('--logging_steps', type=int, default=100, help='logging steps (default : 100)')
	parser.add_argument('--logging_dir', type=str, default='./logs', help='directory for storing logs (default : ./logs)')
	parser.add_argument('--output_dir', type=str, default='./results', help='save checkpoint (default : ./results/expr)')
	parser.add_argument('--train_dir', type=str, default="/opt/ml/input/data/train")
	parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))
	
	
	args = parser.parse_args()
	print(args)
	print(vars(args))
	
	train()
