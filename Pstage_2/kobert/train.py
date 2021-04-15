import torch
import argparse
from kobert.pytorch_kobert import get_pytorch_kobert_model
from sklearn.model_selection import train_test_split
from dataset import *
from model import *
from loss import *
from transformers import AdamW
import random
import gluonnlp as nlp
from kobert.utils import get_tokenizer
from transformers.optimization import get_cosine_schedule_with_warmup
def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def train():
	
	seed_everything(42)
	device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
	bertmodel, vocab = get_pytorch_kobert_model()
	
	dataset_path = r"/opt/ml/input/data/train/train.tsv"
	
	dataset = load_data(dataset_path)
	
	dataset['sentence'] = dataset['entity_01'] + ' [SEP] ' + dataset['entity_02'] + ' [SEP] ' + dataset['sentence']
	print(dataset.head())
	
	train, val = train_test_split(dataset, test_size=0.2, random_state=42)
	print(len(train))
	print(len(val))
	
	train[['sentence', 'label']].to_csv("/opt/ml/input/data/train/train_train.txt", sep='\t', index=False)
	val[['sentence', 'label']].to_csv("/opt/ml/input/data/train/train_val.txt", sep='\t', index=False)

	dataset_train = nlp.data.TSVDataset("/opt/ml/input/data/train/train_train.txt", field_indices=[0,1], num_discard_samples=1)
	dataset_val = nlp.data.TSVDataset("/opt/ml/input/data/train/train_val.txt", field_indices=[0,1], num_discard_samples=1)

	print(dataset_train[0])
	
	tokenizer = get_tokenizer()
	tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
	
	################
	max_len = 128
	batch_size = 32
	warmup_ratio = 0.01
	num_epochs = 20
	max_grad_norm = 1
	log_interval = 50
	learning_rate = 5e-5
	################
	
	data_train = BERTDataset(dataset_train, 0, 1, tok, max_len, True, False)
	data_val = BERTDataset(dataset_val, 0, 1, tok, max_len, True, False)
	
	train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=5)
	val_dataloader = torch.utils.data.DataLoader(data_val, batch_size=batch_size, num_workers=5)
	
	model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)
	no_decay = ['bias', 'LayerNorm.weight']
	optimizer_grouped_parameters = [
		{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
		 'weight_decay': 0.01},
		{'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
	]
	
	optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
	loss_fn = LabelSmoothingLoss()
	
	t_total = len(train_dataloader) * num_epochs
	warmup_step = int(t_total * warmup_ratio)
	
	scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)
	
	for e in range(num_epochs):
		train_acc = 0.0
		test_acc = 0.0
		best_acc = 0.0
		model.train()
		for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(train_dataloader):
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
			train_acc += calc_accuracy(out, label)
			if batch_id % log_interval == 0:
				print("epoch {} batch id {} loss {} train acc {}".format(e + 1, batch_id + 1, loss.data.cpu().numpy(),
																		 train_acc / (batch_id + 1)))
		print("epoch {} train acc {}".format(e + 1, train_acc / (batch_id + 1)))
		model.eval()
		for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(val_dataloader):
			token_ids = token_ids.long().to(device)
			segment_ids = segment_ids.long().to(device)
			valid_length = valid_length
			label = label.long().to(device)
			out = model(token_ids, valid_length, segment_ids)
			test_acc += calc_accuracy(out, label)
		print("epoch {} test acc {}".format(e + 1, test_acc / (batch_id + 1)))
		if test_acc >= best_acc:
			best_acc = test_acc
			torch.save(model.state_dict(), "/opt/ml/model/model_state_dict.pt")
	
	dataset_path = r"/opt/ml/input/data/test/test.tsv"
	
	dataset = load_data(dataset_path)
	
	dataset['sentence'] = dataset['entity_01'] + ' [SEP] ' + dataset['entity_02'] + ' [SEP] ' + dataset['sentence']
	
	dataset[['sentence', 'label']].to_csv("/opt/ml/input/data/test/test.txt", sep='\t', index=False)
	
	dataset_test = nlp.data.TSVDataset("/opt/ml/input/data/test/test.txt", field_indices=[0, 1], num_discard_samples=1)
	
	data_test = BERTDataset(dataset_test, 0, 1, tok, max_len, True, False)
	
	test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=5)
	model.load_state_dict(torch.load("/opt/ml/model/model_state_dict.pt"))
	
	model.eval()
	
	Predict = []
	
	for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
		token_ids = token_ids.long().to(device)
		segment_ids = segment_ids.long().to(device)
		valid_length = valid_length
		label = label.long().to(device)
		out = model(token_ids, valid_length, segment_ids)
		_, predict = torch.max(out, 1)
		Predict.extend(predict.tolist())
	
	output = pd.DataFrame(Predict, columns=['pred'])
	output.to_csv('/opt/ml/result/submission.csv', index=False)
	
if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', type=str, default="Bert", help="transform model choice (default : Bert) ")
	parser.add_argument('--pretrained_model', type=str, default="bert-base-multilingual-cased", help='Which pretrained model will you bring? (default : bert-base-multilingual-cased)')

	parser.add_argument('--epochs', type=int, default=4, help='number of epochs to train (default: 1)')
	parser.add_argument('--batch_size', type=int, default=16, help='input batch size for training (default: 64)')
	parser.add_argument('--lr', type=float, default=5e-5, help='learning rate (default: 5e-5)')
	parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay (default: 0.01)')
	parser.add_argument('--warmup_steps', type=int, default=500, help='number of warmup steps for learning rate scheduler (default : 500)')
	parser.add_argument('--save_steps', type=int, default=500, help='save steps (default : 500) ')
	parser.add_argument('--save_total_limit', type=int, default=3, help='save total limit (default : 3)')
	parser.add_argument('--logging_steps', type=int, default=100, help='logging steps (default : 100)')
	parser.add_argument('--logging_dir', type=str, default='./logs', help='directory for storing logs (default : ./logs)')
	parser.add_argument('--output_dir', type=str, default='./results/expr', help='save checkpoint (default : ./results/expr)')
	parser.add_argument('--train_dir', type=str, default="/opt/ml/input/data/train/train.tsv")
	args = parser.parse_args()
	print(args)
	
	train()
