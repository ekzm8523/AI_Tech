from train import load_data
from dataset import *
import gluonnlp as nlp
import torch
from kobert.pytorch_kobert import get_pytorch_kobert_model
from kobert.utils import get_tokenizer

from model import *

def load_model_tokenizer(model_path, device):

	
	model, vocab = get_pytorch_kobert_model()
	model = BERTClassifier(model, dr_rate=0.5).to(device)
	model.load_state_dict(torch.load(model_path))
	tokenizer = get_tokenizer()
	tokenizer = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
	print("load model and tokenizer compete!!")
	return model, tokenizer


def inference():
	device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
	dataset_path = r"/opt/ml/input/data/test/test.tsv"
	dataset = load_data(dataset_path)
	dataset['sentence'] = dataset['entity_01'] + ' [SEP] ' + dataset['entity_02'] + ' [SEP] ' + dataset['sentence']
	dataset[['sentence', 'label']].to_csv("/opt/ml/input/data/test/test.txt", sep='\t', index=False)
	dataset_test = nlp.data.TSVDataset("/opt/ml/input/data/test/test.txt", field_indices=[0, 1], num_discard_samples=1)

	model, tokenizer = load_model_tokenizer("/opt/ml/model/model_state_dict.pt", device)
	###########
	max_len = 128
	batch_size = 32
	###########
	print("check1")
	data_test = BERTDataset(dataset_test, 0, 1, tokenizer, max_len, True, False)
	print("check2")
	test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=2)
	print("check3")
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
	print("check4")

if __name__ == '__main__':
	inference()
