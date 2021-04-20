from train import load_data
from dataset import *
import gluonnlp as nlp
import torch
from kobert.pytorch_kobert import get_pytorch_kobert_model
from kobert.utils import get_tokenizer
import argparse
from model import *

def hard_voting(Predict):
    count = 0
    mode = 0
    for x in Predict:
        if Predict.count(x) > count:
            count = Predict.count(x)
            mode = x

    return mode

def inference(args):
	##############################
	### 이것 또한 처음만 실행
	# dataset_path = r"/opt/ml/input/data/test/test.tsv"
	# dataset = load_data(dataset_path)
	# dataset['sentence'] = dataset['entity_01'] + ' [SEP] ' + dataset['entity_02'] + ' [SEP] ' + dataset['sentence']
	# dataset[['sentence', 'label']].to_csv("/opt/ml/input/data/test/test.txt", sep='\t', index=False)
	###############################
	
	###########
	max_len = args.max_len
	batch_size = args.batch_size
	device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
	###########
	
	model, vocab = get_pytorch_kobert_model()
	tokenizer = get_tokenizer()
	tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
	dataset_test = nlp.data.TSVDataset("/opt/ml/input/data/test/test.txt", field_indices=[0, 1], num_discard_samples=1)
	data_test = BERTDataset(dataset_test, 0, 1, tok, max_len, True, False)
	test_iter = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=5)
	
	Predict = [[] for i in range(5)]
	
	for fold in range(5):
		model, vocab = get_pytorch_kobert_model()
		model = BERTClassifier(model).to(device)
		model.load_state_dict(torch.load(f"/opt/ml/model/model_state_dict{fold}.pt"))
		model.eval()
		
		for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_iter):
			token_ids = token_ids.long().to(device)
			segment_ids = segment_ids.long().to(device)
			valid_length = valid_length
			label = label.long().to(device)
			out = model(token_ids, valid_length, segment_ids)
			Predict[fold].extend(out.cpu().detach().numpy())
	
	if args.predict_strategy == "soft":
		soft_voting_predict = np.array(Predict[0]) + np.array(Predict[1]) \
							  +np.array(Predict[2]) + np.array(Predict[3]) + np.array(Predict[4])
		voting_predict = list(np.argmax(soft_voting_predict, axis=1))
	else: # hard
		voting_predict = []
		for i in range(len(Predict[0])):
			voting_predict.append(hard_voting(
				[np.argmax(Predict[0][i]), np.argmax(Predict[1][i]),
				 np.argmax(Predict[2][i]), np.argmax(Predict[3][i]),
				 np.argmax(Predict[4][i])]))
	
	output = pd.DataFrame(voting_predict, columns=['pred'])
	output.to_csv('/opt/ml/result/submission.csv', index=False)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--predict_strategy', type=str, default="soft")
	parser.add_argument('--max_len', type=int, default=128)
	parser.add_argument('--batch_size', type=int, default=64)
	
	args = parser.parse_args()
	inference(args)
