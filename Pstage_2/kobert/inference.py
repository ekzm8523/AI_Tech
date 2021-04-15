# from train import load_data
# from dataset import *
# import gluonnlp as nlp
# import torch
#
# def inference():
# 	dataset_path = r"/opt/ml/input/data/test/test.tsv"
#
# 	dataset = load_data(dataset_path)
#
# 	dataset['sentence'] = dataset['entity_01'] + ' [SEP] ' + dataset['entity_02'] + ' [SEP] ' + dataset['sentence']
#
# 	dataset[['sentence', 'label']].to_csv("/opt/ml/input/data/test/test.txt", sep='\t', index=False)
#
# 	dataset_test = nlp.data.TSVDataset("/opt/ml/input/data/test/test.txt", field_indices=[0, 1], num_discard_samples=1)
#
# 	data_test = BERTDataset(dataset_test, 0, 1, tok, max_len, True, False)
#
# 	test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=5)
#
# 	model.load_state_dict(torch.load("/opt/ml/model/model_state_dict.pt"))
#
# 	model.eval()
#
# 	Predict = []
#
# 	for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
# 		token_ids = token_ids.long().to(device)
# 		segment_ids = segment_ids.long().to(device)
# 		valid_length = valid_length
# 		label = label.long().to(device)
# 		out = model(token_ids, valid_length, segment_ids)
# 		_, predict = torch.max(out, 1)
# 		Predict.extend(predict.tolist())
#
# 	output = pd.DataFrame(Predict, columns=['pred'])
# 	output.to_csv('/opt/ml/result/submission.csv', index=False)
