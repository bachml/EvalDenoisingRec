import os
import time
import argparse
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from datetime import datetime

from loss import loss_function, PLC_uncertain_discard, loss_function_origin, loss_function_TCE, loss_function_RCE, drop_rate_schedule

import copy
import csv
from tqdm import tqdm
import toolz
import model
import evaluate
import ipdb
import data_utils
from utils import load_yaml_config, build_param, build_experiment_logger



########################### Temp Softmax #####################################
def temperature_scaled_softmax(logits,temperature):
	logits = logits / temperature
	return torch.softmax(logits, dim=0)




def eval(model, valid_loader, train_mat, count, param):
   
	model.eval()
	
	for user_valid, data_value_valid in valid_loader:
		with torch.no_grad():
			user_valid = user_valid.cuda()
			data_value_valid = data_value_valid.cuda()
			prediction_input_from_train = torch.tensor(train_mat[user_valid.cpu()]).cuda()
   
			#ipdb.set_trace()
			prediction = model(user_valid, prediction_input_from_train)
   

			#prediction_input_from_train = torch.tensor(train_mat[user_valid.cpu()]).cuda()
			#prediction = model(user_valid, prediction_input_from_train)
   
			with torch.no_grad():
				num_ps_per_user = data_value_valid.sum(1) 
				negative_samples = []
				users = []
				for u in range(data_value_valid.size(0)):
					batch_interaction = torch.randint(0, item_num, (int(num_ps_per_user[u].item()),))
					negative_samples.append(batch_interaction)
					users.extend([u] * int(num_ps_per_user[u].item()))

			negative_samples = torch.cat(negative_samples, 0)
			users = torch.LongTensor(users)	
			mask = data_value_valid.clone()
			mask[users, negative_samples] = 1
			groundtruth = data_value_valid[mask > 0.]
			pred = prediction[mask > 0.]
   
			if param['method'] == 'ERM':
				eval_loss = torch.mean(BCE_loss(pred, groundtruth)).detach().item()
			elif param['method'] == 'RCE':
				eval_loss = loss_function_RCE(pred, groundtruth, param["RCE_alpha"]).detach().item()
			elif param['method'] == 'TCE':
				eval_loss = loss_function_TCE(pred, groundtruth, drop_rate_schedule(count, param)).detach().item()
			elif param['method'] == 'UDT':
				eval_loss = torch.mean(BCE_loss(pred, groundtruth)).detach().item()
	return eval_loss
	   



def ERM_RE(model, valid_loader, train_mat, count, param, experiment_logger):
   
	model.eval()
	label0_losses = []
	label1_losses = []
	
	for user_valid, data_value_valid in valid_loader:
		with torch.no_grad():
      
			user_valid = user_valid.cuda()
			data_value_valid = data_value_valid.cuda()
			prediction_input_from_train = torch.tensor(train_mat[user_valid.cpu()]).cuda()
   
			
			prediction = model(user_valid, prediction_input_from_train) # prediction of the batch from train matrix
			#eval_loss = torch.mean(BCE_loss(prediction, prediction_input_from_train)).detach().item()
			#losses = BCE_loss(prediction, prediction_input_from_train)
			with torch.no_grad():
				num_ps_per_user = data_value_valid.sum(1) 
				negative_samples = []
				users = []
				for u in range(data_value_valid.size(0)):
					batch_interaction = torch.randint(0, item_num, (int(num_ps_per_user[u].item()),))
					negative_samples.append(batch_interaction)
					users.extend([u] * int(num_ps_per_user[u].item()))

			negative_samples = torch.cat(negative_samples, 0)
			users = torch.LongTensor(users)	
			mask = data_value_valid.clone()
			mask[users, negative_samples] = 1
			groundtruth = data_value_valid[mask > 0.]
			pred = prediction[mask > 0.]

			prediction = model(user_valid, prediction_input_from_train) # prediction of the batch from train 
			losses = BCE_loss(pred, groundtruth)
    
		label0_mask = (groundtruth == 0)
		label1_mask = (groundtruth == 1)
		
		label0_losses.extend(losses[label0_mask].detach().cpu().numpy().tolist())
		label1_losses.extend(losses[label1_mask].detach().cpu().numpy().tolist())


	label1_losses.sort()
	keep_count = int(len(label1_losses) * 0.8)
	label1_losses = label1_losses[:keep_count]

	eval_loss = (sum(label1_losses) + sum(label0_losses)) / (len(label1_losses) + len(label0_losses))

	return eval_loss
	   
	   
    
    

def test_per_epoch(model, test_data_pos, train_mat, valid_mat, epoch, eval_loss, experiment_logger, top_k):
	
	model.eval()
	predictedIndices = [] # predictions
	GroundTruth = list(test_data_pos.values())


	for users in toolz.partition_all(900, list(test_data_pos.keys())): # looping through users in test set
		user_id = torch.tensor(list(users)).cuda()
		data_value_test = torch.tensor(train_mat[list(users)]).cuda()
		predictions = model(user_id, data_value_test) # model prediction for given data
		test_data_mask = (train_mat[list(users)] + valid_mat[list(users)]) * -9999

		predictions = predictions + torch.tensor(test_data_mask).float().cuda()
		_, indices = torch.topk(predictions, top_k[-1]) # returns sortexd index based on highest probability
		indices = indices.cpu().numpy().tolist()
		predictedIndices += indices # a list of top 100 predicted indices

	precision, recall, NDCG, MRR = evaluate.compute_acc(GroundTruth, predictedIndices, top_k)
	
	file_exists = os.path.isfile(experiment_logger)
	with open(experiment_logger, 'a', newline='') as f:
		writer = csv.writer(f)
		if not file_exists:
			header = ['epoch', 'eval_loss', 
					 'Recall@5', 'Recall@10', 'Recall@20', 'Recall@50', 'Recall@100',
					 'NDCG@5', 'NDCG@10', 'NDCG@20', 'NDCG@50', 'NDCG@100']
			writer.writerow(header)
	
		row = [epoch, eval_loss]
		row.extend(recall)
		row.extend(NDCG)
		writer.writerow(row)
  
	print("################### TEST ######################")
	print("Recall {:.4f}-{:.4f}-{:.4f}-{:.4f}-{:.4f}".format(recall[0], recall[1], recall[2], recall[3], recall[4]))
	print("NDCG {:.4f}-{:.4f}-{:.4f}-{:.4f}-{:.4f}".format(NDCG[0], NDCG[1], NDCG[2], NDCG[3], NDCG[4]))
	
	
	




if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', 
		type = str,
		help = 'dataset used for training, options: amazon_book, yelp, adressa',
		default = 'adressa')
	parser.add_argument('--model', 
		type = str,
		help = 'model used for training. options: GMF, NeuMF',
		default = 'CDAE')
	parser.add_argument('--method', 
		type = str,
		help = 'method used for training. options: DCF',
		default = 'ERM')
	parser.add_argument("--gpu", 
		type=str,
		default="1",
		help="gpu card ID")
	parser.add_argument('--seed', 
		type = int,
		default = 2025,
		help = 'seed')
	parser.add_argument('--epochs', 
		type = int,
		default = 10,
		help = 'epochs')




	args = parser.parse_args()

	yaml_config = load_yaml_config(args)
	param = build_param(args, yaml_config)
 
	experiment_logger = build_experiment_logger(param)
	
	

	os.environ["CUDA_VISIBLE_DEVICES"] = param['gpu']
	cudnn.benchmark = True

	seed_num = param['seed']
	torch.manual_seed(param['seed']) # cpu
	torch.cuda.manual_seed(param['seed']) #gpu
	np.random.seed(param['seed']) #numpy
	random.seed(param['seed']) #random and transforms
	torch.backends.cudnn.deterministic=True # cudnn

	def worker_init_fn(worker_id):
		np.random.seed(seed_num + worker_id)


	data_path = f'../../data/{args.dataset}/'
	model_path = f'./models/'



	############################## PREPARE DATASET ##########################

	train_data, valid_data, train_data_pos, valid_data_pos, test_data_pos, user_pos, user_num ,item_num, train_mat, valid_mat, train_data_noisy = data_utils.load_all(f'{args.dataset}', data_path)



	train_mat_dense = train_mat.toarray()
	users_list = np.array([i for i in range(user_num)])
	train_dataset = data_utils.DenseMatrixUsers(users_list ,train_mat_dense)
	train_loader = data.DataLoader(train_dataset, batch_size=param['batch_size'], shuffle=True)

	valid_mat_dense = valid_mat.toarray()
	valid_dataset = data_utils.DenseMatrixUsers(users_list, valid_mat_dense)
	valid_loader = data.DataLoader(valid_dataset, batch_size=4096, shuffle=True)

	########################### CREATE MODEL #################################

	model = model.CDAE(user_num, item_num, 32, 0.2)
	model.cuda()
	BCE_loss = nn.BCEWithLogitsLoss(reduction='none')
	num_ns = 1 # negative samples
	optimizer = optim.Adam(model.parameters(), lr=param['lr'])


	########################### Training #####################################
	top_k = param['top_k']
	all_user_loss = [float(0) for _ in range(user_num)] # user level loss
	careless_users = []
	best_recall = 0
	count = 0 
	min_eval_loss = 1e9

	if param['method'] == 'UDT':
		user_temperature = np.linspace(param['temp1'], param['temp2'], user_num)
		user_factor = np.linspace(param['userfact1'], param['userfact2'], user_num)
	elif param['method'] == 'DCF':
		co_lambda_plan = param["co_lambda"] * np.linspace(1, 0, param["epochs"]) 
  

	for epoch in range(param['epochs']):
		if param["method"] == 'DCF' and epoch % param["time_step"] == 0:
			print('Time step initializing...')
			before_loss = np.zeros((len(train_dataset), 1))
			sn = torch.from_numpy(np.ones((len(train_dataset), 1)))
			before_loss_list = []	
			ind_update_list = []
     
   
		model.train()
		train_loss = 0

		for user, data_value in train_loader:
			user = user.cuda()
			data_value = data_value.cuda()
			prediction = model(user, data_value)
   
			if param["method"] == 'DCF':
				start_point = int(i * param["batch_size"])
				stop_point = int((i + 1) * param["batch_size"])
			#negative sampling
			with torch.no_grad():
				num_ns_per_user = data_value.sum(1) * num_ns
				negative_samples = []
				users = []
				for u in range(data_value.size(0)):
					batch_interaction = torch.randint(0, item_num, (int(num_ns_per_user[u].item()),))
					negative_samples.append(batch_interaction)
					users.extend([u] * int(num_ns_per_user[u].item()))


			negative_samples = torch.cat(negative_samples, 0)
			users = torch.LongTensor(users)
			mask = data_value.clone()
			mask[users, negative_samples] = 1
			groundtruth = data_value[mask > 0.]
			pred = prediction [mask > 0.]

			if param['method'] == 'ERM':
				loss = BCE_loss(pred, groundtruth)
    
			elif param['method'] == 'RCE':
				loss = loss_function_RCE(pred, groundtruth, param["RCE_alpha"])
    
			elif param['method'] == 'TCE':
				loss = loss_function_TCE(pred, groundtruth, drop_rate_schedule(count, param))

			elif param["method"] == 'DCF':
				pass
    
			elif param["method"] == 'UDT':
				loss_ = BCE_loss(pred, groundtruth)
				# section the loss base on mask sum
				split_loss = torch.split(loss_,mask.sum(1, dtype=torch.int).tolist(), dim=0)
				split_groundtruth = torch.split(groundtruth,mask.sum(1, dtype=torch.int).tolist(), dim=0)
				with torch.no_grad():
					for i , user_id in enumerate(user):
						# user level loss
						user_loss = torch.mean(split_loss[i], dim = 0)
						all_user_loss[user_id] = user_loss.item()

						# per user interaction lossloss
						if epoch <= 1:
							ui_factor = torch.ones_like(split_loss[i])

						else:                    
							temp = user_temperature[np.where(sorted_users==user[i].item())[0][0]]

							fact = user_factor[np.where(sorted_users==user[i].item())[0][0]] # user level factor
							ui_factor = torch.ones_like(split_loss[i])
							ui_factor[split_groundtruth[i]==1] = temperature_scaled_softmax(-split_loss[i][split_groundtruth[i] == 1], temp) * mask[i].sum() / 2.0 

							ui_factor *= fact

						if i == 0:
							ui_loss_factor = ui_factor
						else:
							ui_loss_factor = torch.cat((ui_loss_factor, ui_factor))
				ui_loss_factor = ui_loss_factor.cuda()

						# calculating loss

				loss = torch.mean(ui_loss_factor * loss_)
		

			optimizer.zero_grad()
		
			loss = torch.mean(loss)
			loss.backward()
			optimizer.step()
   
			count += 1
   
  
		if param["method"] == 'UDT':
			user_loss_np = np.array(all_user_loss)
			sorted_users = np.argsort(user_loss_np) # lowest to highest loss

    
		if param['method'] == 'ERM_RE':
			eval_loss = ERM_RE(model, valid_loader, train_mat_dense, count, param, experiment_logger)
		else:
			eval_loss = eval(model, valid_loader, train_mat_dense, count, param)
  
		print("epoch: {}, loss:{}".format(epoch, eval_loss))
		
		if param['dataset'] != "yelp":
			if (epoch+1) % 20 == 0:
				test_per_epoch(model, test_data_pos, train_mat_dense, valid_mat_dense, epoch, eval_loss, experiment_logger, param['top_k'])
		else:	
			if (epoch+1) % 25 == 0:
				test_per_epoch(model, test_data_pos, train_mat_dense, valid_mat_dense, epoch, eval_loss, experiment_logger, param['top_k'])
  
  
	

	