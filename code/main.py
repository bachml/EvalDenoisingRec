import os
import time
import argparse
import numpy as np
import random
import os
import time
import csv
from datetime import datetime
import yaml

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import numpy as np
from utils import load_yaml_config, build_param, build_experiment_logger

import model
import evaluate
import data_utils
from loss import *
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import temperature_scaled_softmax
import time

from loss import drop_rate_schedule




########################### Eval #####################################
def eval(model, valid_loader, count, param):
		
	model.eval()
	eval_loss = 0
	valid_loader.dataset.ng_sample() # negative sampling
	for user, item, label, noisy_or_not in valid_loader:
		user = user.cuda()
		item = item.cuda()
  
  
		label = label.float().cuda()
		
		if args.method == 'ERM':
			prediction = model(user, item)
			eval_loss += BCE_loss(prediction, label).detach().item()
		elif args.method == 'TCE':
			prediction = model(user, item)	
			eval_loss += loss_function_TCE(prediction, label, drop_rate_schedule(count, param)).detach().item()
		elif args.method == 'RCE':
			prediction = model(user, item)
			eval_loss += loss_function_RCE(prediction, label, param["RCE_alpha"]).detach().item()
		elif args.method == 'DCF':
			# we follow the orginal implementation by the author
			prediction = model(user, item)
			eval_loss += BCE_loss(prediction, label).detach().item()
		elif args.method == 'UDT':
			prediction = model(user, item)
			eval_loss += BCE_loss(prediction, label).detach().item()
	eval_loss = eval_loss / len(valid_loader)

	print("################### EVAL ######################")
	print("Eval loss:{}".format(eval_loss))

	return eval_loss
	



########################### Eval #####################################
def eval_RevistingEvaluation(model, valid_loader, count, param):
	model.eval()
	eval_loss = 0
	valid_loader.dataset.ng_sample() # negative sampling
	
	label0_losses = []
	label1_losses = []
	
	for user, item, label, noisy_or_not in valid_loader:
		user = user.cuda()
		item = item.cuda()
		label = label.float().cuda()
		
		prediction = model(user, item)
		losses = F.binary_cross_entropy_with_logits(prediction, label, reduction='none')
   
		label0_mask = (label == 0)
		label1_mask = (label == 1)

		label0_losses.extend(losses[label0_mask].detach().cpu().numpy().tolist())
		label1_losses.extend(losses[label1_mask].detach().cpu().numpy().tolist())

	# Process loss for label=1: sort and discard the top 20% elements with highest loss values
	label1_losses.sort()
	
	keep_count = int(len(label1_losses) * 0.8)
	label1_losses = label1_losses[:keep_count]
	eval_loss = (sum(label1_losses) + sum(label0_losses)) / (len(label1_losses) + len(label0_losses))

	print("################### EVAL ######################")
	print("Eval loss:{}".format(eval_loss))

	return eval_loss
	




 
def test_per_epoch(model, test_data_pos, user_pos, epoch, eval_loss, experiment_logger, param):
	top_k = param["top_k"]
	model.eval()
	# The second item is too large, and there may be a problem with vacant values in predictions.
	_, recall, NDCG, _ = evaluate.test_all_users(model, param["test_batch_size"], item_num, test_data_pos, user_pos, top_k)
	print("################### TEST ######################")
	print("Recall {:.4f}-{:.4f}-{:.4f}-{:.4f}-{:.4f}".format(recall[0], recall[1], recall[2], recall[3], recall[4]))
	print("NDCG {:.4f}-{:.4f}-{:.4f}-{:.4f}-{:.4f}".format(NDCG[0], NDCG[1], NDCG[2], NDCG[3], NDCG[4]))

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
	


if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	parser.add_argument('--dataset', 
		type = str,
		help = 'dataset used for training, options: amazon_book, yelp, adressa',
		default = 'ml-100k')
	parser.add_argument('--model', 
		type = str,
		help = 'model used for training. options: GMF, NeuMF',
		default = 'GMF')
	parser.add_argument('--method', 
		type = str,
		help = 'method used for training. options: DCF',
		default = 'DCF')
	parser.add_argument("--gpu", 
		type=str,
		default="0",
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
 

	os.environ["CUDA_VISIBLE_DEVICES"] = param["gpu"]
	cudnn.benchmark = True

	seed_num =  param["seed"]
	torch.manual_seed(seed_num) # cpu
	torch.cuda.manual_seed(seed_num) #gpu
	np.random.seed(seed_num) #numpy
	random.seed(seed_num) #random and transforms
	torch.backends.cudnn.deterministic=True # cudnn

	def worker_init_fn(seed_num, worker_id):
		np.random.seed(seed_num + worker_id)

	#sys.stdout = open('../loggg.log', 'a')
	data_path = '../data/{}/'.format(args.dataset)
	#model_path = 'models/{}/'.format(args.dataset)


 
	def worker_init_fn(worker_id):
		np.random.seed(seed_num + worker_id)

 

	############################## PREPARE DATASET ##########################
	train_data, valid_data, test_data_pos, user_pos, user_num ,item_num, train_mat, train_data_noisy = data_utils.load_all(args.dataset, data_path)
	train_adj = data_utils.create_adj_mat(train_mat, user_num, item_num)

	# construct the train and test datasets
	train_dataset = data_utils.NCFData(
			train_data, item_num, train_mat, param["num_ng"], 0, train_data_noisy)
	valid_dataset = data_utils.NCFData(
			valid_data, item_num, train_mat, param["num_ng"], 1)

	train_loader = data.DataLoader(train_dataset,
			batch_size=param["batch_size"], shuffle=True, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
	valid_loader = data.DataLoader(valid_dataset,
			batch_size=param["batch_size"], shuffle=True, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

	print("data loaded! user_num:{}, item_num:{} train_data_len:{} test_user_num:{}".format(user_num, item_num, len(train_data), len(test_data_pos)))

	########################### CREATE MODEL #################################
	GMF_model = None
	MLP_model = None

	if args.model == 'GMF':
		model = model.NCF(user_num, item_num, param["factor_num"], param["num_layers"], 
						param["dropout"], param["model"], GMF_model, MLP_model)
	elif args.model == 'NeuMF':
		model = model.NCF(user_num, item_num, param["factor_num"], param["num_layers"], 
						param["dropout"], param["model"], GMF_model, MLP_model)
	elif args.model == 'LightGCN':
		model = model.LightGCN(user_num, item_num, train_adj, param["factor_num"], param["num_layers"])

	#best_model = None
	model.cuda()
	BCE_loss = nn.BCEWithLogitsLoss()


	if args.method == 'DCF':
		co_lambda_plan = param["co_lambda"] * np.linspace(1, 0, param["epochs"]) 
	elif args.method == 'UDT':
		user_temperature = torch.linspace(param["temp1"], param["temp2"], steps=user_num)
		user_factor = torch.linspace(param["userfact1"], param["userfact2"], steps=user_num)
		train_mat_dense = torch.tensor(train_mat.toarray()).cpu()
		train_mat_sum = train_mat_dense.sum(1)


	optimizer = optim.Adam(model.parameters(), lr=param["lr"])


	########################### TRAINING #####################################
	count, best_hr = 0, 0
	best_loss = 1e9




	for epoch in range(param["epochs"]):
		if param["method"] == 'DCF' and epoch % param["time_step"] == 0:
			print('Time step initializing...')
			before_loss = np.zeros((len(train_dataset), 1))
			sn = torch.from_numpy(np.ones((len(train_dataset), 1)))
			before_loss_list=[]	
			ind_update_list = []
		if param["method"] == 'UDT':
			loss_mat = float('inf') - torch.zeros_like(train_mat_dense)
			user_loss = torch.zeros_like(train_mat_sum)
		
		model.train()
		start_time = time.time()
		train_loader.dataset.ng_sample()
		

		for i, (user, item, label, noisy_or_not) in enumerate(train_loader):
		
			user = user.cuda()
			item = item.cuda()
   
			if param["method"] == 'DCF':
				start_point = int(i * param["batch_size"])
				stop_point = int((i + 1) * param["batch_size"])

			label = torch.tensor(train_mat[user.cpu().numpy().tolist(), item.cpu().numpy().tolist()].todense()).squeeze().cuda()

			model.zero_grad()
		
			
			if param["method"] == 'DCF':
				prediction = model(user, item)
				loss, train_adj, loss_mean, ind_update = PLC_uncertain_discard(user, item, train_mat, prediction, label, drop_rate_schedule(count, param), epoch, sn[start_point:stop_point], before_loss[start_point:stop_point], co_lambda_plan[epoch], param["relabel_ratio"])
				before_loss_list += list(np.array(loss_mean.detach().cpu()))
				ind_update_list += list(np.array(ind_update.cpu() + i * param["batch_size"]))
				train_mat = train_adj
	
			elif param["method"] in ['ERM', 'ERM_RE']:
				prediction = model(user, item)
				loss = loss_function_origin(prediction, label)
	
			elif param["method"] == 'TCE':
				prediction = model(user, item)
				loss = loss_function_TCE(prediction, label, drop_rate_schedule(count, param))

			elif param["method"] == 'RCE':
				prediction = model(user, item)
				loss = loss_function_RCE(prediction, label, param["RCE_alpha"])
	
			elif param["method"] == 'UDT':
				batch_pos_user = user[label > 0.]
				batch_pos_item = item[label > 0.]
	
				prediction = model(user, item)
				loss_pre = nn.BCEWithLogitsLoss(reduction='none')(prediction, label)

				# for user level
				user_loss[user.cpu()] += loss_pre.cpu()
				with torch.no_grad():
					if epoch <= 1:
						mul_factor = torch.ones_like(loss_pre)
					else: 
						mul_factor = ui_factor[user.cpu(), item.cpu()].cuda()

				# for updating user iteraction level
				loss_ui = loss_pre[label > 0.].cpu()
				loss_mat[batch_pos_user.cpu(), batch_pos_item.cpu()] = loss_ui
				loss = torch.mean(mul_factor * loss_pre)

			loss.backward()
			optimizer.step()
	
			count += 1
			

		print("epoch: {}, iter: {}, loss:{}".format(epoch, count, loss))
  
		if param["method"] == 'ERM_RE':
			eval_loss = eval_RevistingEvaluation(model, valid_loader, count, param)
		else:
			eval_loss = eval(model, valid_loader, count, param)
  

		# Only perform global test every k epochs due to high computational cost
		if param['dataset'] != "yelp":
			if param["model"] == "LightGCN":
				if (epoch+1) % 10 == 0:
					test_per_epoch(model, test_data_pos, user_pos, epoch, eval_loss, experiment_logger, param)
			else:
				test_per_epoch(model, test_data_pos, user_pos, epoch, eval_loss,  experiment_logger, param)
		else:
			if param["model"] == "LightGCN":
				if (epoch+1) % 15 == 0:
					test_per_epoch(model, test_data_pos, user_pos, epoch, eval_loss,  experiment_logger, param)
			else:
				if (epoch+1) % 5 == 0:
					test_per_epoch(model, test_data_pos, user_pos, epoch, eval_loss, experiment_logger, param)
		
		


		model.train()
	
		if param["method"] == 'DCF':
			before_loss = np.array(before_loss_list).astype(float)
			all_zero_array = np.zeros((len(train_dataset), 1))
			all_zero_array[np.array(ind_update_list)] = 1	
			sn += torch.from_numpy(all_zero_array)

		if param["method"] == 'UDT':
			all_user_loss = user_loss/train_mat_sum # avg of different number of interactions
			sorted_users = torch.argsort(all_user_loss) # users with lowest loss to highest loss

			temp_fact = torch.zeros_like(user_temperature)
			temp_fact[sorted_users] = user_temperature


			user_fact = torch.zeros_like(user_temperature)
			user_fact[sorted_users] = user_factor
			
			# ui level
			ui_factor = temperature_scaled_softmax(-loss_mat, temp_fact) * torch.unsqueeze(train_mat_sum,1) #multiply to make it even
			ui_factor[ui_factor==0] = 1.0


			ui_factor *=  torch.unsqueeze(user_fact,1)



print("############################## Training End. ##############################")