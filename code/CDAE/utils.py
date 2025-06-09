import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import os
import yaml
import re
from datetime import datetime



def temperature_scaled_softmax(logits,temperature):
    logits = logits / torch.unsqueeze(temperature,1)
    return torch.softmax(logits, dim=1)
  

def load_yaml_config(args):

    config_dir = os.path.join('experiments_config', args.dataset)
    config_file = f"{args.model}-{args.method}.yaml"
    config_path = os.path.join(config_dir, config_file)
    

    if os.path.exists(config_path):
        print(f"Loading configuration from: {config_path}")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        for key, value in config.items():
            if isinstance(value, str) and re.match(r'^-?\d+(\.\d+)?[eE][+-]?\d+$', value):
                config[key] = float(value)
                
        return config
    else:
        print(f"Warning: Configuration file not found at {config_path}")
        return {}



def build_param(args, yaml_config):
    param_dict = {}

    for key in vars(args):
        param_dict[key] = getattr(args, key)
  
    for key, value in yaml_config.items():
        param_dict[key] = value
        
    return param_dict


def build_experiment_logger(param):
    if not os.path.exists('metric_records'):
        os.makedirs('metric_records')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    project_name = f"{param['model']}-{param['dataset']}-{param['method']}-{param['seed']}"

    experment_log_folder = os.path.join('metric_records', project_name)
    if not os.path.exists(experment_log_folder):
        os.makedirs(experment_log_folder)

    experment_log_file = os.path.join(experment_log_folder, f"{project_name}_{timestamp}.csv")
    return experment_log_file




def new_build_experiment_logger(param):
    if not os.path.exists('new_metric_records'):
        os.makedirs('new_metric_records')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    project_name = f"{param['model']}-{param['dataset']}-{param['method']}-{param['seed']}"

    experment_log_folder = os.path.join('new_metric_records', project_name)
    if not os.path.exists(experment_log_folder):
        os.makedirs(experment_log_folder)

    experment_log_file = os.path.join(experment_log_folder, f"{project_name}_{timestamp}.csv")
    return experment_log_file

