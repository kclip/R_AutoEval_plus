import functools
import os
from typing import Dict
import argparse

import tqdm
import numpy as np
from pathlib import Path

import pickle
import pathlib
from transformers import AutoTokenizer, AutoModelForCausalLM

# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataeval.load import *

# print(os.getcwd())

import sys
sys.path.append('.')
# sys.path.append('dataeval')
# sys.path.append('models')
# sys.path.append('utils')

# import load_worker as lw

# import persist_to_disk as ptd
# ptd.config.generate_config()

import dataeval.load_worker as lw
import models

import models.nli as sc
import utils


parser = argparse.ArgumentParser(description='ePP+_LLM_quant')
parser.add_argument('--model', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct', help='model name of the LLM to be quantized')
parser.add_argument('--dataset', type=str, default=None, required=False)
#parser.add_argument('--quant_setting', type=str, default=None)
parser.add_argument('--mx_spec', type=str, required=True)
parser.add_argument('--seed', type=int, default=10)
parser.add_argument("--batch_size", default=None, type=int, required=True,
                help="Number of questions in each batch")

parser.add_argument('--fake_quality', type=str, default=None)

args = parser.parse_args()


DEFAULT_DEVICE = 'cuda:7'

IGNORE_INDEX = -100


if __name__ == '__main__':
    import _settings 
    device = 'cuda:0'
    model_name = args.model
    if '/' in model_name:
        model_name = model_name.replace('/', '_')
    llama_path = _settings.LLAMA_PATH
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=llama_path) #, return_token_type_ids=False
     
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    num_debug = None
    cache_dir = os.path.join(_settings.GENERATION_FOLDER, f'{model_name}_{args.dataset}_{args.seed}/batch_results/{args.batch_size}_None')
    if args.mx_spec == 'full_precision':
        pass
    else:
        cache_dir  =  cache_dir + '/' + args.mx_spec + '/'
    cache_dir = Path(cache_dir)
    pkl_files = list(cache_dir.glob('*.pkl'))
    if not pkl_files:
        raise FileNotFoundError("No .pkl files found in the specified folder.")
    else:
        path_Y_hat = os.path.join(cache_dir, pkl_files[0].name)   
        if args.mx_spec == 'full_precision':
            qsnr = -0.1
            avg_bitwidth = 16
            print('qsnr: ', qsnr, 'avg bitwidth: ', avg_bitwidth)
        else:
            parts = Path(pkl_files[0].name).stem.split('_')
            qsnr, avg_bitwidth = map(float, parts[-2:])
            avg_bitwidth += 1 # sign bit
            print('qsnr: ', qsnr, 'avg bitwidth: ', avg_bitwidth)
    print('reading cleaned outputs...\n')
    cleaned_seq = read_cleaned_outputs_new(path_Y_hat, tokenizer)
    #print('cleaned_seq', cleaned_seq[:10])
    ids = [_['id'] for _ in cleaned_seq]
    print('total data size', len(ids), len(cleaned_seq))
    # For evaluation of the generated responses
    print('reading rouge scores...\n')

    if args.fake_quality == 'best':
        mx_spec_fake = ''
    elif args.fake_quality == 'good':
        mx_spec_fake = 'mx9'
    elif args.fake_quality == 'med':
        mx_spec_fake = 'mx6'
    elif args.fake_quality == 'bad':
        mx_spec_fake = 'mx4'
    else:
        raise NotImplementedError
    
    fake_model_name = 'meta-llama/Llama-3.3-70B-Instruct'
    if '/' in fake_model_name:
        fake_model_name = fake_model_name.replace('/', '_')
    fake_cache_dir = os.path.join(_settings.GENERATION_FOLDER, f'{fake_model_name}_{args.dataset}_{args.seed}/batch_results/{args.batch_size}_None')
    fake_cache_dir  =  fake_cache_dir + '/' + mx_spec_fake + '/'
    fake_cache_dir = Path(fake_cache_dir)
    print('------------running setting: ', args.mx_spec, args.fake_quality)
    fake_pkl_files = list(fake_cache_dir.glob('*.pkl'))
    if not fake_pkl_files:
        raise FileNotFoundError("No .pkl files found in the specified folder.")
    else:
        fake_path = os.path.join(fake_cache_dir, fake_pkl_files[0].name)      
    rouges = read_rouges_new(path_Y_hat, 0, args.batch_size, tokenizer, parallel=False, num_eval=num_debug, if_fake_label=True, fake_path=fake_path) # compute the rougeL scores
    avg_score = 0
    num_samples = len(rouges)
    idx_list = []
    score_list = np.full((num_samples, 1), np.nan)
    actual_spec = np.zeros(2)
    actual_spec[0] = qsnr
    actual_spec[1] = avg_bitwidth
    ind_sample = 0
    for rouge in rouges:
        idx_list.append(rouge['id'])
        score_list[ind_sample] = rouge['generations'][0]['rougeL']
        ind_sample += 1
    path_common = f'./loss_tables/num_sample_{num_samples}' + '/' + args.fake_quality + '/' + args.dataset + '/' + model_name + '/' + args.mx_spec + '/'
    path_idx = Path(path_common + 'idx.npy')
    path_idx.parent.mkdir(parents=True, exist_ok=True) 
    np.save(path_idx, np.array(idx_list))
    path_score = Path(path_common + 'score.npy')
    np.save(path_score, score_list)
    path_actual_spec = Path(path_common + 'actual_spec.npy')
    np.save(path_actual_spec, actual_spec)
    idx_list = np.load(path_idx)
    score_list = np.load(path_score)
    actual_spec = np.load(path_actual_spec)
    print(idx_list, 'num sample: ', len(score_list), 'avg score: ', np.average(score_list), 'actual_spec: ', actual_spec)
