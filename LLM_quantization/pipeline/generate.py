
import argparse
import glob
import json
import os
import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from transformers import AutoTokenizer, AutoModelForCausalLM
from mx_quantization.mx import mx_quant
from mx_quantization.mx_cpu import mx_quant_cpu
import json

#, , AutoModelForSeq2SeqLM, StoppingCriteria, StoppingCriteriaList#, pipeline

import pandas as pd
import torch
import tqdm
import transformers

import _settings
import dataeval.coqa as coqa
import dataeval.triviaqa as triviaqa
from datasets import load_dataset, Dataset
# import models
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='meta-llama/Llama-3.1-8B-Instruct')
# parser.add_argument('--model', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct')
parser.add_argument('--dataset', type=str, default=None, required=False)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--fraction_of_data_to_use', type=float, default=1.0)
parser.add_argument('--num_generations_per_prompt', type=int, default=20)
parser.add_argument('--temperature', type=float, default='1.0')
parser.add_argument('--decoding_method', type=str, default='greedy')
parser.add_argument('--top_p', type=float, default=1.0)
parser.add_argument('--top_k', type=int, default=0)
parser.add_argument('--seed', type=int, default=10)
parser.add_argument('--nprocess', type=int, default=None)
parser.add_argument("--idx", default=None, type=int, required=False,
                        help="Index of sequences of questions")
parser.add_argument("--batch_size", default=None, type=int, required=True,
                    help="Number of questions in each batch")

parser.add_argument('--mx_spec', type=str, required=True)


args = parser.parse_args()

# _UNUSED_TOKENIZER = models.load_tokenizer()
def get_dataset_fn(data_name):
    if data_name == 'triviaqa':
        return triviaqa.get_dataset
    if data_name == 'coqa':
        return coqa.get_dataset
    if data_name == 'nq_open':
        return nq_open.get_dataset

def get_generation_config(input_ids, tokenizer, data_name):
    assert len(input_ids.shape) == 2
    max_length_of_generated_sequence = 256
    if data_name == 'triviaqa':
        generation_config = triviaqa._generate_config(tokenizer)
    if data_name == 'coqa':
        generation_config = coqa._generate_config(tokenizer)
    if data_name == 'nq_open':
        generation_config = nq_open._generate_config(tokenizer)
    generation_config['max_new_tokens'] = max_length_of_generated_sequence
    generation_config['early_stopping'] = True
    # https://jaketae.github.io/study/gpt2/#setup
    generation_config['pad_token_id'] = tokenizer.eos_token_id
    return generation_config

@torch.no_grad()
def get_generations(model_name:str, args, seed=10, old_sequences=None, max_num_gen_once=4, llama_path=None, data_path = None):
    device = args.device
    #model, tokenizer = models.load_model_and_tokenizer(model_name, args.device)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map='auto', torch_dtype="bfloat16", cache_dir=llama_path)#.to(cls._device)
    model.eval()
    
    

    ##### weight quantization ######
    print('--------------------------------')
    print(args.mx_spec)
    print('--------------------------------')
    if args.mx_spec == 'full_precision':
        pass
    elif args.mx_spec == 'mx9':
        mx_dict = {}
        mx_dict['k1'] = 16
        mx_dict['k2'] = 2
        mx_dict['d1'] = 8
        mx_dict['d2'] = 1
        mx_dict['m'] = 7
        try:
            avg_qsnr, avg_bitwidth, model = mx_quant(model, mx_dict)
        except:
            avg_qsnr, avg_bitwidth, model = mx_quant_cpu(model, mx_dict)
    elif args.mx_spec == 'mx6':
        mx_dict = {}
        mx_dict['k1'] = 16
        mx_dict['k2'] = 2
        mx_dict['d1'] = 8
        mx_dict['d2'] = 1
        mx_dict['m'] = 4
        try:
            avg_qsnr, avg_bitwidth, model = mx_quant(model, mx_dict)
        except:
            avg_qsnr, avg_bitwidth, model = mx_quant_cpu(model, mx_dict)
        print('avg qsnr', avg_qsnr, 'avg bitwidht', avg_bitwidth)
    elif args.mx_spec == 'mx4':
        mx_dict = {}
        mx_dict['k1'] = 16
        mx_dict['k2'] = 2
        mx_dict['d1'] = 8
        mx_dict['d2'] = 1
        mx_dict['m'] = 2
        try:
            avg_qsnr, avg_bitwidth, model = mx_quant(model, mx_dict)
        except:
            avg_qsnr, avg_bitwidth, model = mx_quant_cpu(model, mx_dict)
        print('avg qsnr', avg_qsnr, 'avg bitwidht', avg_bitwidth)
    else:
        try:  # --mx_spec '{"m": 2,3,4,5,6,7,8,9,10, "ratio": {2,4,8} }'
            mx_dict = json.loads(args.mx_spec)
            print('quantization setting', mx_dict)
            if 'k1' in mx_dict.keys():
                mx_dict['k1'] = mx_dict['k1'] 
            else:
                mx_dict['k1'] = 16 # default, Table II https://arxiv.org/pdf/2302.08007
            mx_dict['k2'] = max(int(mx_dict['k1']//mx_dict['ratio']), 1) 
            mx_dict['d1'] = 8
            mx_dict['d2'] = 1
        except:
            print('you gave: ', mx_dict, 'which is not supported here')
            raise NotImplementedError
            #print('mx_spec not given in a right format, it should be in the format of: --mx_spec '{"m": 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16, "ratio": {2,4,8} }'')
        try:
            avg_qsnr, avg_bitwidth, model = mx_quant(model, mx_dict)
        except:
            avg_qsnr, avg_bitwidth, model = mx_quant_cpu(model, mx_dict)
        print('avg qsnr', avg_qsnr, 'avg bitwidht', avg_bitwidth)
    ################################    
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=llama_path) #, return_token_type_ids=False
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.eos_token_id = tokenizer.pad_token_id
    model.config.pad_token_id = tokenizer.eos_token_id
    utils.seed_everything(seed)
    if os.path.isfile(data_path + args.dataset + '.pth'):
        dataset = torch.load(data_path + args.dataset + '.pth')
    else:
        dataset = get_dataset_fn(args.dataset)(tokenizer)
        if args.fraction_of_data_to_use < 1.0:
            dataset = dataset.train_test_split(test_size=(1 - args.fraction_of_data_to_use), seed=seed)['train']
        torch.save(dataset, data_path + args.dataset + '.pth')    
    # subsetting
    # dataset = Dataset.from_dict(dataset[((args.idx-1)*args.batch_size):(args.idx*args.batch_size)])
    assert args.fraction_of_data_to_use == 1
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    if old_sequences is None:
        old_sequences = []
    old_sequences = {_['id']: _ for _ in old_sequences}

    sequences = []
    for batch_idx, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        # # masking
        # if batch_idx < (args.idx-1)*args.batch_size or batch_idx >= args.idx*args.batch_size:
        #     continue;

        # if batch['id'][0] in old_sequences:
        #     sequences.append(old_sequences[batch['id'][0]])
        #     continue

        input_ids = batch['input_ids'].to(device)
        input_length = input_ids.shape[1]
        generation_config = get_generation_config(input_ids, tokenizer, args.dataset)
        generation_config = transformers.GenerationConfig(**generation_config)
        if args.decoding_method == 'beam_search':
            raise NotImplementedError()
        elif args.decoding_method == 'greedy':
            most_likely_generations = model.generate(input_ids, attention_mask=batch['attention_mask'].to(device),
                                                    num_beams=1,
                                                    do_sample=False,
                                                    generation_config=generation_config).cpu()[0, input_length:]
        # remember the data
        curr_seq = dict(
            prompt=input_ids.cpu()[0],
            id=batch['id'][0],
            question=batch['question'][0],
            answer=batch['answer'][0],
            additional_answers=[],
        )
        curr_seq.update(
            dict(
                most_likely_generation_ids = most_likely_generations,
            )
        )
        curr_seq.update(
            dict(
                most_likely_generation=tokenizer.decode(curr_seq['most_likely_generation_ids'], skip_special_tokens=True),
            )
        )
        if args.dataset == 'coqa':
            curr_seq['additional_answers'] = [x[0] for x in batch['additional_answers']]
        sequences.append(curr_seq)
    return sequences, avg_qsnr, avg_bitwidth


def main(overwrite=False, continue_from=None, parallel:int=None):
    old_sequences = []
    model_name = args.model
    if '/' in model_name:
        model_name = model_name.replace('/', '_')
    cache_dir = os.path.join(_settings.GENERATION_FOLDER, f'{model_name}_{args.dataset}_{args.seed}/batch_results/{args.batch_size}_{args.idx}')
    cache_dir  =  cache_dir + '/' + args.mx_spec + '/'
    
    LLAMA_PATH = _settings.LLAMA_PATH
    DATA_PATH = _settings.DATA_PATH
    os.makedirs(cache_dir, exist_ok=True)
    old_results = glob.glob(os.path.join(cache_dir, '*.pkl'))
    old_results = [_ for _ in old_results if '_partial' not in _]
    if len(old_results) > 0 and not overwrite:
        print(f'Found {len(old_results)} generations in {cache_dir}.')
        return
    run_id = len(old_results)

    with open(os.path.join(cache_dir, f'args{run_id}.json'), 'w') as f:
        json.dump(args.__dict__, f)
    print(f'Generating {args.num_generations_per_prompt} generations per prompt for {model_name} on {args.dataset}...')
    print(f"Saving to {os.path.join(cache_dir, f'{run_id}_{args.batch_size}_{args.idx}.pkl')}")
    
    sequences, avg_qsnr, avg_bitwidth, = get_generations(model_name, args, seed=args.seed, old_sequences=old_sequences, llama_path=LLAMA_PATH, data_path=DATA_PATH)
    print('sequences:', sequences)

    #cache_dir_quan = cache_dir + '/' + args.mx_spec + '/'
    print(f'Writing {len(sequences)} generations to {cache_dir}...')
    pd.to_pickle(sequences, os.path.join(cache_dir, f'{run_id}_{args.batch_size}_{args.idx}_{avg_qsnr}_{avg_bitwidth}.pkl'))
    return

if __name__ == '__main__':
    task_runner = main(parallel=args.nprocess)