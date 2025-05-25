import random
import argparse
import glob
import json
import os
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from experiments.data.instruction_induction.load_data import load_data, tasks
from automatic_prompt_engineer import ape, data, config, template_including_icl, llm
from experiments.evaluation.instruction_induction.exec_accuracy import exec_accuracy_evaluator
from pathlib import Path
import pickle 
import numpy as np
import pandas as pd
import torch
import tqdm
import transformers

import _settings
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='meta-llama/Llama-3.1-8B-Instruct')
parser.add_argument('--fake_quality', type=str, default='good')
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
parser.add_argument('--task', type=str, default='None')

args = parser.parse_args()


def get_query(prompt, eval_template, input_, output_, demo_data, demos_template):
    #print(demos_template, demo_data)
    demos = demos_template.fill(demo_data)
    query = eval_template.fill(prompt=prompt,
                               input=input_,
                               output='',
                               full_demo=demos)
    return query


@torch.no_grad()
def get_generations(model_name:str, batch_size, loaded_model, tokenizer,  args, prompt, eval_template, eval_data,demos_template,few_shot_data, config, seed=10, old_sequences=None, max_num_gen_once=4, llama_path=None, data_path = None):
    device = args.device
    config['model']['gpt_config']['model'] = args.model

    if config['model']['gpt_config']['model'] == 'meta-llama/Llama-3.1-8B-Instruct':
        eval_template = "Instruction: [PROMPT]\n\nInput: [INPUT]\nOutput: [OUTPUT]"
        eval_template = template_including_icl.EvalTemplate(eval_template)
        gen_mode = 'APE'
        args.fake_quality = 'None'
        config['model']['batch_size'] = batch_size # 100
    elif config['model']['gpt_config']['model'] == 'meta-llama/Llama-3.3-70B-Instruct':
        # ICL
        config['model']['batch_size'] = batch_size # 100
        eval_template = "Input: [INPUT]\nOutput: [OUTPUT]"
        eval_template = template_including_icl.DemosTemplate(eval_template, '\n\n' + eval_template)
        gen_mode = 'ICL'
        tot_few_shot = len(few_shot_data[0])
        

        if args.fake_quality == 'best':
            config['num_few_shot'] = tot_few_shot #'all'
        elif args.fake_quality == 'gooderer':
            config['num_few_shot'] = 7
        elif args.fake_quality == 'gooder':
            config['num_few_shot'] = 6
        elif args.fake_quality == 'good':
            config['num_few_shot'] = 5 
        elif args.fake_quality == 'medgood':
            config['num_few_shot'] = 4
        elif args.fake_quality == 'med':
            config['num_few_shot'] = 3 
        elif args.fake_quality == 'medbad':
            config['num_few_shot'] = 2
        elif args.fake_quality == 'bad':
            config['num_few_shot'] = 1
        else:
            pass
        print('tot few shot', tot_few_shot, 'we are now using: ', config['num_few_shot'])
        if config['num_few_shot'] > tot_few_shot:
            print('now enough few shot data')
            raise NotImplementedError
    else:
        raise NotImplementedError



    # few shot data should be the same as the one used for prompt gen. 
    queries = []
    answers = []
    inputs = []
    if gen_mode == 'ICL':
        demo_data = data.subsample_data(
                few_shot_data, config['num_few_shot'])
    else:
        demo_data = None
    for d in zip(*eval_data):
        input_, output_ = d
        if gen_mode == 'APE':
            query = eval_template.fill(prompt=prompt, input=input_, output='')
        elif gen_mode == 'ICL':
            query = eval_template.fill(data=demo_data, input=input_, output='')
        else:
            raise NotImplementedError
        queries.append(query)
        answers.append(output_)
        inputs.append(input_)

    model = llm.model_from_config(config['model'])
    model_outputs = model.generate_text(queries, 1, loaded_model, tokenizer)
    total_seq = dict(inputs=inputs, 
        queries = queries,
        answers=answers,
        model_outputs=model_outputs,
        )
    return total_seq, len(prompt)


def main(task, loaded_model, tokenizer, batch_size, overwrite=False, continue_from=None, parallel:int=None, success_accum=None):

    total_prompts = np.load('./saved_data/'+ task + '/total_cand_prompts.npy')
    try:
        with open(f'./saved_data/{task}/prompt_gen_data.pkl', 'rb') as f:
            prompt_gen_data_entire = pickle.load(f)
        with open(f'./saved_data/{task}/eval_data.pkl', 'rb') as f:
            eval_data_only = pickle.load(f)
        with open(f'./saved_data/{task}/test_data.pkl', 'rb') as f:
            test_data_only = pickle.load(f)
        with open(f'./saved_data/{task}/prompt_gen_data_sample_single.pkl', 'rb') as f:
            prompt_gen_data = pickle.load(f)
    except:
        raise NotImplementedError
    
    try:
        with open(f'./saved_data/{task}/eval_te.pkl', 'rb') as f:
            eval_data = pickle.load(f)
    except:
        eval_data = [ eval_data_only[0] + test_data_only[0] ,  eval_data_only[1] + test_data_only[1]  ] 
        with open(f'./saved_data/{task}/eval_te.pkl', "wb") as f:
            pickle.dump(eval_data, f)
    eval_template = "Instruction: [PROMPT]\n\nInput: [INPUT]\nOutput: [OUTPUT]"
    prompt_gen_template = "I gave a friend a instruction. Based on the instruction they produced " \
                          "the following input-output pairs:\n\n[full_DEMO]\n\nThe instruction was to [APE]" 
    demos_template = "Input: [INPUT]\nOutput: [OUTPUT]"

    base_conf = '../experiments/configs/instruction_induction.yaml'
    conf = {
        'generation': {
            'num_subsamples': 3, #3
            'num_demos': 5, #5
            'num_prompts_per_subsample': 30, #30
            'model': {
                'gpt_config': {
                    # 'model': 'text-ada-001'
                }
            }
        },
        'evaluation': {
            'method': exec_accuracy_evaluator,
            'task': task,
            'num_samples': 'entire', # this is the number of eval data used to estimate the performance for each prompt -- per prompt, randomly sample
            'model': {
                'gpt_config': {
                    # 'model': 'text-ada-001'
                }
            }
        }
    }
    conf = config.update_config(conf, base_conf)

    if args.model == 'meta-llama/Llama-3.1-8B-Instruct':
        pass
    elif args.model == 'meta-llama/Llama-3.3-70B-Instruct':
        total_prompts = ['ICL_'+args.fake_quality] #['None'] # we generate answers based on ICL
    else:
        raise NotImplementedError

    for prompt in tqdm.tqdm(total_prompts):
        model_name = args.model
        if '/' in model_name:
            model_name = model_name.replace('/', '_')
        cache_dir = os.path.join(_settings.GENERATION_FOLDER, f'{model_name}_{args.dataset}_{args.seed}/batch_results/')
        if success_accum is None:
            cache_dir  =  cache_dir + '/' + task + '/' + prompt[:200] + '/'  # only for saving, since file name too long, error
        else:
            cache_dir  =  cache_dir + '/' + task + '/' + prompt[:200] + '/multi_answer_' + str(success_accum) + '/'  # only for saving, since file name too long, error
        
        LLAMA_PATH = _settings.LLAMA_PATH
        DATA_PATH = _settings.DATA_PATH
        os.makedirs(cache_dir, exist_ok=True)

        sequences, len_prompt = get_generations(model_name, batch_size, loaded_model, tokenizer, args, prompt, eval_template, eval_data, demos_template,prompt_gen_data,conf['evaluation'], seed=args.seed, old_sequences=[], llama_path=LLAMA_PATH, data_path=DATA_PATH)
        pd.to_pickle(sequences, os.path.join(cache_dir, f'{len_prompt}.pkl'))
    return True

if __name__ == '__main__':
    tasks = ['antonyms', 'cause_and_effect', 'common_concept', 'diff',  'first_word_letter',
             'informal_to_formal', 'larger_animal', 'letters_list',  'negation', 'num_to_verbal', 'active_to_passive', 'rhymes',
             'second_word_letter', 'sentence_similarity', 'singular_to_plural', 'sentiment','orthography_starts_with','taxonomy_animal',
             'sum', 'synonyms', 'translation_en-de', 'translation_en-es',
             'translation_en-fr', 'word_in_context']
    tasks_mul_answers = ['common_concept', 'rhymes', 'translation_en-de', 'translation_en-es', 'translation_en-fr']

    LLAMA_PATH = _settings.LLAMA_PATH
    loaded_model = AutoModelForCausalLM.from_pretrained(args.model, device_map='auto', torch_dtype="bfloat16", cache_dir=LLAMA_PATH)#.to(cls._device)
    loaded_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=LLAMA_PATH) #, return_token_type_ids=False
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    loaded_model.config.eos_token_id = tokenizer.pad_token_id
    loaded_model.config.pad_token_id = tokenizer.eos_token_id
    for task in tasks:
        if task in tasks_mul_answers:
            # repeat 10 times to get multiple answers
            batch_size = 1000 
            print('task: ', task)
            success_accum = 0
            while success_accum <= 10:
                try:
                    success_indicator = main(task, loaded_model, tokenizer, batch_size, parallel=args.nprocess, success_accum=success_accum)
                    success_accum += int(success_indicator)
                except KeyboardInterrupt:
                    print("KeyboardInterrupt detected. Exiting...")
                    raise  # re-raise it to actually exit
                except:
                    batch_size = batch_size//2
                    print('reducing batch to: ', batch_size)
        else:
            batch_size = 200 #500 #500 #1000 #2000
            print('task: ', task)
            success_indicator = False
            while success_indicator is False:
                # success_indicator = main(task, loaded_model, tokenizer, batch_size, parallel=args.nprocess)
                try:
                    success_indicator = main(task, loaded_model, tokenizer, batch_size, parallel=args.nprocess)
                except KeyboardInterrupt:
                    print("KeyboardInterrupt detected. Exiting...")
                    raise  # re-raise it to actually exit
                except:
                    batch_size = batch_size//2
                    print('reducing batch to: ', batch_size)
                    

