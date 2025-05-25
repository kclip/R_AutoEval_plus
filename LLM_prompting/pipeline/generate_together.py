import random
import functools
import os
from typing import Dict
import argparse
import pandas as pd

import tqdm
import numpy as np
from pathlib import Path

import pickle
import pathlib
from transformers import AutoTokenizer, AutoModelForCausalLM

from experiments.data.instruction_induction.load_data import load_data, tasks
from automatic_prompt_engineer import ape, data, config, template, llm
from experiments.evaluation.instruction_induction.exec_accuracy import exec_accuracy_evaluator
from experiments.evaluation.instruction_induction import utility


parser = argparse.ArgumentParser(description='ePP+_LLM_prompt')
parser.add_argument('--model', type=str, default='meta-llama/Llama-3.1-8B-Instruct')
parser.add_argument('--task', type=str, default='None')
parser.add_argument('--fake_quality', type=str, default='good')
parser.add_argument('--seed', type=int, default=10)
args = parser.parse_args()


DEFAULT_DEVICE = 'cuda:7'

IGNORE_INDEX = -100


def clean_fake_answer(fake_answers):
    strings_to_filter_on = [
            '.', '\n', 'Q:', 'A:', 'question:', 'answer:', 'Question:', 'Answer:', 'Questions:', 'questions:', 'QUESTION:',
            'ANSWER:', 'Input:', 'Output:'
        ]
    fake_answers_cleaned = []
    for cleaned_text in fake_answers:
        for string in strings_to_filter_on:
            if string in cleaned_text:
                cleaned_text = cleaned_text.split(string)[0]
        fake_answers_cleaned.append([cleaned_text.strip()]) 
    return fake_answers_cleaned

if __name__ == '__main__':
    import _settings 
    tasks_mul_answers = ['common_concept', 'rhymes', 'translation_en-de', 'translation_en-es', 'translation_en-fr']

    tasks = ['antonyms', 'cause_and_effect', 'common_concept', 'diff',  'first_word_letter',
             'informal_to_formal', 'larger_animal', 'letters_list',  'negation', 'num_to_verbal', 'active_to_passive', 'rhymes',
             'second_word_letter', 'sentence_similarity', 'singular_to_plural', 'sentiment','orthography_starts_with','taxonomy_animal',
             'sum', 'synonyms', 'translation_en-de', 'translation_en-es',
             'translation_en-fr', 'word_in_context']

    for task in tqdm.tqdm(tasks): 
        device = 'cuda:0'
        model_name = args.model
        if '/' in model_name:
            model_name = model_name.replace('/', '_')
        llama_path = _settings.LLAMA_PATH
        tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=llama_path) #, return_token_type_ids=False
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        num_debug = None
        total_prompts = np.load('./saved_data/'+ task + '/total_cand_prompts.npy')
        print('task', task)
        ## load fake answer
        large_model_name = 'meta-llama_Llama-3.3-70B-Instruct'
        fake_cache_dir_ori = os.path.join(_settings.GENERATION_FOLDER, f'{large_model_name}_None_{args.seed}/batch_results/')
        ICL_prompt = 'ICL_' + args.fake_quality
        if task in tasks_mul_answers:
            for ind_mul in range(10):
                fake_cache_dir  =  fake_cache_dir_ori + '/' + task + '/' + ICL_prompt + '/multi_answer_'+str(ind_mul) 
                fake_cache_dir = Path(fake_cache_dir)
                fake_pkl_files = list(fake_cache_dir.glob('*.pkl'))
                path_f_X_hat = os.path.join(fake_cache_dir, fake_pkl_files[0].name)  
                if ind_mul == 0:
                    fake_sequences = pd.read_pickle(path_f_X_hat) 
                    fake_sequences['cleaned_output']  = clean_fake_answer(fake_sequences['model_outputs'])
                else:
                    fake_sequences_add = pd.read_pickle(path_f_X_hat) 
                    fake_sequences_add['cleaned_output']  = clean_fake_answer(fake_sequences_add['model_outputs'])
                    assert np.array_equal(np.array(fake_sequences['inputs'],dtype=object), np.array(fake_sequences_add['inputs'],dtype=object) ) == True
                    
                    for a, b in zip(fake_sequences['cleaned_output'], fake_sequences_add['cleaned_output']):
                        if b[0] not in a:
                            a.append(b[0])
        else:
            fake_cache_dir  =  fake_cache_dir_ori + '/' + task + '/' + ICL_prompt + '/' 
            fake_cache_dir = Path(fake_cache_dir)
            fake_pkl_files = list(fake_cache_dir.glob('*.pkl'))
            path_f_X_hat = os.path.join(fake_cache_dir, fake_pkl_files[0].name)  
            fake_sequences = pd.read_pickle(path_f_X_hat) 
            fake_sequences['cleaned_output']  = clean_fake_answer(fake_sequences['model_outputs'])
        
        ind_prompt = 0
        for prompt in tqdm.tqdm(total_prompts):
            cache_dir = os.path.join(_settings.GENERATION_FOLDER, f'{model_name}_None_{args.seed}/batch_results/')
            cache_dir  =  cache_dir + '/' + task + '/' + prompt[:200] + '/' 
            cache_dir = Path(cache_dir)
            pkl_files = list(cache_dir.glob('*.pkl'))
            if not pkl_files:
                raise FileNotFoundError("No .pkl files found in the specified folder.")
            else:
                path_Y_hat = os.path.join(cache_dir, pkl_files[0].name)   
                len_prompt = int(Path(pkl_files[0].name).stem)
                assert len_prompt == len(prompt)
            sequences = pd.read_pickle(path_Y_hat)        
            if ind_prompt == 0:
                ref_answers = fake_sequences['answers']
                num_samples = len(ref_answers)
                true_score_table = np.full((len(total_prompts), num_samples), np.nan)
                fake_score_table = np.full((len(total_prompts), num_samples), np.nan)
                prompt_len_table = np.full((len(total_prompts), 1), np.nan)
                prompt_itself_table = []
            else:
                pass
            inputs = sequences['inputs']
            assert np.array_equal(np.array(ref_answers,dtype=object), np.array(sequences['answers'],dtype=object) ) == True
            true_answers = sequences['answers']
            fake_answers = fake_sequences['cleaned_output'] # fake labelling
            model_outputs = sequences['model_outputs']
            metric = utility.TASK_TO_METRIC.get(task, utility.default_metric)
            if metric == 'f1':
                score_fn = utility.get_multi_answer_f1
            elif metric == 'es':
                score_fn = utility.get_multi_answer_exact_set
            elif metric == 'contains':
                score_fn = utility.get_multi_answer_contains
            elif metric == 'em':
                score_fn = utility.get_multi_answer_em

            true_scores = []
            for prediction, ans_ in zip(model_outputs, true_answers):
                true_score = score_fn(prediction, ans_)
                true_scores.append(true_score)
            
            fake_scores = []
            for prediction, ans_ in zip(model_outputs, fake_answers):
                fake_score = score_fn(prediction, ans_)
                fake_scores.append(fake_score)

            true_score_table[ind_prompt, :] = np.array(true_scores)
            fake_score_table[ind_prompt, :] = np.array(fake_scores)
            prompt_len_table[ind_prompt] = len_prompt
            prompt_itself_table.append(prompt)
            # Reshape the scores so that it is num_prompts x num_samples
            ind_prompt += 1
        path_common = f'./score_tables/' + args.fake_quality  + '/' + task + '/' + model_name + '/'
        path_prompts_itself = Path(path_common + 'prompts_itself.npy')
        path_prompts_itself.parent.mkdir(parents=True, exist_ok=True) 
        np.save(path_prompts_itself, np.array(prompt_itself_table))
        
        true_path_score = Path(path_common + 'true_score.npy')
        np.save(true_path_score, true_score_table)

        fake_path_score = Path(path_common + 'fake_score.npy')
        np.save(fake_path_score, fake_score_table)

        path_actual_spec = Path(path_common + 'actual_spec.npy')
        np.save(path_actual_spec, prompt_len_table)
        true_score_table = np.load(true_path_score)
        fake_score_table = np.load(fake_path_score)
        prompt_len_table = np.load(path_actual_spec)
        print('task: ', task, 'prompt len', prompt_len_table, 'true score', np.average(true_score_table, axis=1), 'fake score', np.average(fake_score_table, axis=1))