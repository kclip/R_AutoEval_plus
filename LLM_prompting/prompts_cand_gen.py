import random

import torch
import sys
import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from automatic_prompt_engineer import ape, data
from experiments.data.instruction_induction.load_data import load_data, tasks
from experiments.evaluation.instruction_induction.exec_accuracy import exec_accuracy_evaluator
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import pickle
import _settings

def run(task, strong_model_name, loaded_model, loaded_tokenizer):
    assert task in tasks, 'Task not found!'
    try:
        with open(f'./saved_data/{task}/prompt_gen_data.pkl', 'rb') as f:
            prompt_gen_data_entire = pickle.load(f)
        with open(f'./saved_data/{task}/eval_data.pkl', 'rb') as f:
            eval_data = pickle.load(f)
        with open(f'./saved_data/{task}/test_data.pkl', 'rb') as f:
            test_data = pickle.load(f)
        with open(f'./saved_data/{task}/prompt_gen_data_sample_single.pkl', 'rb') as f:
            prompt_gen_data = pickle.load(f)
    except:
        induce_data, test_data = load_data('induce', task), load_data('eval', task)
        # Get size of the induce data
        induce_data_size = len(induce_data[0])
        prompt_gen_size = min(int(induce_data_size * 0.5), 100)
        # Induce data is split into prompt_gen_data and eval_data
        prompt_gen_data, eval_data = data.create_split(
            induce_data, prompt_gen_size)
        prompt_gen_data_path = Path(f'./saved_data/{task}/prompt_gen_data.pkl')
        prompt_gen_data_path.parent.mkdir(parents=True, exist_ok=True)

        with open(prompt_gen_data_path, "wb") as f:
            pickle.dump(prompt_gen_data, f)
        with open(f'./saved_data/{task}/eval_data.pkl', "wb") as f:
            pickle.dump(eval_data, f)
        with open(f'./saved_data/{task}/test_data.pkl', "wb") as f:
            pickle.dump(test_data, f)

        #print('prompt_gen_data', prompt_gen_data)
        # Data is in the form input: single item, output: list of items
        # For prompt_gen_data, sample a single item from the output list
        prompt_gen_data = prompt_gen_data[0], [random.sample(output, 1)[0]
                                            for output in prompt_gen_data[1]]
        with open(f'./saved_data/{task}/prompt_gen_data_sample_single.pkl', "wb") as f:
            pickle.dump(prompt_gen_data, f)

    eval_template = "Instruction: [PROMPT]\n\nInput: [INPUT]\nOutput: [OUTPUT]"
    prompt_gen_template = "I gave a friend a instruction. Based on the instruction they produced " \
                          "the following input-output pairs:\n\n[full_DEMO]\n\nThe instruction was to [APE]" 
    demos_template = "Input: [INPUT]\nOutput: [OUTPUT]"

    base_config = '../experiments/configs/instruction_induction.yaml'
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
            'num_samples': min(20, len(eval_data[0])), # this is the number of eval data used to estimate the performance for each prompt -- per prompt, randomly sample
            'model': {
                'gpt_config': {
                    # 'model': 'text-ada-001'
                }
            }
        }
    }

    conf['generation']['model']['gpt_config']['model'] = strong_model_name

    num_tot_prompt = 0
    conf['generation']['num_subsamples'] = 3
    while num_tot_prompt < 100:
        total_prompts = ape.find_prompts(cand_prompts=None, eval_template=eval_template,
                                    prompt_gen_data=prompt_gen_data,
                                    eval_data=None,
                                    conf=conf,
                                    base_conf=base_config,
                                    few_shot_data=prompt_gen_data,
                                    demos_template=demos_template,
                                    prompt_gen_template=prompt_gen_template,
                                    loaded_model=loaded_model, loaded_tokenizer=tokenizer)
        num_tot_prompt = len(total_prompts)
        conf['generation']['num_subsamples'] += 1
    
    total_prompts = random.sample(total_prompts, 100)
    total_prompts = list(set(total_prompts))
    assert len(total_prompts) == 100
    
    os.makedirs('./saved_data/' + task, exist_ok=True)
    print('generated prompts', total_prompts, len(total_prompts))
    total_prompts = np.array(total_prompts)
    np.save('./saved_data/' + task + '/total_cand_prompts.npy', total_prompts)


if __name__ == '__main__':
    
    sub_tasks = ['antonyms', 'cause_and_effect', 'common_concept', 'diff', 'first_word_letter',
             'informal_to_formal', 'larger_animal', 'letters_list', 'taxonomy_animal', 'negation', 'num_to_verbal',
             'active_to_passive', 'singular_to_plural', 'rhymes',
             'second_word_letter', 'sentence_similarity', 'sentiment', 'orthography_starts_with',
             'sum', 'synonyms', 'translation_en-de', 'translation_en-es',
             'translation_en-fr', 'word_in_context']
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    llama_path = _settings.LLAMA_PATH
    strong_model_name = 'meta-llama/Llama-3.3-70B-Instruct'
    loaded_model = AutoModelForCausalLM.from_pretrained(strong_model_name, device_map='auto', torch_dtype="bfloat16", cache_dir=llama_path)#.to(cls._device)
    loaded_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(strong_model_name, cache_dir=llama_path) #, return_token_type_ids=False
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    loaded_model.config.eos_token_id = tokenizer.pad_token_id
    loaded_model.config.pad_token_id = tokenizer.eos_token_id
    
    for task in sub_tasks:
        print('task: ', task)
        run(task, strong_model_name, loaded_model, tokenizer)

