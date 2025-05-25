import numpy as np
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import os

# Get the parent directory
# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.insert(0, parent_dir)

from testing_by_betting import WSR_PPI_p_value, val_p_value_naive
import pickle
from tqdm import tqdm
import pandas as pd
import seaborn as sns


parser = argparse.ArgumentParser(description='LLM_prompt')
parser.add_argument('--model', type=str, default='meta-llama/Llama-3.1-8B-Instruct')
parser.add_argument('--task', type=str, default='larger_animal')
parser.add_argument('--fake_quality', type=str, default='good')
parser.add_argument('--seed', type=int, default=10)
parser.add_argument('--N_n_ratio', type=float, default=9)
parser.add_argument('--N', type=int, default=None)
parser.add_argument('--n', type=int, default=200)
parser.add_argument('--num_rep', type=int, default=100)
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--delta', type=float, default=0.1)
parser.add_argument('--num_prompts', type=int, default=25)
parser.add_argument('--bet_mode', type=str, default='WSR')
args = parser.parse_args()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _split(D, n):
    # split D into n, |D|-n
    return D[:, :n], D[:, n:]

def main(args, task, n, N, scheme, ind_rep):
    
    path = BASE_DIR+'/score_tables/'+args.fake_quality + '/' + task + '/meta-llama_Llama-3.1-8B-Instruct/'
    true_loss_table = 1-np.load(path+'true_score.npy')
    fake_loss_table = 1-np.load(path+'fake_score.npy')
    prompt_lens = np.load(path+'actual_spec.npy')
    prompt_itself = np.load(path+'prompts_itself.npy')
    # choose diverse 25 prompts based on their lengths among 100 (not using any data!)
    A = np.squeeze(prompt_lens)
    sorted_indices = np.argsort(A)
    sorted_indices = sorted_indices[0::prompt_lens.shape[0]//args.num_prompts]
    true_loss_table = true_loss_table[sorted_indices, :]
    fake_loss_table = fake_loss_table[sorted_indices, :]
    prompt_lens = prompt_lens[sorted_indices, :]
    prompt_itself = prompt_itself[sorted_indices]
    
    N_tot = true_loss_table.shape[1]
    num_hyper_cand = true_loss_table.shape[0]

    args.n_te = 100 
    n_te = args.n_te
    rand_path_table = path + f'/permutations/shuffle_{ind_rep}_perm.npy'
    try:
        perm_path = Path(rand_path_table)
        perm = np.load(perm_path)
    except:
        perm = np.random.permutation(N_tot)
        perm_path.parent.mkdir(parents=True, exist_ok=True) 
        np.save(perm_path, perm)
    D_true = true_loss_table[:, perm]
    D_fake = fake_loss_table[:, perm]
    D_test, D_cal_true = _split(D_true, n_te)
    _, D_cal_fake = _split(D_fake, n_te)
    D_n_true, _ = _split(D_cal_true, n) # we only have n labelled data!s
    assert D_n_true.shape[1] == n
    D_n_fake, D_rem_fake = _split(D_cal_fake, n)
    ## get exact N and raise error if not enough
    D_N_fake, _ = _split(D_rem_fake, N)
    try:
        assert D_N_fake.shape[1] == N
    except:
        print('we have', D_cal_true.shape)
        return -1000, -1000, True
    # we first get vector of p-values: |\Lambda| * 1
    delta_Bon = args.delta/num_hyper_cand

    if scheme == 'LTT':
        rhos = np.array([0.0])
        M = 1
        p_values = np.array([WSR_PPI_p_value( D_n_true[i, :], D_n_fake[i, :], D_N_fake[i, :], args.alpha, delta_Bon, rhos, betting_mode=args.bet_mode) for i in range(num_hyper_cand)])
    
    elif 'PPLTT' in scheme:
        if '++' in scheme: # proposed
            M = int(scheme[7:])
            rhos = np.linspace(0, 1, M)
        else: # original
            M = 1
            rhos = np.array([float(scheme[5:])])

        p_values = np.array([WSR_PPI_p_value( D_n_true[i, :], D_n_fake[i, :], D_N_fake[i, :], args.alpha, delta_Bon, rhos, betting_mode=args.bet_mode) for i in range(num_hyper_cand)])
    
    elif 'imputeLTT' in scheme:
        rhos = np.array([0.0])
        if 'v1' in scheme:
            p_values = np.array([impute_p_value_v1( D_n_true[i, :],  D_N_fake[i, :], args.alpha, delta_Bon, rhos) for i in range(num_hyper_cand)])
        elif 'v2' in scheme:
            p_values = np.array([impute_p_value_v2( D_n_true[i, :],  D_N_fake[i, :], args.alpha, delta_Bon, rhos) for i in range(num_hyper_cand)])
        else:
            raise NotImplementedError
    elif 'val' in scheme:
        if 'true' in scheme:
            p_values = np.array([val_p_value_naive( D_n_true[i, :], args.alpha) for i in range(num_hyper_cand)])
        elif 'fake' in scheme:
            p_values = np.array([val_p_value_naive( D_N_fake[i, :], args.alpha) for i in range(num_hyper_cand)])
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    if 'val' in scheme:
        where_delta = np.where(p_values<=args.delta)
    else:
        where_delta = np.where(p_values<=delta_Bon) # Bonferrnoni
    good_prompts = where_delta[0]
    if len(good_prompts) == 0:
        if_fail = True
        good_prompts = [np.argmax(prompt_lens)]
    else:
        if_fail = False
    test_loss_chosen_prompts = D_test[good_prompts, :]
    promt_lens_chosen_prompts = prompt_lens[good_prompts]
    
    shortest_len_ind = np.argmin(prompt_lens[good_prompts])
    return test_loss_chosen_prompts[shortest_len_ind], promt_lens_chosen_prompts[shortest_len_ind],  if_fail
if __name__ == '__main__':



    schemes = ['PPLTT1','LTT','PPLTT++10']
    my_c = {'k': 'k', 'g': '#009051', 'b': '#0076BA', 'r': '#EF5FA7','y': '#F8BA00'}
    colors = [ my_c['b'], my_c['g'], my_c['r'], my_c['r'], my_c['r']]
    lines = ['-', '-','-', '-', '-','-', '-', '-','-','-','--', '-', '--', '-', '--']
    markers = ['p', 'x', 'o',  '_', '+']
    n = args.n
    f_list = ['bad', 'medbad', 'med','medgood', 'good', 'gooder','gooderer'] # 1 2 3 5 all
    f_list_vis = [1,2,3,4,5,6,7]
    sns_rows = []
    np.random.seed(10)

    all_tasks = ['antonyms', 'cause_and_effect', 'common_concept', 'diff',  'first_word_letter',
             'informal_to_formal', 'larger_animal', 'letters_list',  'negation', 'num_to_verbal', 'active_to_passive', 'rhymes',
             'second_word_letter', 'sentence_similarity', 'singular_to_plural', 'sentiment','orthography_starts_with','taxonomy_animal',
             'sum', 'synonyms', 'translation_en-de', 'translation_en-es',
             'translation_en-fr', 'word_in_context']
    
    tasks = []
    for task in all_tasks:
        path = BASE_DIR + '/score_tables/'+args.fake_quality + '/' + task + '/meta-llama_Llama-3.1-8B-Instruct/'
        true_loss_table = 1-np.load(path+'true_score.npy')
        if true_loss_table.shape[1]> 2000:
            tasks.append(task)

    print('testing for ', tasks)
    cols = 6  
    rows = (len(tasks) + cols - 1) // cols
    plt.rc('font', family='serif', serif='Computer Modern Roman')
    plt.rcParams.update({
        'font.size': 13,
        'text.usetex': True,
    })

    fig, axes = plt.subplots(2*rows, cols, figsize=(3 * cols, 3 * 2* rows))
    axes = axes.flatten()

    results_dict_shortest_len = {}
    results_dict_loss = {}
    results_dict_loss_std = {}
    for scheme in schemes: #np.full((num_hyper_cand, N_tot), np.nan)
        results_dict_shortest_len[scheme] = np.full([args.num_rep, len(tasks), len(f_list)], np.nan)
        results_dict_loss[scheme] = np.full([args.num_rep,len(tasks), len(f_list)], np.nan)
        results_dict_loss_std[scheme] = np.full([args.num_rep, len(tasks), len(f_list)], np.nan)
    
    if args.N is None:
        saving_name = './figs/LLM_prompt/' + args.bet_mode + '/' + '/per_f/' + f'/delta_{args.delta}/N_n_ratio_{args.N_n_ratio}/num_prompts_{args.num_prompts}/n_{n}/'
    else:
        saving_name = './figs/LLM_prompt/'+ args.bet_mode + '/' +'/per_f/' + f'/delta_{args.delta}/fixed_N_{args.N}num_prompts_{args.num_prompts}/n_{n}/'
    alpha_info = {}
    ind_task = 0
    for task in tqdm(tasks):
        print(task)
        args.alpha = 0.05
        ref_fail = True
        while ref_fail:
            try:
                for ind_rep in tqdm(range(args.num_rep)):
                    ind_scheme = 0
                    for scheme in schemes:
                        actual_n_list = []
                        tmp_bit_list = []
                        tmp_te_list = []
                        ind_f = 0
                        for fake_quality in f_list:
                            args.fake_quality = fake_quality 
                            if scheme == 'LTT' or scheme == 'val_true':
                                N = 0
                            else:
                                if args.N is None:
                                    N = int(np.ceil(n * args.N_n_ratio))
                                else:
                                    N = args.N
                            if scheme == 'val_fake':
                                n_actual = 0
                            else:
                                n_actual = n
                            if scheme == 'LTT' and fake_quality != 'bad': # LTT has no dependency on autoevaluator quality
                                mean_loss = results_dict_loss[scheme][ind_rep, ind_task, 0]
                                std_loss = results_dict_loss_std[scheme][ind_rep, ind_task, 0]
                            else:   
                                test_loss_shotest_len, shortest_len, if_fail = main(args, task, n, N, scheme, ind_rep)
                                if scheme == 'PPLTT1' and fake_quality == 'gooderer':
                                    if if_fail:
                                        if args.alpha < 0.95:
                                            ref_fail = True
                                            raise ValueError(f"PPLTT fail at {args.alpha}, increase alpha")
                                        else:
                                            ref_fail = False
                                    else:
                                        ref_fail = False
                                mean_loss = np.average(test_loss_shotest_len)
                                std_loss = np.std(test_loss_shotest_len)              
                            results_dict_shortest_len[scheme][ind_rep, ind_task, ind_f] = shortest_len
                            results_dict_loss[scheme][ind_rep, ind_task, ind_f] = mean_loss
                            results_dict_loss_std[scheme][ind_rep, ind_task, ind_f] = std_loss
                            alpha_info[task] = args.alpha
                            ind_f += 1
                        ind_scheme += 1
            except Exception as e:
                print(e)
                args.alpha += 0.05
        avg_alpha = np.array(list(alpha_info.values())).mean()
        ax = axes[ind_task]
        ax.set_title(task)

        ax_loss = axes[ind_task+cols*rows]
        ax_loss.set_title(task)
        ind_scheme = 0
        for scheme in schemes:
            bit_mean = np.mean(results_dict_shortest_len[scheme][:, ind_task, :], axis=0)
            bit_std = np.std(results_dict_shortest_len[scheme][:, ind_task, :], axis=0)*1.96/np.sqrt(args.num_rep)
            test_mean = np.mean(results_dict_loss[scheme][:, ind_task, :], axis=0)
            test_std = np.mean(results_dict_loss_std[scheme][:, ind_task, :], axis=0)*1.96/np.sqrt(args.num_rep)

            ax.plot(f_list_vis, bit_mean, marker=markers[ind_scheme], markersize=4, c=colors[ind_scheme], label=scheme)
            ax.fill_between(f_list_vis, bit_mean-bit_std, bit_mean+bit_std,  facecolor=colors[ind_scheme], alpha=0.2)
            ax.set_xlabel(r'\shortstack{autoevaluator prompt size}')
            ax.set_ylabel(r'\shortstack{shortest \\ prompt length}')
            ax_loss.plot(f_list_vis, test_mean, marker=markers[ind_scheme], markersize=4, c=colors[ind_scheme], label=scheme)
            ax_loss.fill_between(f_list_vis, test_mean-test_std, test_mean+test_std,  facecolor=colors[ind_scheme], alpha=0.2)
            ax_loss.axhline(alpha_info[task], c='y', linestyle='--')
            ax_loss.set_xlabel(r'\shortstack{autoevaluator prompt size}')
            ax_loss.set_ylabel(r'$1$- execution accuracy')
            ind_scheme += 1
        plt.tight_layout()
        path = Path(saving_name + str(alpha_info[task]) +task+'.png')
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=200)
        ind_task += 1
    results_dict_list = [results_dict_shortest_len, results_dict_loss, results_dict_loss_std]
    path_dict = Path(saving_name +f'num_rep_{args.num_rep}.pkl')
    path_dict.parent.mkdir(parents=True, exist_ok=True) 
    with open(path_dict, 'wb') as f:
        pickle.dump(results_dict_list, f)
    plt.rcParams['figure.figsize'] = [6/ 1.1, 5 / 1.1]
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(2.5/ 1.1, 5/ 1.1))
    ind_scheme = 0
    for scheme in schemes:
        bit_mean = np.nanmean(results_dict_shortest_len[scheme], axis=(0,1))
        bit_std = np.nanmean(np.std(results_dict_shortest_len[scheme], axis=0)*1.96/np.sqrt(args.num_rep), axis=0)
        test_mean = np.nanmean(results_dict_loss[scheme], axis=(0,1))
        test_std = np.nanmean(np.mean(results_dict_loss_std[scheme], axis=0)*1.96/np.sqrt(args.n_te), axis=0) # take the averaged std associated with testing number -- around 1000
        axs[0].plot(f_list_vis, bit_mean, marker=markers[ind_scheme], markersize=2, c=colors[ind_scheme], label=scheme)
        axs[0].fill_between(f_list_vis, bit_mean-bit_std, bit_mean+bit_std,  facecolor=colors[ind_scheme], alpha=0.2)
        
        axs[1].plot(f_list_vis, test_mean, marker=markers[ind_scheme], markersize=2, c=colors[ind_scheme], label=scheme)
        axs[1].fill_between(f_list_vis, test_mean-test_std, test_mean+test_std,  facecolor=colors[ind_scheme], alpha=0.2)

        axs[1].plot(f_list_vis, test_mean, marker=markers[ind_scheme], markersize=2, c=colors[ind_scheme], label=scheme)
        ind_scheme += 1


    axs[0].set_ylabel(r'\shortstack{shortest \\ prompt length}')
    axs[1].set_ylabel(r'$1$- execution accuracy')
    axs[0].set_xlabel(r'autoevaluator quality (in-context samples)')
    axs[0].set_xticks([1,4,7])
    axs[1].set_xticks([1,4,7])
    axs[0].set_xlabel(r'\shortstack{autoevaluator \\ prompt size}')
    axs[1].axhline(avg_alpha, c='y', linestyle='--')
    # axs[1].legend()
    plt.tight_layout()
    path = Path(saving_name+'over_the_tasks.png')
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.show()
    plt.close(fig)



    

    