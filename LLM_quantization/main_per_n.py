import numpy as np
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
from testing_by_betting import WSR_PPI_p_value, val_p_value_naive
import pickle
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import os


parser = argparse.ArgumentParser(description='LLM_quantization')
parser.add_argument('--fake_quality', type=str, default='good')
parser.add_argument('--N_n_ratio', type=float, default=5)
parser.add_argument('--N', type=int, default=None)
parser.add_argument('--num_rep', type=int, default=100)
parser.add_argument('--dataset', type=str, default='coqa')
parser.add_argument('--risk_mode', type=str, default='rel') 
parser.add_argument('--bet_mode', type=str, default='UP') 
parser.add_argument('--alpha', type=float, default=None)
parser.add_argument('--delta', type=float, default=0.2)

args = parser.parse_args()
def avg_bitwidth_to_gb(num_param_in_B, avg_bitwidth):
    return num_param_in_B*avg_bitwidth/(8)
    

def _split(D, n):
    return D[:, :n], D[:, n:]

def _FST(p_values, delta):
    where_delta = np.where(p_values>delta)
    if where_delta[0].size == 0: # all pass
        if np.any(np.isnan(p_values)):
            return np.nan
        else:
            return p_values.shape[0]-1
    else:
        return where_delta[0][0]-1

def main(args, n, N, ind_rep, scheme):
    # load loss_entire_table
    if args.dataset == 'triviaqa':
        N_tot = 9960
    elif args.dataset == 'coqa':
        N_tot = 7983
    else:
        raise NotImplementedError
    if args.dataset == 'triviaqa':
        args.n_te = 960
        n_te = args.n_te
    elif args.dataset == 'coqa':
        args.n_te = 983
        n_te = args.n_te
    else:
        raise NotImplementedError
    
    cand_bitwidhts = [10,9,8,7,6,5,4,3]
    cand_mx_ratios = [8,4,2]
    cand_k1 = [16, 64]
    num_hyper_cand = len(cand_bitwidhts)*len(cand_mx_ratios)*len(cand_k1)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    try:
        path_table = BASE_DIR+'/loss_tables_post_process/' + args.risk_mode + '/'  + args.dataset + '/' + args.fake_quality + '/'
        true_loss_table = np.load(path_table + 'true_loss_table.npy')
        fake_loss_table = np.load(path_table + 'fake_loss_table.npy')
        spec_vec = np.load(path_table + 'spec.npy')
    except:
        true_loss_table = np.full((num_hyper_cand, N_tot), np.nan)
        fake_loss_table = np.full((num_hyper_cand, N_tot), np.nan)
        spec_vec = np.full((num_hyper_cand,1), np.nan)

        idx_ref = None
        ind_hyper = 0
        for bitwidth in cand_bitwidhts:
            for k1 in cand_k1:
                for mx_ratio in cand_mx_ratios:
                    if k1 == 16:
                        hyper_setting = json.dumps({"m": bitwidth, "ratio": mx_ratio})
                    elif k1 == 64:
                        hyper_setting = json.dumps({"m": bitwidth, "ratio": mx_ratio, "k1": k1})
                    else:
                        raise NotImplementedError
                    path_dir = BASE_DIR+f'/loss_tables/num_sample_{N_tot}/' + 'true' + '/' + args.dataset + '/meta-llama_Llama-3.1-8B-Instruct/' + hyper_setting + '/'
                    if idx_ref is None:
                        idx_ref = np.load(path_dir + 'idx.npy')
                    else:
                        idx_curr = np.load(path_dir + 'idx.npy')
                        assert np.array_equal(idx_ref, idx_curr) == True
                    if args.risk_mode == 'rel':
                        true_loss_vec = np.load(path_dir + 'rel_loss.npy')
                    elif args.risk_mode == 'abs':
                        true_loss_vec = 1- np.load(path_dir + 'score.npy')
                    else:
                        raise NotImplementedError

                    true_loss_table[ind_hyper, :] = np.squeeze(true_loss_vec)

                    fake_path_dir = BASE_DIR+f'/loss_tables/num_sample_{N_tot}/' + args.fake_quality + '/' + args.dataset + '/meta-llama_Llama-3.1-8B-Instruct/' + hyper_setting + '/'
                    assert np.array_equal(np.load(fake_path_dir + 'idx.npy'), idx_ref) == True
                    if args.risk_mode == 'rel':
                        fake_loss_vec = np.load(fake_path_dir + 'rel_loss.npy')
                    elif args.risk_mode == 'abs':
                        fake_loss_vec = 1-np.load(fake_path_dir + 'score.npy')
                    else:
                        raise NotImplementedError

                    fake_loss_table[ind_hyper, :] = np.squeeze(fake_loss_vec)
                    if ind_hyper == 0:
                        _, prev_avg_bit = np.load(path_dir + 'actual_spec.npy')
                        spec_vec[ind_hyper] = prev_avg_bit
                    else:
                        _, curr_avg_bit = np.load(path_dir + 'actual_spec.npy')
                        spec_vec[ind_hyper] = curr_avg_bit
                        assert curr_avg_bit < prev_avg_bit # fixed sequence testing
                        prev_avg_bit = curr_avg_bit
                    ind_hyper += 1
        path_table = BASE_DIR+'/loss_tables_post_process/' + args.risk_mode + '/' + args.dataset + '/' + args.fake_quality + '/'
        true_loss_table_path = Path(path_table + 'true_loss_table.npy')
        true_loss_table_path.parent.mkdir(parents=True, exist_ok=True) 
        np.save(true_loss_table_path, true_loss_table)
        fake_loss_table_path = Path(path_table + 'fake_loss_table.npy')
        np.save(fake_loss_table_path, fake_loss_table)
        spec_vec_path = Path(path_table + 'spec.npy')
        np.save(spec_vec_path, spec_vec)
    # shuffle 
    try:
        perm_path = Path(BASE_DIR+'/loss_tables_post_process/'  + args.risk_mode + '/' + args.dataset + '/permutations/' + f'shuffle_{ind_rep}_perm.npy')
        perm = np.load(perm_path)
    except:
        perm = np.random.permutation(N_tot)
        perm_path.parent.mkdir(parents=True, exist_ok=True) 
        np.save(perm_path, perm)
    D_true = true_loss_table[:, perm]
    D_fake = fake_loss_table[:, perm]
    ## split test
    D_test, D_cal_true = _split(D_true, n_te)
    _, D_cal_fake = _split(D_fake, n_te)
    ## split into labelled (n) and unlabelled (N)
    D_n_true, _ = _split(D_cal_true, n) # we only have n labelled data!s
    D_n_fake, D_rem_fake = _split(D_cal_fake, n)
    ## get exact N and raise error if not enough
    D_N_fake, _ = _split(D_rem_fake, N)
    assert D_N_fake.shape[1] == N

    # we first get vector of p-values: |\Lambda| * 1
    if scheme == 'LTT':
        rhos = np.array([0.0])
        M = 1
        p_values = np.array([WSR_PPI_p_value( D_n_true[i, :], D_n_fake[i, :], D_N_fake[i, :], args.alpha, args.delta, rhos, betting_mode=args.bet_mode) for i in range(num_hyper_cand)])
    elif 'PPLTT' in scheme:
        if '++' in scheme: # proposed
            M = int(scheme[7:])
            rhos = np.linspace(0, 1, M)
        else: # original
            M = 1
            rhos = np.array([float(scheme[5:])])
        p_values = np.array([WSR_PPI_p_value( D_n_true[i, :], D_n_fake[i, :], D_N_fake[i, :], args.alpha, args.delta, rhos, betting_mode=args.bet_mode) for i in range(num_hyper_cand)])
    elif 'val' in scheme:
        if 'true' in scheme:
            p_values = np.array([val_p_value_naive( D_n_true[i, :], args.alpha) for i in range(num_hyper_cand)])
        elif 'fake' in scheme:
            p_values = np.array([val_p_value_naive( D_N_fake[i, :], args.alpha) for i in range(num_hyper_cand)])
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    # we consider fixed-sequence-testing for MHT
    if 'val' in scheme:
        where_delta = np.where(p_values<=args.delta)
        best_hyper_idx = where_delta[0][-1]
    else:
        best_hyper_idx  = _FST(p_values, args.delta)
    if best_hyper_idx == -1:
        best_hyper_idx = 0 # we choose the most conservative
    elif np.isnan(best_hyper_idx):
        best_hyper_idx = np.nan
    else:
        pass
    if np.isnan(best_hyper_idx):
        return np.nan, np.nan
    else:
        best_avg_bitwidth = spec_vec[best_hyper_idx]
        return best_avg_bitwidth, np.mean(D_test[best_hyper_idx, :]), np.std(D_test[best_hyper_idx, :]), D_test[best_hyper_idx, :]

if __name__ == '__main__':
    schemes = ['LTT', 'PPLTT1', 'PPLTT++10']
    plt.rc('font', family='serif', serif='Times New Roman')
    plt.rcParams.update({
        'font.size': 13,
        'text.usetex': True,
    })
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(2/ 1.1, 4 / 1.1))
    my_c = {'k': 'k', 'g': '#009051', 'b': '#0076BA', 'r': '#EF5FA7','y': '#F8BA00'}
    colors = [my_c['g'], my_c['b'], my_c['r'], my_c['r'], my_c['r']]
    lines = [':','-', '-','--', '-.', '-', '-','-','-','--', '-', '--', '-', '--']
    markers = ['x','v', 'o', 'o', '.', '.', '.', '.', 'o','.','.', 'o', '^', 'v', '.', 'v', '^', 'o']
    n_list = np.linspace(100,300,11) 
    results_dict_bit = {}
    results_dict_loss = {}
    results_dict_loss_std = {}
    for scheme in schemes: 
        results_dict_bit[scheme] = np.full([args.num_rep, len(n_list)], np.nan)
        results_dict_loss[scheme] = np.full([args.num_rep, len(n_list)], np.nan)
        results_dict_loss_std[scheme] = np.full([args.num_rep, len(n_list)], np.nan)
    ind_n = 0
    for n in n_list:
        n = int(n)
        print('n', n)
        if args.N is None:
            saving_name = './figs/LLM_quant/' + args.bet_mode +'/' + args.risk_mode + '/' + args.dataset + '/' + str(n)  + '/' + f'alpha_{args.alpha}/delta_{args.delta}/N_n_ratio_{args.N_n_ratio}/'+args.fake_quality
        else:
            saving_name = './figs/LLM_quant/' + args.bet_mode +'/'+ args.risk_mode + '/' + args.dataset + '/' + str(n) + '/' + f'alpha_{args.alpha}/delta_{args.delta}/fixed_N_{args.N}/'+args.fake_quality
            np.random.seed(10)
        for ind_rep in tqdm(range(args.num_rep)):
            ind_scheme = 0
            for scheme in schemes:
                actual_n_list = []
                tmp_bit_list = []
                tmp_te_list = []
                try:
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
                    avg_bitwidth, test_loss_mean, test_loss_std, test_loss = main(args, n_actual, N, ind_rep, scheme)
                    results_dict_bit[scheme][ind_rep, ind_n] = avg_bitwidth
                    results_dict_loss[scheme][ind_rep, ind_n] = test_loss_mean
                    results_dict_loss_std[scheme][ind_rep, ind_n] = test_loss_std
                    avg_bitwidth = avg_bitwidth_to_gb(8.03, avg_bitwidth)
                except KeyboardInterrupt:
                    print("\n Keyboard Interrupt.")
                    raise NotImplementedError
                except:
                    pass
        ind_n += 1

    ind_scheme = 0
    for scheme in schemes:
        bit_mean = np.mean(results_dict_bit[scheme], axis=0)
        bit_std = np.std(results_dict_bit[scheme], axis=0)*1.96/np.sqrt(args.num_rep)

        test_mean = np.mean(results_dict_loss[scheme], axis=0)
        test_std = np.std(results_dict_loss_std[scheme], axis=0)*1.96/np.sqrt(args.n_te) # take the averaged std associated with testing number -- around 1000
        
        axs[0].plot(n_list, bit_mean, marker=markers[ind_scheme], markersize=2, c=colors[ind_scheme], linestyle=lines[ind_scheme], label=scheme)
        axs[0].fill_between(n_list, bit_mean-bit_std, bit_mean+bit_std,  facecolor=colors[ind_scheme], alpha=0.2)
        
        axs[1].plot(n_list, test_mean, marker=markers[ind_scheme], markersize=2, c=colors[ind_scheme], linestyle=lines[ind_scheme], label=scheme)
        axs[1].fill_between(n_list, test_mean-test_std, test_mean+test_std,  facecolor=colors[ind_scheme], alpha=0.2)

        axs[1].plot(n_list, test_mean, marker=markers[ind_scheme], markersize=2, c=colors[ind_scheme],linestyle=lines[ind_scheme], label=scheme)
        ind_scheme += 1
    axs[0].set_xscale('linear')
    axs[1].set_xscale('linear')
    # axs[0].set_xticks([100,300])
    # axs[0].set_xticklabels(['100', '300'])
    axs[0].set_ylabel(r'model size [GB]')
    axs[1].set_ylabel(r'test loss')
    axs[0].set_xlabel(r'human labeled data size $(n)$')
    # axs[0].set_ylim([5.1,6.3])
    axs[1].axhline(args.alpha, c='orange', linestyle='--')
    plt.tight_layout()
    path = Path(saving_name+'.png')
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=300)
    plt.show()
    plt.close(fig)
    results_all = [results_dict_bit, results_dict_loss, results_dict_loss_std]
    path_dict = Path(saving_name +f'num_rep_{args.num_rep}.pkl')
    path_dict.parent.mkdir(parents=True, exist_ok=True) 
    with open(path_dict, 'wb') as f:
        pickle.dump(results_all, f)
        