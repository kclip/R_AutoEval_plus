import numpy as np
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import os
# from testing_by_betting import testing_by_betting
from testing_by_betting import WSR_PPI_p_value, val_p_value_naive
import pickle
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib as mpl




parser = argparse.ArgumentParser(description='LLM_quantization')
parser.add_argument('--fake_quality', type=str, default='good')
parser.add_argument('--N_n_ratio', type=float, default=5)
parser.add_argument('--N', type=int, default=None)
parser.add_argument('--num_rep', type=int, default=500)
parser.add_argument('--dataset', type=str, default='coqa')
parser.add_argument('--risk_mode', type=str, default='rel')
parser.add_argument('--alpha', type=float, default=None)
parser.add_argument('--delta', type=float, default=0.2)
parser.add_argument('--bet_mode', type=str, default='UP') 


args = parser.parse_args()
def avg_bitwidth_to_gb(num_param_in_B, avg_bitwidth):
    #8.03B
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

    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
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
        perm_path = Path(BASE_DIR+'/loss_tables_post_process/'  + args.risk_mode + '/' + args.dataset + '/permutations/' + f'shuffle_{ind_rep}.perm.npy')
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
    schemes = ['LTT', 'PPLTT1', 'PPLTT++10','val_true', 'val_fake'] 
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
    mpl.rcParams['font.size'] = 12
    n_list = [150]
    for n in n_list:
        if args.N is None:
            saving_name = './figs/LLM_quant/' + args.bet_mode + '/' + args.risk_mode + '/' + args.dataset + '/' + str(n)  + '/' + f'alpha_{args.alpha}/delta_{args.delta}/N_n_ratio_{args.N_n_ratio}/'+args.fake_quality
        else:
            saving_name = './figs/LLM_quant/' + args.bet_mode + '/' + args.risk_mode + '/' + args.dataset + '/' + str(n) + '/' + f'alpha_{args.alpha}/delta_{args.delta}/fixed_N_{args.N}/'+args.fake_quality
        try: 
            with open(saving_name + '/pd.pkl', "rb") as f:
                sns_rows = pickle.load(f)
        except:
            sns_rows = []
            np.random.seed(10)
            y_range_min = 999
            y_range_max = -1
            for ind_rep in tqdm(range(args.num_rep)):
                ind_scheme = 0
                for scheme in schemes:
                    actual_n_list = []
                    tmp_bit_list = []
                    tmp_te_list = []
                    ind_n = 0
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
                        if avg_bitwidth < y_range_min:
                            y_range_min = avg_bitwidth
                        if avg_bitwidth > y_range_max:
                            y_range_max = avg_bitwidth
                        sns_rows.append({'scheme': scheme, 'metric': 'loss', 'value': float(test_loss_mean)})
                        avg_bitwidth = avg_bitwidth_to_gb(8.03, avg_bitwidth) # Llama-3.1-8B-Instruct
                        sns_rows.append({'scheme': scheme, 'metric': 'memory [GB]', 'value': float(avg_bitwidth)})
                    except KeyboardInterrupt:
                        print("\n Keyboard Interrupt .")
                        raise NotImplementedErrors
                    except:
                        pass
        df = pd.DataFrame(sns_rows)
        df['scheme'] = pd.Categorical(df['scheme'], categories=schemes, ordered=True)
        df['metric'] = pd.Categorical(df['metric'], categories=['loss', 'memory [GB]'], ordered=True)
        sns.set(style="whitegrid", font_scale=1.2)
        palette = "Set2"
        unique_metrics = df['metric'].unique()
        fig, axes = plt.subplots(1, len(unique_metrics), figsize=(16, 5))
        try:
            ylim_dict = {'loss': [0,args.alpha+0.02], 'memory [GB]': [y_range_min-0.1, y_range_max+0.1]}
        except:
            y_range_min = 7.1
            y_range_max= 9.3
            ylim_dict = {'loss': [0,args.alpha+0.02], 'memory [GB]': [y_range_min-0.1, y_range_max+0.1]}
        hline_dict = {
                    "loss": args.alpha,
                }
        for ax, metric in zip(axes, unique_metrics):
            if metric in hline_dict:
                ax.axhline(
                    y=hline_dict[metric],
                    color="yellow",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.7,
                    label="target"
                )
            data = df[df['metric'] == metric]
            sns.boxplot(
                data=data,
                x="scheme",
                y="value",
                palette=palette,
                ax=ax,
                width=0.6,
                fliersize=3
            )
            if metric in ylim_dict:
                ax.set_ylim(*ylim_dict[metric])        
            ax.set_title(metric)
            ax.set_xlabel("")
            ax.set_ylabel("value" if ax == axes[0] else "")
        fig.suptitle("Box  Plots for Each metric", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.tight_layout()
        path = Path(saving_name+'box.png')
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=200)
        # plt.show()
        plt.close(fig)
        path.parent.mkdir(parents=True, exist_ok=True)
        path = Path(saving_name+'/pd.pkl')
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(saving_name + '/pd.pkl', "wb") as f:
            pickle.dump(sns_rows, f)
