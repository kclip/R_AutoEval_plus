import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import tqdm
import argparse
from toy.testing_by_betting.CS import PP_CS

n_max = 20000

R = 0.1

parser = argparse.ArgumentParser(description='toy experiment (confidence interval)')
parser.add_argument('--bet_mode', type=str, default='WSR')
parser.add_argument('--num_rep', type=int, default=50)
args = parser.parse_args()

def rv_gen(n, N):
    def _dist(m):
        return np.random.binomial(n=1, p=R, size=m)
    tmp = _dist(1000000)
    true_mean = np.mean(tmp)
    return _dist(n), _dist(N), true_mean
def noise_add(X, f_noise):
    if f_noise == -1:
        return np.random.binomial(n=1, p=0.5, size=len(X))
    else:
        flip = np.random.uniform(0,1,size=len(X)) <= f_noise
        return X*(1-flip) + (1-X)*flip


def main(dataset, n, N, delta, num_rep, mode, f_noise):
    size_dict = {}
    fail_dict = {}
    lcb_dict = {}
    ucb_dict = {}
    size_dict[mode] = np.full((num_rep), np.nan)
    fail_dict[mode] = np.full((num_rep), np.nan)
    lcb_dict[mode] = np.full((num_rep), np.nan)
    ucb_dict[mode] = np.full((num_rep), np.nan)
    assert num_rep == 1
    for ind_rep in range(num_rep):
        PP_UCB_LCB = PP_CS(rv_gen, noise_add, n, N, delta, f_noise, labelled_data=dataset['lab'][ind_rep][:n], unlabelled_data=dataset['unl'][ind_rep][:N], betting_mode=args.bet_mode)
        lcb, ucb = PP_UCB_LCB.forward(mode)
        size_dict[mode][ind_rep] = ucb-lcb 
        fail_dict[mode][ind_rep] = (ucb<dataset['true'][ind_rep]) or (lcb>dataset['true'][ind_rep])
        lcb_dict[mode][ind_rep] = lcb
        ucb_dict[mode][ind_rep] = ucb
    avg_size = np.average(size_dict[mode])
    return avg_size, lcb, ucb

if __name__ == "__main__":   
    N_n_ratio = 10
    f_noise_list = [0.01, 0.1, 0.3]
    delta_list = [0.01]
    num_rep = args.num_rep 
    size_dict = {}
    ucb_dict = {}
    lcb_dict = {}
    n_list = [100,200,500,1000,2000,5000,10000,20000]
    ind_delta = 0
    mode_list = ['LTT', 'PPLTT','PPLTT++10']
    for mode in mode_list:
        size_dict[mode] = np.full((num_rep, len(n_list), len(f_noise_list)), np.nan)
        ucb_dict[mode] = np.full((num_rep, len(n_list), len(f_noise_list)), np.nan)
        lcb_dict[mode] = np.full((num_rep, len(n_list), len(f_noise_list)), np.nan)
    target_size = 0.1
    for ind_rep in tqdm.tqdm(range(num_rep)):
        dataset = {}
        dataset['lab'] = []
        dataset['unl'] = []
        dataset['true'] = []
        
        for _ in range(1):
            labelled_data, unlabelled_data, true_mean = rv_gen(n_max, int(n_max*N_n_ratio) )
            dataset['lab'].append(labelled_data)
            dataset['unl'].append(unlabelled_data)
            dataset['true'].append(true_mean)
        for ind_f in range(len(f_noise_list)):
            for mode in mode_list:
                ind_n = 0
                for n in n_list:
                    N = int(n*N_n_ratio)
                    size, lcb, ucb = main(dataset, n, N, delta_list[0], 1, mode, f_noise_list[ind_f])
                    size_dict[mode][ind_rep, ind_n, ind_f] = size
                    ucb_dict[mode][ind_rep, ind_n, ind_f] = ucb
                    lcb_dict[mode][ind_rep, ind_n, ind_f] = lcb
                    ind_n += 1
    plt.rc('font', family='serif', serif='Times New Roman')
    plt.rcParams.update({
        'font.size': 13,
        'text.usetex': True,
    })
    fig_n, axs_n = plt.subplots(nrows=len(f_noise_list),ncols=2,figsize=(4,8))
    my_c = {'k': 'k', 'g': '#009051', 'b': '#0076BA', 'r': '#EF5FA7','y': '#F8BA00'}
    colors = [my_c['g'], my_c['b'], my_c['r'], my_c['r'], my_c['r']]
    lines = [':','-', '-','--', '-.', '-', '-','-','-','--', '-', '--', '-', '--']
    markers = ['x','v', 'o', 'o', '.', '.', '.', '.', 'o','.','.', 'o', '^', 'v', '.', 'v', '^', 'o']

    for ind_f in range(len(f_noise_list)):
        ind_mode = 0
        for mode in mode_list:
            mean = np.mean(size_dict[mode][:, :, ind_f], axis=0)
            std = 1.96*np.std(size_dict[mode][:, :, ind_f], axis=0)/np.sqrt(num_rep)
            axs_n[ind_f][0].plot(np.array(n_list), mean, color=colors[ind_mode], marker=markers[ind_mode], markersize=0, linestyle=lines[ind_mode],label=mode)
            axs_n[ind_f][0].fill_between(np.array(n_list), mean-std, mean+std, alpha=0.2, facecolor=colors[ind_mode],linestyle=lines[ind_mode])
            axs_n[ind_f][0].set_xlabel(r'\shortstack{human-labeled \\ data size ($n$)}')
            axs_n[ind_f][0].set_ylabel(r'\shortstack{confidence interval width}')
            axs_n[ind_f][0].set_xscale('log')
            mean = np.mean(ucb_dict[mode][:, :, ind_f], axis=0)
            std = 1.96*np.std(ucb_dict[mode][:, :, ind_f], axis=0)/np.sqrt(num_rep)
            axs_n[ind_f][1].plot(np.array(n_list), mean, color=colors[ind_mode], marker=markers[ind_mode], markersize=0, linestyle=lines[ind_mode],label=mode)
            axs_n[ind_f][1].fill_between(np.array(n_list), mean-std, mean+std, alpha=0.2, facecolor=colors[ind_mode],linestyle=lines[ind_mode])
            
            mean = np.mean(lcb_dict[mode][:, :, ind_f], axis=0)
            std = 1.96*np.std(lcb_dict[mode][:, :, ind_f], axis=0)/np.sqrt(num_rep)
            axs_n[ind_f][1].plot(np.array(n_list), mean, color=colors[ind_mode], marker=markers[ind_mode], markersize=0, linestyle=lines[ind_mode],label=mode)
            axs_n[ind_f][1].fill_between(np.array(n_list), mean-std, mean+std, alpha=0.2, facecolor=colors[ind_mode],linestyle=lines[ind_mode])
            axs_n[ind_f][1].set_xlabel(r'\shortstack{human-labeled \\ data size ($n$)}')
            axs_n[ind_f][1].set_ylabel(r'\shortstack{confidence interval}')
            axs_n[ind_f][1].set_xscale('log')
            axs_n[ind_f][0].set_ylim([0,0.2])
            axs_n[ind_f][1].set_ylim([0,0.2])
            ind_mode += 1
            axs_n[ind_f][1].axhline(R, c='orange', linestyle=':')
    plt.tight_layout() 
    path = Path('./figs/toy/confidence_interval/' + args.bet_mode +f'delta_{delta_list[0]}_num_rep_{num_rep}.png')
    path.parent.mkdir(parents=True, exist_ok=True) 
    plt.savefig(path, dpi=300)
    plt.show()
    plt.close(fig_n)

    path_dict = Path('./figs/toy/confidence_interval/' + args.bet_mode +f'delta_{delta_list[0]}_num_rep_{num_rep}.pkl')
    path_dict.parent.mkdir(parents=True, exist_ok=True) 
    with open(path_dict, 'wb') as f:
        pickle.dump(size_dict, f)
