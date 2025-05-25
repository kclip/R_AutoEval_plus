import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import tqdm
import argparse
from toy.testing_by_betting.testing_ucb import Testing
from scipy.optimize import root_scalar
from matplotlib.ticker import ScalarFormatter

n_max = 3000
R = 0.1

parser = argparse.ArgumentParser(description='toy experiment (sample complexity)')
parser.add_argument('--bet_mode', type=str, default='UP')
parser.add_argument('--num_rep', type=int, default=200)
args = parser.parse_args()


def rv_gen(n, N):
    def _dist(m):
        return np.random.binomial(n=1, p=R, size=m)
    return _dist(n), _dist(N)
def noise_add(X, f_noise):
    flip = np.random.uniform(0,1,size=len(X)) <= f_noise
    return X*(1-flip) + (1-X)*flip
def get_g_star(alpha, R, f_noise, rho):
    assert R <= alpha
    # assume N is sufficiently large
    # first find lambda_star
    if rho == 0:
        D = 0.0 - alpha
        p = [(1-R), R] 
        c = [D, D+1]
    else:
        gamma = 1-f_noise # as per our example description
        D = (R*gamma + (1-R)*(1-gamma))*rho - alpha
        p = [(1-R)*gamma, (1-R)*(1-gamma), R*gamma, R*(1-gamma)]
        c = [D, D-rho, D+1-rho, D+1]
    bet_max = 1/(1+rho-alpha)
    bet_min = 0
    num_grid = 1000
    lambda_grid = np.linspace(bet_min, bet_max, num_grid)
    g_star = sum(  p_i * np.log(1-lambda_grid*c_i )  for p_i, c_i in zip(p, c))
    return np.nanmax(g_star)
    
if __name__ == "__main__":
    alpha= 0.12
    rho_list = np.linspace(0, 1,10)
    f_noise_list = [0.01, 0.1, 0.3]
    plt.rc('font', family='serif', serif='Times New Roman')
    plt.rcParams.update({
        'font.size': 13,
        'text.usetex': True,
    })

    fig_n, axs_n = plt.subplots(nrows=2,ncols=3,figsize=(4.5,5.2))
    my_c = {'k': 'k', 'g': '#009051', 'b': '#0076BA', 'r': '#EF5FA7','y': '#F8BA00'}
    colors = [my_c['g'], my_c['b'], my_c['r'], my_c['r'], my_c['r']]
    lines = ['-','-', '-','--', '-.', '-', '-','-','-','--', '-', '--', '-', '--']
    markers = ['x','v', 'o', 'o', '.', '.', '.', '.', 'o','.','.', 'o', '^', 'v', '.', 'v', '^', 'o']
    r = 10
    num_rep = args.num_rep
    mode_list = ['LTT', 'PPLTT','PPLTT++10']
    delta_list = np.logspace(-1, -3, 4) 
    n_o_dict = {}
    for mode in mode_list:
        n_o_dict[mode] = np.full((num_rep, len(f_noise_list), len(delta_list)), np.nan)
    
    for ind_rep in tqdm.tqdm(range(num_rep)):
        for ind_f in range(len(f_noise_list)):
            for mode in mode_list:
                ind_delta = 0
                for delta in delta_list:
                    if mode == 'LTT' and ind_f != 0:
                        n_o = n_o_dict[mode][ind_rep, 0, ind_delta]
                    else:
                        PP_UCB_LCB = Testing(rv_gen, noise_add, r, delta, f_noise_list[ind_f], betting_mode=args.bet_mode)
                        n_o = PP_UCB_LCB.forward(alpha, mode)
                    n_o_dict[mode][ind_rep, ind_f, ind_delta] = n_o
                    ind_delta += 1

    formatter2 = ScalarFormatter(useMathText=True)
    formatter2.set_powerlimits((-2, -2))  # 10^-3 notation
    formatter2.set_useOffset(False)
    g_list_per_f = []
    for ind_f in range(len(f_noise_list)):
        g_list = [get_g_star(alpha, R, f_noise_list[ind_f], rho) for rho in rho_list]
        g_list_per_f.append(g_list)
        axs_n[1][ind_f].plot(rho_list, g_list, color='k', linestyle='-', marker ='o', markersize=3)
        axs_n[1][ind_f].set_ylim([0.0005,0.011])
        axs_n[1][ind_f].set_xlabel(r'$\rho_s$')
        axs_n[1][ind_f].set_ylabel(r'$g_{s,\star}$')
        axs_n[1][ind_f].yaxis.set_major_formatter(formatter2)
        axs_n[1][ind_f].ticklabel_format(axis='y', style='sci', scilimits=(-2, -2))


    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((4, 4))  # 10^4 notation
    formatter.set_useOffset(False)
    ind_delta = 9
    for ind_f in range(len(f_noise_list)):
        ind_mode = 0
        for mode in mode_list:
            mean = np.mean(n_o_dict[mode][:, ind_f, :ind_delta], axis=0)
            std = 1.96*np.std(n_o_dict[mode][:, ind_f, :ind_delta], axis=0)/np.sqrt(num_rep)
            axs_n[0][ind_f].plot(np.log(1/delta_list[:ind_delta]), mean, color=colors[ind_mode], marker=markers[ind_mode], markersize=2.5, linestyle=lines[ind_mode],label=mode)
            axs_n[0][ind_f].fill_between(np.log(1/delta_list[:ind_delta]), mean-std, mean+std, alpha=0.2, facecolor=colors[ind_mode],linestyle=lines[ind_mode])
            axs_n[0][ind_f].set_xlabel(r'$\log(1/\delta)$')
            axs_n[0][ind_f].set_ylabel(r'$n_{\small \textrm{min}}(\delta)$')
            axs_n[0][ind_f].set_xscale('linear')
            axs_n[0][ind_f].set_ylim([0.0,10000])
            axs_n[0][ind_f].set_xticks([5])
            axs_n[0][ind_f].set_yticks([5000,10000])
            axs_n[0][ind_f].yaxis.set_major_formatter(formatter)
            axs_n[0][ind_f].ticklabel_format(axis='y', style='sci', scilimits=(4, 4))
            ind_mode += 1
    plt.subplots_adjust(hspace=0)
    plt.tight_layout() 
    path = Path('./figs/toy/sample_complexity/' + args.bet_mode +f'alpha_{alpha}_num_rep_{num_rep}.png')
    path.parent.mkdir(parents=True, exist_ok=True) 
    path_dict = Path('./figs/toy/sample_complexity/' + args.bet_mode +f'alpha_{alpha}_num_rep_{num_rep}.pkl')
    path_dict.parent.mkdir(parents=True, exist_ok=True) 
    with open(path_dict, 'wb') as f:
        pickle.dump(n_o_dict, f)
    plt.savefig(path, dpi=300)
    plt.show()
    plt.close(fig_n)