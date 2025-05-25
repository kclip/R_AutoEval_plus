import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import tqdm
import argparse
from toy.testing_by_betting.testing_ucb import Testing
from scipy.optimize import root_scalar

R = 0.1

parser = argparse.ArgumentParser(description='toy experiment (weight evolution visualization)')
parser.add_argument('--bet_mode', type=str, default='UP')
parser.add_argument('--num_rep', type=int, default=1)
parser.add_argument('--n_max', type=int, default=10000)
args = parser.parse_args()

def rv_gen(n, N):
    def _dist(m):
        return np.random.binomial(n=1, p=R, size=m)
    return _dist(n), _dist(N)
def noise_add(X, f_noise):
    flip = np.random.uniform(0,1,size=len(X)) <= f_noise
    return X*(1-flip) + (1-X)*flip

if __name__ == "__main__":
    alpha= 0.12
    rho_list = np.linspace(0, 1,10)
    f_noise_list = [0.01, 0.1, 0.3]
    plt.rc('font', family='serif', serif='Times New Roman')
    plt.rcParams.update({
        'font.size': 15,
        'text.usetex': True,
    })
    r = 10
    num_rep = 1
    mode_list = ['PPLTT++100']
    delta = 0.01
    n = args.n_max 
    w_history_dict = {}

    M = int(mode_list[0][7:])
    for ind_f in tqdm.tqdm(range(len(f_noise_list))):
        PP_UCB = Testing(rv_gen, noise_add, r, delta, f_noise_list[ind_f], betting_mode=args.bet_mode, if_vis_weight=True)
        w_history_dict[ind_f] = PP_UCB.forward_for_vis(alpha, mode_list[0], n)
    plt.rc('font', family='serif', serif='Times New Roman')
    plt.rcParams.update({
        'font.size': 11,
        'text.usetex': True,
    })

    fig_n, axs_n = plt.subplots(nrows=1,ncols=len(f_noise_list),figsize=(5.5,1.1))
    my_c = {'k': 'k', 'g': '#009051', 'b': '#0076BA', 'r': '#EF5FA7','y': '#F8BA00'}
    colors = [my_c['g'], my_c['b'], my_c['r'], my_c['r'], my_c['r']]
    lines = [':','-', '-','--', '-.', '-', '-','-','-','--', '-', '--', '-', '--']
    markers = ['x','v', 'o', 'o', '.', '.', '.', '.', 'o','.','.', 'o', '^', 'v', '.', 'v', '^', 'o']

    for ind_f in range(len(f_noise_list)):
        im = axs_n[ind_f].imshow(w_history_dict[ind_f], cmap='Greys', aspect='auto')
        axs_n[ind_f].set_xlabel(r'rounds ($i$)')
        if ind_f == 0:
            axs_n[ind_f].set_ylabel(r'reliance factors $\rho_s$')
        yticks = np.linspace(0, M-1, 3)      
        yticklabels = np.linspace(0, 1, 3) 
        if ind_f == 0:
            axs_n[ind_f].set_yticks(yticks)
            axs_n[ind_f].set_yticklabels([f'{y:.2f}' for y in yticklabels])
        else:
            axs_n[ind_f].set_yticks(yticks)
            axs_n[ind_f].set_yticklabels([f'{y:.2f}' for y in yticklabels])
        axs_n[ind_f].invert_yaxis() 
        axs_n[ind_f].set_ylim([-5, M+4])
        axs_n[ind_f].set_xticks([0,n])
    plt.tight_layout() 
    fig_n.subplots_adjust(right=0.85)
    cbar_ax = fig_n.add_axes([0.88, 0.25, 0.02, 0.65])
    cbar = fig_n.colorbar(im, ax=axs_n.ravel().tolist(), cax=cbar_ax,label='portfolio ($w_i$)')
    path = Path('./figs/toy/vis_evolution/' + args.bet_mode +'heatmap.png')
    path.parent.mkdir(parents=True, exist_ok=True) 
    plt.savefig(path, dpi=300)
    plt.show()
    plt.close(fig_n)
