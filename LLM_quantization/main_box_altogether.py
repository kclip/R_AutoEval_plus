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
from matplotlib.patches import PathPatch


parser = argparse.ArgumentParser(description='LLM_quantization')
parser.add_argument('--fake_quality', type=str, default='good')
parser.add_argument('--N_n_ratio', type=float, default=5.0)
parser.add_argument('--N', type=int, default=None)
parser.add_argument('--num_rep', type=int, default=500)
parser.add_argument('--dataset', type=str, default='triviaqa')
parser.add_argument('--risk_mode', type=str, default='rel')
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--delta', type=float, default=0.1)
parser.add_argument('--bet_mode', type=str, default='UP') 
args = parser.parse_args()

if __name__ == '__main__':
    schemes = ['LTT', 'PPLTT1', 'PPLTT++10','val_true', 'val_fake'] #, 'imputeLTTv1', 'imputeLTTv2']
    plt.rc('font', family='serif', serif='Times New Roman')
    plt.rcParams.update({
        'font.size': 15,
        'text.usetex': True,
    })

    n_list = [150]
    n = n_list[0] 
    altogether = []
    altogether = pd.DataFrame(altogether)
    for fake_quality in ['None', 'best', 'med', 'bad']:
        if fake_quality == 'None':
            saving_name = './figs/LLM_quant/' + args.bet_mode + '/'   + args.risk_mode + '/' + args.dataset + '/' + str(n) + '/' + f'alpha_{args.alpha}/delta_{args.delta}/N_n_ratio_{args.N_n_ratio}/'+'best'
        else:
            saving_name = './figs/LLM_quant/' + args.bet_mode + '/'   + args.risk_mode + '/' + args.dataset + '/' + str(n) + '/' + f'alpha_{args.alpha}/delta_{args.delta}/N_n_ratio_{args.N_n_ratio}/'+fake_quality
        with open(saving_name + '/pd.pkl', "rb") as f:
            sns_rows = pickle.load(f)

        df = pd.DataFrame(sns_rows)
        df['scheme'] = pd.Categorical(df['scheme'], categories=schemes, ordered=True)
        df['metric'] = pd.Categorical(df['metric'], categories=['loss', 'memory [GB]'], ordered=True)
        df['scheme']  = fake_quality + df['scheme'].astype(str)
        altogether = pd.concat([altogether, df], ignore_index=True)
    print('df', df)
    sns.set(style="whitegrid", font_scale=1.2, font="Times New Roman")
    my_c = {'k': 'k', 'g': '#009051', 'b': '#0076BA', 'r': '#EF5FA7','y': '#F8BA00'}
    palette = [my_c['k'], my_c['g'], my_c['y'], my_c['b'], my_c['r'], my_c['y'],my_c['b'], my_c['r'], my_c['y'],my_c['b'],my_c['r']]
    unique_metrics = altogether['metric'].unique()
    fig, axes = plt.subplots(1, len(unique_metrics), figsize=(16, 5))
    try:
        ylim_dict = {'loss': [0,args.alpha*100+0.01], 'memory [GB]': [y_range_min-0.1, y_range_max+0.1]}
    except:
        y_range_min = 5.2
        y_range_max= 7.8 
        ylim_dict = {'loss': [0,args.alpha*100+0.2*100], 'memory [GB]': [y_range_min-0.1, y_range_max+0.1]}
    hline_dict = {
                "loss": args.alpha*100,
            }
    label_map = {"Noneval_true": r"na\"ive", 
                    "NoneLTT": r"LTT",
                    "bestval_fake": r"na\"ive (sim. high)",
                    "bestPPLTT1": r"PPLTT (sim. high)", 
                    "bestPPLTT++10": r"PPLTT++ (sim. high)" ,
                    "medval_fake": r"na\"ive (sim. med)",
                    "medPPLTT1": r"PPLTT (sim. med)", 
                    "medPPLTT++10": r"PPLTT++ (sim. med)", 
                    "badval_fake": r"na\"ive (sim. bad)",
                    "badPPLTT1": r"PPLTT (sim. bad)", 
                    "badPPLTT++10": r"PPLTT++ (sim. bad)" }
    altogether['scheme'] = altogether['scheme'].map(label_map)
    desired_scheme_order = [r'na\"ive', r'LTT', r"na\"ive (sim. high)", r"PPLTT (sim. high)", r"PPLTT++ (sim. high)", r"na\"ive (sim. med)",r"PPLTT (sim. med)",r"PPLTT++ (sim. med)",r"na\"ive (sim. bad)",r"PPLTT (sim. bad)",r"PPLTT++ (sim. bad)"]
    grouped = altogether.groupby(['scheme', 'metric'])['value']
    q1 = grouped.quantile(0.25)
    q3 = grouped.quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + 1.5*iqr
    q1 = q1.reset_index()
    q3 = q3.reset_index()
    upper_bound = upper_bound.reset_index(name='ub')
    merged = altogether.merge(upper_bound, on=['scheme', 'metric'])
    whisker = (
        merged[merged['value'] <= merged['ub']]
        .groupby(['scheme', 'metric'])['value']
        .max()
        .reset_index()
    )
    for ax, metric in zip(axes, unique_metrics):
        data = altogether[altogether['metric'] == metric].copy()
        if metric == 'loss':
            data['value'] *= 100
        upper_v = whisker[whisker['metric'] == metric]

        upper_v = whisker[whisker['metric'] == metric].copy()
        if metric == 'loss':
            upper_v['value'] *= 100

        box = sns.boxplot(
            data=data,
            x="scheme",
            y="value",
            palette=palette,
            ax=ax,
            width=0.6,
            order=desired_scheme_order,
            showfliers=True,
            flierprops=dict(marker='o', markersize=2, alpha=0.01),
            showmeans=True, meanprops={'marker':'o','markerfacecolor':'white','markeredgecolor':'black','markersize':'6'},
        )
        alpha_box = 0.8
        alpha_bar = 0.5
        for i, line in enumerate(box.lines):
            if i % 6 < 7: 
                line.set_alpha(alpha_box)
        for patch in box.patches:
            patch.set_alpha(alpha_box)
        if metric == 'loss':
            bar = sns.barplot(data=upper_v, x='scheme', y='value',palette=palette, edgecolor='black', width=0.6, alpha=alpha_bar, ax=ax, order=desired_scheme_order,)
        else:
            bar = sns.barplot(data=upper_v, x='scheme', y='value',palette=palette, edgecolor='black', width=0.6, alpha=alpha_bar, ax=ax, order=desired_scheme_order,)

        hatches = ['//'] * (len(bar.patches))
        hatches_n = [''] * (len(bar.patches))
        for i,thisbar in enumerate(bar.patches):
            if i == len(bar.patches) - 9 or i == len(bar.patches) - 11 or i == len(bar.patches) - 6 or i == len(bar.patches) - 3:
                thisbar.set_hatch('//')
            else:
                thisbar.set_hatch('')

        if metric in hline_dict:
            ax.axhline(
                y=hline_dict[metric],
                color="orange",
                linestyle="--",
                linewidth=3,
                alpha=1,
                label="target"
            )
        if metric in ylim_dict:
            ax.set_ylim(*ylim_dict[metric])        
        ax.set_title(metric)
        ax.set_xlabel("")
        ax.set_ylabel(r"performance drop [\%]" if ax == axes[0] else r"model size [GB]")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45,ha="right")
    fig.suptitle("Box  Plots for Each metric", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.tight_layout()
    path = Path(saving_name+'altogether_box.png')
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.show()
    plt.close(fig)
    path.parent.mkdir(parents=True, exist_ok=True)
    path = Path(saving_name+'/pd.pkl')
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(saving_name + '/pd.pkl', "wb") as f:
        pickle.dump(sns_rows, f)

