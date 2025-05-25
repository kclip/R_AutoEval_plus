import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json


plt.rc('font', family='serif', serif='Computer Modern Roman')
plt.rcParams.update({
    'font.size': 15,
})


# plt.rcParams.update({
#     'mathtext.fontset': 'custom',
#     'mathtext.rm': 'Times New Roman',
#     'font.family': 'serif',
#     'font.serif': ['Times New Roman'],
#     'font.size': 15,
# })

# plt.rcParams.update({
#     'text.usetex': True,  # Use LaTeX for all text
#     'font.family': 'serif',
#     'font.serif': ['Times New Roman'],  # Set desired serif font
#     'font.size': 15,
# })

plt.rcParams['figure.figsize'] = [15 / 1.1, 10 / 1.1]
# fig, axs = plt.subplots(nrows=4,ncols=1,figsize=(3,12))
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8 / 1.1, 5 / 1.1))
# colors = ['#B51700','k','k','#009051', 'k', '#0076BA', '#0076BA', '#F8BA00', '#009051', '#F8BA00', '#009051', 'k', '#A62B17', '#A62B17', '#3274B5', '#3274B5', '#EF5FA7','#EF5FA7']
# lines = ['-', '--',':', '-', '-.','-', '-', '-','-','-','--', '-', '--', '-', '--']
colors = ['#B51700','k','#009051', '#0076BA', '#F8BA00', '#FF8D28', '#56C1FF', '#EF5FA7', '#F8BA00', '#009051', 'k', '#A62B17', '#A62B17', '#3274B5', '#3274B5', '#EF5FA7','#EF5FA7']
lines = ['-', '--','-', '-', '-','-', '-', '-','-','-','--', '-', '--', '-', '--']
markers = ['^', 'o', 'v' ]

avg_bitwidths = []
qsnrs = []
scores = []

ind_m = 0
for m in [3,4,5,6,7,8,9,10]:
    color = colors[ind_m]
    ind_ratio = 0
    for ratio in [2,4,8]:
        marker = markers[ind_ratio]
        config = {"m": m, "ratio": ratio}
        config_str = json.dumps(config)
        path = './' + config_str + '/'
        try:
            rel_loss = np.load(path+'rel_loss.npy')
        except:
            score_quan = np.load(path+'score.npy')
            score_full = np.load('./full_precision/score.npy')
            idx_quan = np.load(path+'idx.npy')
            idx_full = np.load('./full_precision/idx.npy')
            assert np.array_equal(idx_quan, idx_full) == True

            rel_loss = np.maximum(score_full - score_quan, 0) # we focus on performance drop due to quantization
            np.save(path+'rel_loss.npy', rel_loss)

        actual_spec = np.load(path+'actual_spec.npy')
        qsnr = actual_spec[0]
        avg_bitwidth = actual_spec[1]
        avg_bitwidths.append(avg_bitwidth)
        qsnrs.append(qsnr)
        scores.append(np.average(rel_loss))
        axs[0].scatter(avg_bitwidth, np.average(rel_loss), marker=marker, c=color, label=config_str)
        axs[1].scatter(avg_bitwidth, qsnr, marker=marker, c=color, label=config_str, alpha=0)
        ind_ratio += 1
    ind_m += 1        

# unquan = np.load('./full_precision/score.npy')
# axs[0].axhline(np.average(unquan), linestyle='--')
axs[0].set_xlabel('avg. bitwidth')
axs[0].set_ylabel('score drop')
# axs[1].legend()
axs[1].set_xlabel('avg. bitwidth')
axs[1].set_ylabel('qsnr')
# axs[0].axhline(np.average(unquan), linestyle='--')
# axs[0].plot(avg_bitwidths, scores, marker='o')
# axs[0].set_xlabel('avg. bitwidth')
# axs[0].set_ylabel('score')

# axs[1].plot(avg_bitwidths, qsnrs, marker='o')
# axs[1].set_xlabel('avg. bitwidth')
# axs[1].set_ylabel('qsnr')
plt.tight_layout()
path = Path('./vis_rel.png', dpi=200)
path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(path)
plt.show()
plt.close(fig)




        