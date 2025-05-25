import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json


plt.rc('font', family='serif', serif='Computer Modern Roman')
plt.rcParams.update({
    'font.size': 15,
})

plt.rcParams['figure.figsize'] = [8 / 1.1, 5 / 1.1]
# fig, axs = plt.subplots(nrows=4,ncols=1,figsize=(3,12))
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8 / 1.1, 15 / 1.1))
# colors = ['#B51700','k','k','#009051', 'k', '#0076BA', '#0076BA', '#F8BA00', '#009051', '#F8BA00', '#009051', 'k', '#A62B17', '#A62B17', '#3274B5', '#3274B5', '#EF5FA7','#EF5FA7']
# lines = ['-', '--',':', '-', '-.','-', '-', '-','-','-','--', '-', '--', '-', '--']
colors = ['#B51700','k','#009051', '#0076BA', '#F8BA00', '#0076BA', '#F8BA00', '#009051', '#F8BA00', '#009051', 'k', '#A62B17', '#A62B17', '#3274B5', '#3274B5', '#EF5FA7','#EF5FA7']
lines = ['-', '--','-', '-', '-','-', '-', '-','-','-','--', '-', '--', '-', '--']


avg_bitwidths = []
qsnrs = []
scores = []

for m in [2,3,4,5,6,7,8,9,10]:
    for ratio in [2,4,8]:
        try:
            config = {"m": m, "ratio": ratio}
            config_str = json.dumps(config)
            path = './' + config_str + '/'
            score = np.load(path+'score.npy')
            actual_spec = np.load(path+'actual_spec.npy')
            qsnr = actual_spec[0]
            avg_bitwidth = actual_spec[1]
            avg_bitwidths.append(avg_bitwidth)
            qsnrs.append(qsnr)
            scores.append(np.average(score))
            axs[0].scatter(avg_bitwidth, np.average(score), marker='o', label=config_str)
            axs[1].scatter(avg_bitwidth, qsnr, marker='o', label=config_str)
        except:
            pass

# unquan = np.load('./full_precision/score.npy')
# axs[0].axhline(np.average(unquan), linestyle='--')
axs[0].set_xlabel('avg. bitwidth')
axs[0].set_ylabel('score')
axs[1].legend()
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
path = Path('./vis.png', dpi=200)
path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(path)
plt.show()
plt.close(fig)




        