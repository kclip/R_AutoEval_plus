import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json



def fake_quality_to_bitwidth(fake_quality):
    def _mx_to_avg_bit(k1, k2, d1, d2, m):
        return m + d1/k1 + d2/k2 + 1
    if fake_quality == 'true':
        avg_bitwidth = 999
    elif fake_quality == 'best':
        avg_bitwidth = 16
    elif fake_quality in ['good', 'med', 'bad']:
        k1 = 16
        k2 = 2
        d1 = 8
        d2 = 1
        if fake_quality == 'good':
            m = 7
        elif fake_quality == 'med':
            m = 4
        elif fake_quality == 'bad':
            m = 2
        else:
            print(fake_quality)
            raise NotImplementedError
        avg_bitwidth = _mx_to_avg_bit(k1, k2, d1, d2, m)
    else:
        raise NotImplementedError
    return avg_bitwidth

plt.rc('font', family='serif', serif='Computer Modern Roman')
plt.rcParams.update({
    'font.size': 15,
})

plt.rcParams['figure.figsize'] = [20 / 1.1, 10 / 1.1]
fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(20 / 1.1, 10 / 1.1))
colors = ['#B51700','k','#009051', '#0076BA', '#F8BA00', '#FF8D28', '#56C1FF', '#EF5FA7', '#F8BA00', '#009051', 'k', '#A62B17', '#A62B17', '#3274B5', '#3274B5', '#EF5FA7','#EF5FA7']
lines = ['-', '--','-', '-', '-','-', '-', '-','-','-','--', '-', '--', '-', '--']
markers = ['^', 'o', 'v' ]



col = 0
data = 'coqa'
model = 'meta-llama_Llama-3.1-8B-Instruct'
for fake_quality in ['true', 'best', 'med', 'bad']: #['true', 'best', 'good', 'med', 'med2', 'med3', 'mx4+', 'bad']:
    avg_bitwidths = []
    qsnrs = []
    scores = []
    ind_m = 0
    for m in [3,4,5,6,7,8,9,10]:
        color = colors[ind_m]
        for k1 in [16, 64]:
            ind_ratio = 0
            for ratio in [2,4,8]:
                marker = markers[ind_ratio]
                if k1 == 16:
                    config = {"m": m, "ratio": ratio}
                else:
                    config = {"m": m, "ratio": ratio, "k1": 64}

                config_str = json.dumps(config)
                path = './' + fake_quality +  '/' + data + '/' + model + '/' + config_str + '/'
                try:
                    error = False
                    rel_loss = np.load(path+'rel_loss.npy')
                except:
                    try:
                        error = False
                        score_quan = np.load(path+'score.npy')
                        score_full = np.load('./' + fake_quality +  '/' + data + '/' + model + '/full_precision/score.npy')
                        idx_quan = np.load(path+'idx.npy')
                        idx_full = np.load('./' + fake_quality +  '/' + data + '/' + model + '/full_precision/idx.npy' )
                        assert np.array_equal(idx_quan, idx_full) == True

                        rel_loss = np.maximum(score_full - score_quan, 0) # we focus on performance drop due to quantization
                        np.save(path+'rel_loss.npy', rel_loss)
                    except:
                        error = True
                        print(path, 'not exist!')
                        pass
                if not error:
                    actual_spec = np.load(path+'actual_spec.npy')
                    qsnr = actual_spec[0]
                    avg_bitwidth = actual_spec[1]
                    avg_bitwidths.append(avg_bitwidth)
                    qsnrs.append(qsnr)
                    scores.append(np.average(rel_loss))
                    axs[0][col].scatter(avg_bitwidth, np.average(rel_loss), marker=marker, c=color, label=config_str)
                    axs[0][col].set_ylim([0.0, 0.45])
                    # axs[0][col].set_title(fake_quality+ f'bitwidth {fake_quality_to_bitwidth(fake_quality)}')
                    axs[0][col].set_title(fake_quality)
                    # axs[1][col].scatter(avg_bitwidth, np.std(rel_loss), marker=marker, c=color, label=config_str)
                    # axs[1][col].set_ylim([0.0, 0.5])
                    # axs[1][col].set_xlabel('avg. bitwidth')
                    axs[1][col].scatter(avg_bitwidth, qsnr, marker=marker, c=color, label=config_str, alpha=1)
                    axs[0][col].set_xlabel('avg. bitwidth')
                    axs[1][col].set_xlabel('avg. bitwidth')
            ind_ratio += 1
        ind_m += 1        
    col += 1
    
    axs[0][0].set_ylabel('relative loss')
    axs[1][0].set_ylabel('QSNR [dB]')
    
plt.tight_layout()
path = Path('./vis_mx_quant.png', dpi=200)
path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(path)
plt.show()
plt.close(fig)




        