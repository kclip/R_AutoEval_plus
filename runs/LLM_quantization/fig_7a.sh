#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR/../.."
python3 -m LLM_quantization.main_per_n --num_rep 100 --bet_mode 'WSR' --fake_quality 'best' --dataset 'triviaqa' --N_n_ratio 3 --delta 0.1 --alpha 0.1;
python3 -m LLM_quantization.main_per_n --num_rep 100 --bet_mode 'WSR' --fake_quality 'med' --dataset 'triviaqa' --N_n_ratio 3 --delta 0.1 --alpha 0.1;
python3 -m LLM_quantization.main_per_n --num_rep 100 --bet_mode 'WSR' --fake_quality 'bad' --dataset 'triviaqa' --N_n_ratio 3 --delta 0.1 --alpha 0.1;
