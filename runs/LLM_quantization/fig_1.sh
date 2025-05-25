#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR/../.."
python3 -m LLM_quantization.main_box --num_rep 500 --bet_mode 'WSR' --fake_quality 'best' --dataset 'triviaqa' --N_n_ratio 5 --delta 0.1 --alpha 0.1;
python3 -m LLM_quantization.main_box --num_rep 500 --bet_mode 'WSR' --fake_quality 'med' --dataset 'triviaqa' --N_n_ratio 5 --delta 0.1 --alpha 0.1;
python3 -m LLM_quantization.main_box --num_rep 500 --bet_mode 'WSR' --fake_quality 'bad' --dataset 'triviaqa' --N_n_ratio 5 --delta 0.1 --alpha 0.1;
python3 -m LLM_quantization.main_box_altogether --num_rep 500 --bet_mode 'WSR'