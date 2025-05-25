#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR/../.."

for bitwidth in 10 9 8 7 6 5 4 3; do
    for mx_ratio in 2 4 8; do
        mx_spec="{\"m\": $bitwidth, \"ratio\": $mx_ratio}"  
        python3 -m dataeval.generate_true_loss_table --dataset 'triviaqa' --batch_size 20 --model 'meta-llama/Llama-3.1-8B-Instruct' --mx_spec "$mx_spec";
        mx_spec="{\"m\": $bitwidth, \"ratio\": $mx_ratio, \"k1\": 64}"  
        python3 -m dataeval.generate_true_loss_table --dataset 'triviaqa' --batch_size 20 --model 'meta-llama/Llama-3.1-8B-Instruct' --mx_spec "$mx_spec";
    done
done

python3 -m dataeval.generate_true_loss_table --dataset 'triviaqa' --batch_size 20 --model 'meta-llama/Llama-3.1-8B-Instruct' --mx_spec "full_precision";