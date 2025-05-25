#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR/../.."

for bitwidth in 10 9 8 7 6 5 4 3; do
    for mx_ratio in 8 4 2; do
        mx_spec="{\"m\": $bitwidth, \"ratio\": $mx_ratio}"  # default: k1=16
        python3 -m pipeline.generate --dataset 'triviaqa' --batch_size 20 --mx_spec "$mx_spec"  --model 'meta-llama/Llama-3.1-8B-Instruct';
        mx_spec="{\"m\": $bitwidth, \"ratio\": $mx_ratio, \"k1\": 64}"  
        python3 -m pipeline.generate --dataset 'triviaqa' --batch_size 20 --mx_spec "$mx_spec"  --model 'meta-llama/Llama-3.1-8B-Instruct';
    done
done
