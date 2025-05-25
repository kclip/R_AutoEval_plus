#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR/../.."

python3 -m pipeline.generate --dataset 'triviaqa' --batch_size 20 --mx_spec "full_precision" --model 'meta-llama/Llama-3.3-70B-Instruct';
python3 -m pipeline.generate --dataset 'triviaqa' --batch_size 20 --mx_spec "mx6" --model 'meta-llama/Llama-3.3-70B-Instruct';
python3 -m pipeline.generate --dataset 'triviaqa' --batch_size 20 --mx_spec "mx4" --model 'meta-llama/Llama-3.3-70B-Instruct';