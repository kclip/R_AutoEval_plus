#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR/../.."

python3 -m LLM_prompting.main_prompt_selection --num_rep 100 --n 200 --bet_mode 'UP' --delta 0.1 --num_prompts 25 --N_n_ratio 9;

