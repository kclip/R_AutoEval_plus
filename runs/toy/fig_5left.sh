#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR/../.."

python3 -m toy.weight_evolution --bet_mode 'WSR' --num_rep 1  --n_max 10000

