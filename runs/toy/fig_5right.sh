#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR/../.."

python3 -m toy.sample_complexity --bet_mode 'WSR' --num_rep 200

