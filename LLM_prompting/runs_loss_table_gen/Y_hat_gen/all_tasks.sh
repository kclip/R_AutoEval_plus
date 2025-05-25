#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR/../.."

python3 -m pipeline.generate --model 'meta-llama/Llama-3.1-8B-Instruct';
