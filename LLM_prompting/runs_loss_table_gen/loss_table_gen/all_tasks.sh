#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR/../.."

python3 -m pipeline.generate_together  --fake_quality 'gooderer';
python3 -m pipeline.generate_together  --fake_quality 'gooder';
python3 -m pipeline.generate_together  --fake_quality 'good';
python3 -m pipeline.generate_together  --fake_quality 'medgood';
python3 -m pipeline.generate_together  --fake_quality 'med';
python3 -m pipeline.generate_together  --fake_quality 'medbad';
python3 -m pipeline.generate_together  --fake_quality 'bad';