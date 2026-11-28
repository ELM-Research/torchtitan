CHECKPOINT="/path/to/checkpoint/step-1000"
ORAH_MODEL=4B python -m torchtitan.experiments.orah.evaluate "$CHECKPOINT" \
  --benchmark cot --limit 5
