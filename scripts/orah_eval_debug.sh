CHECKPOINT="/p01/whan/torchtitan/outputs/orah_pipeline_debug/run.Xy9tehsi/pretrain1/checkpoint/step-100"
ORAH_MODEL=4B python -m torchtitan.experiments.orah.evaluate "$CHECKPOINT" \
  --benchmark cot --limit 5
