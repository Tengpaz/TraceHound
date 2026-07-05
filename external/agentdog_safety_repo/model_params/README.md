# Model Parameter Snapshots

This directory stores small, Git-friendly parameter/configuration snapshots only.

Large checkpoints are intentionally excluded from Git:

- Binary SFT 8e-6 / 330 step local archive:
  `E:\summercamp\SAIL\checkpoints\binary_dedup_lr8e-6_steps330_checkpoint-330.tar.gz`
- Server binary SFT checkpoint:
  `/root/autodl-tmp/sft_dedup_binary/outputs/models/binary_dedup_lr8e-6_steps330/checkpoint-330`
- Taxonomy LoRA checkpoint:
  `/root/autodl-tmp/taxonomy_lora/outputs/qwen35_08b_taxonomy_lora_lr8e-6_steps200/checkpoint-200`

Put downloaded model files under `checkpoints/` locally when reproducing runs.
