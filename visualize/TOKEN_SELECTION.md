# Token Selection Visualization

This folder now contains `token_selection.py`, a utility that reproduces the token
selection heuristic from `env/last-vit/conf.py` and renders which ViT patches are
selected when keeping the top *K* tokens.

## Quick start

```bash
cd /2024233235
python3 -m pip install matplotlib  # required once
python3 videollm-online/visualize/token_selection.py \
  --image env/last-vit/sample_vis_2.jpg \
  --k-values 5 10 20 \
  --per-channel-topk 1 \
  --output-dir videollm-online/visualize/output
```

The script will create PNG overlays inside `videollm-online/visualize/output`.
Each file highlights the tokens selected for the corresponding *K* while also
showing the aggregated channel votes as a heatmap.

## Arguments

- `--image`: path to the image to analyse. It will be resized to `224×224`.
- `--k-values`: list of token counts (K) to visualise.
- `--per-channel-topk`: number of tokens taken per channel before aggregating
  counts (defaults to `1`).
- `--device`: computation device (defaults to CUDA if available, otherwise CPU).
- `--output-dir`: directory where the figures will be saved.

The console output also prints the token indices (0-based, row-major) chosen for
each *K*, which you can reuse for downstream analysis or comparisons.

