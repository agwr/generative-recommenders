# Generative Recommenders

Research implementation for a decoder-only Transformer which autoregressively predicts a user's next post and per-engagement scores from a user-action sequences.

## Training

Datasets are Parquet shards (single file, glob, or directory) with per-row schema:

| column        | type                  | shape                  |
| ------------- | --------------------- | ---------------------- |
| `post_ids`    | `list<int64>`         | `[T]`                  |
| `author_ids`  | `list<int64>`         | `[T]`                  |
| `engagements` | `list<list<float32>>` | `[T, n_engagements]`   |

Launch on 8 GPUs:

```bash
uv run torchrun --nproc_per_node=8 generative_recommenders/train.py \
    --train-dataset path/to/train_shards/ \
    --val-dataset   path/to/val_shards/ \
    --model-cfg.d-model 512 \
    --model-cfg.n-layers 8 \
    --trainer-cfg.batch-size 64 \
    --trainer-cfg.learning-rate 3e-4 \
    --trainer-cfg.checkpoint-dir checkpoints/
```

Use `uv run generative_recommenders/train.py --help` to see all model and trainer flags.