import logging
from dataclasses import dataclass, field
from pathlib import Path

import tyro
from datasets import Dataset, load_dataset

from generative_recommenders.model import GenerativeRecommendationModelConfig
from generative_recommenders.trainer import (
    GenerativeRecommendationTrainer,
    GenerativeRecommendationTrainerConfig,
)


DEFAULT_MODEL_CFG = GenerativeRecommendationModelConfig(
    d_model=256,
    n_layers=4,
    n_heads=8,
    n_kv_heads=4,
    head_dim=32,
    ffn_hidden_dim=1024,
    max_seq_len=256,
    rope_base=10000.0,
    rms_norm_eps=1e-5,
    dropout=0.1,
    bias=False,
    n_post_embeddings=1 << 20,
    n_author_embeddings=1 << 18,
    n_engagements=8,
    n_id_hashes=2,
    engagement_loss_weight=1.0,
    engagement_priors=(0.05,) * 8,
)

DEFAULT_TRAINER_CFG = GenerativeRecommendationTrainerConfig(
    learning_rate=3e-4,
    batch_size=32,
    n_epochs=1,
    weight_decay=0.0,
    grad_clip=1.0,
    num_workers=4,
    log_interval=10,
    checkpoint_dir=None,
    checkpoint_interval=1000,
    val_interval=500,
)


@dataclass
class TrainArgs:
    train_dataset: Path
    val_dataset: Path
    model_cfg: GenerativeRecommendationModelConfig = field(default_factory=lambda: DEFAULT_MODEL_CFG)
    trainer_cfg: GenerativeRecommendationTrainerConfig = field(default_factory=lambda: DEFAULT_TRAINER_CFG)


def load_parquet(path: Path) -> Dataset:
    if path.is_dir():
        data_files = sorted(str(p) for p in path.rglob("*.parquet"))
        assert data_files, f"No parquet shards under {path}"
    else:
        data_files = str(path)
    return load_dataset("parquet", data_files=data_files, split="train").with_format("torch")


def main() -> None:
    args = tyro.cli(TrainArgs)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    GenerativeRecommendationTrainer(
        args.model_cfg,
        args.trainer_cfg,
        load_parquet(args.train_dataset),  # type: ignore[arg-type]
        load_parquet(args.val_dataset),  # type: ignore[arg-type]
    ).train()


if __name__ == "__main__":
    main()
