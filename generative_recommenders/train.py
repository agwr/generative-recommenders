import logging
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from jaxtyping import Float
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from generative_recommenders.model import (
    GenerativeRecommendationModel,
    GenerativeRecommendationModelConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class GenerativeRecommendationTrainerConfig:
    learning_rate: float
    batch_size: int  # per-device batch size; global batch = batch_size * world_size
    n_epochs: int
    weight_decay: float
    grad_clip: float
    num_workers: int
    log_interval: int
    checkpoint_dir: Path | None  # None disables checkpointing
    checkpoint_interval: int
    val_interval: int


class GenerativeRecommendationTrainer:
    """
    Trains a GenerativeRecommendationModel with Adam on a map-style Dataset. Always runs
    through DDP + DistributedSampler; single-GPU is just world_size=1 (no branching).

    Each dataset item must be a dict with:
        post_ids:    int64 Tensor [seq_len]
        author_ids:  int64 Tensor [seq_len]
        engagements: float Tensor [seq_len, n_engagements]
    All items must share seq_len so the default collate can stack them.

    Launch:
        single-GPU: python train.py
        multi-GPU:  torchrun --nproc_per_node=NGPU train.py
    """

    def __init__(
        self,
        model_cfg: GenerativeRecommendationModelConfig,
        trainer_cfg: GenerativeRecommendationTrainerConfig,
        train_dataset: Dataset,
        val_dataset: Dataset,
    ):
        assert torch.cuda.is_available(), "trainer requires CUDA"
        self.validate_schema(train_dataset, model_cfg)
        self.validate_schema(val_dataset, model_cfg)
        self.cfg = trainer_cfg

        # torchrun sets these env vars; defaults collapse to world_size=1 so the
        # single-GPU path is structurally identical to the multi-GPU one.
        for k, v in {
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "29500",
            "WORLD_SIZE": "1",
            "RANK": "0",
            "LOCAL_RANK": "0",
        }.items():
            os.environ.setdefault(k, v)
        dist.init_process_group(backend="nccl")
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.rank, self.world_size = dist.get_rank(), dist.get_world_size()
        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.device)

        # Place the model on this rank's device, then wrap in DDP for gradient all-reduce.
        self.inner: GenerativeRecommendationModel = GenerativeRecommendationModel(model_cfg).to(self.device)
        self.model = DDP(self.inner, device_ids=[self.local_rank])

        # Train shuffles each epoch via set_epoch; val is shuffle=False so coverage is
        # deterministic and disjointly partitioned across ranks.
        self.train_sampler = DistributedSampler(train_dataset, shuffle=True)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=trainer_cfg.batch_size,
            sampler=self.train_sampler,
            num_workers=trainer_cfg.num_workers,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=trainer_cfg.batch_size,
            sampler=DistributedSampler(val_dataset, shuffle=False),
            num_workers=trainer_cfg.num_workers,
            pin_memory=True,
        )

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=trainer_cfg.learning_rate,
            weight_decay=trainer_cfg.weight_decay,
        )
        self.step_idx = 0

    @staticmethod
    def validate_schema(dataset: Dataset, cfg: GenerativeRecommendationModelConfig) -> None:
        # Each item must be a dict with exactly the three expected keys.
        sample = dataset[0]
        assert isinstance(sample, dict), f"dataset items must be dicts, got {type(sample).__name__}"
        expected = {"post_ids", "author_ids", "engagements"}
        assert set(sample) == expected, f"dataset item keys must be {expected}, got {set(sample)}"
        post_ids, author_ids, engagements = sample["post_ids"], sample["author_ids"], sample["engagements"]

        # ID tensors must be 1D int64, with author_ids one-to-one with post_ids.
        assert post_ids.ndim == 1 and post_ids.dtype == torch.int64, "post_ids must be int64 [seq_len]"
        assert author_ids.shape == post_ids.shape and author_ids.dtype == torch.int64, (
            "author_ids must match post_ids shape and be int64"
        )

        # Engagements are float [seq_len, n_engagements], aligned with the ID sequence.
        (seq_len,) = post_ids.shape
        assert engagements.shape == (seq_len, cfg.n_engagements) and engagements.dtype.is_floating_point, (
            f"engagements must be float [seq_len, {cfg.n_engagements}], got {tuple(engagements.shape)} {engagements.dtype}"
        )

    def train(self) -> None:
        for epoch in range(self.cfg.n_epochs):
            # set_epoch reseeds the sampler so each epoch's shuffle is different
            # and consistent across ranks.
            self.train_sampler.set_epoch(epoch)
            for batch in self.train_loader:
                losses = self.train_step(batch)

                # Rank-0-only logging avoids interleaved output across processes.
                if self.rank == 0 and self.step_idx % self.cfg.log_interval == 0:
                    total, post_ce, eng_bce = losses.tolist()
                    logger.info(
                        "train epoch %d step %d: total=%.4f post_ce=%.4f eng_bce=%.4f",
                        epoch,
                        self.step_idx,
                        total,
                        post_ce,
                        eng_bce,
                    )

                # Checkpoint write is rank-0-gated inside save_checkpoint.
                self.save_checkpoint()

                # All ranks must enter val_step together (collective all-reduce inside);
                # only rank 0 prints the result.
                if self.step_idx > 0 and self.step_idx % self.cfg.val_interval == 0:
                    total, post_ce, eng_bce = self.val_step().tolist()
                    if self.rank == 0:
                        logger.info(
                            "val   epoch %d step %d: total=%.4f post_ce=%.4f eng_bce=%.4f",
                            epoch,
                            self.step_idx,
                            total,
                            post_ce,
                            eng_bce,
                        )
                self.step_idx += 1
        dist.destroy_process_group()

    def train_step(self, batch: dict[str, Tensor]) -> Float[Tensor, "3"]:
        self.model.train()
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

        # Forward through self.model (the DDP wrapper) so the reducer schedules the next
        # backward's gradient all-reduce. Loss math runs on the returned logits.
        post_logits, eng_logits = self.model(
            batch["post_ids"], batch["author_ids"], batch["engagements"], decode_all=True
        )
        loss = self.inner.loss(post_logits, eng_logits, batch["post_ids"], batch["engagements"])

        self.optimizer.zero_grad()
        loss.total.backward()
        if self.cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.optimizer.step()

        # Detach so logging downstream doesn't pin the autograd graph.
        return torch.stack([loss.total, loss.post_ce, loss.engagement_bce]).detach()

    @torch.no_grad()
    def val_step(self) -> Float[Tensor, "3"]:
        self.model.eval()
        totals = torch.zeros(3, device=self.device)

        # Each rank sums losses over its disjoint val shard.
        for batch in self.val_loader:
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            post_logits, eng_logits = self.model(
                batch["post_ids"], batch["author_ids"], batch["engagements"], decode_all=True
            )
            loss = self.inner.loss(post_logits, eng_logits, batch["post_ids"], batch["engagements"])
            totals += torch.stack([loss.total, loss.post_ce, loss.engagement_bce])

        # Sum partial sums across ranks, then divide by the global batch count.
        dist.all_reduce(totals)
        return totals / (len(self.val_loader) * self.world_size)

    def save_checkpoint(self) -> None:
        if (
            self.rank != 0
            or self.cfg.checkpoint_dir is None
            or self.step_idx == 0
            or self.step_idx % self.cfg.checkpoint_interval != 0
        ):
            return

        self.cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.cfg.checkpoint_dir / f"step_{self.step_idx}.pt"
        torch.save(
            {"model": self.inner.state_dict(), "optimizer": self.optimizer.state_dict(), "step": self.step_idx},
            path,
        )
        logger.info("saved checkpoint to %s", path)
