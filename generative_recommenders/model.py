from dataclasses import dataclass
import torch
import torch.nn.functional as F
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor


def hash_multipliers(n: int) -> Tensor:
    g = torch.Generator().manual_seed(0)
    return torch.randint(-(2**62), 2**62 - 1, (n,), generator=g, dtype=torch.int64) * 2 + 1


@dataclass
class TransformerConfig:
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    ffn_hidden_dim: int
    max_seq_len: int
    rope_base: float
    rms_norm_eps: float
    dropout: float
    bias: bool


class SwiGLU(nn.Module):
    """
    Gated activation for the transformer FFN. The learned multiplicative gate lets the
    layer selectively modulate which hidden units propagate, which empirically beats unary
    activations like ReLU/GELU at matched parameter count.
    """

    def __init__(self, d_in: int, d_hidden: int, bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(d_in, d_hidden, bias=bias)
        self.up_proj = nn.Linear(d_in, d_hidden, bias=bias)

    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_hidden"]:
        return F.silu(self.gate_proj(x)) * self.up_proj(x)


class FeedForwardNetwork(nn.Module):
    """
    Position-wise nonlinear transformation MLP to learn richer feature interactions across
    channels. Uses SwiGLU.
    """

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.swiglu = SwiGLU(cfg.d_model, cfg.ffn_hidden_dim, bias=cfg.bias)
        self.down_proj = nn.Linear(cfg.ffn_hidden_dim, cfg.d_model, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        return self.dropout(self.down_proj(self.swiglu(x)))


class RotaryPositionalEmbedding(nn.Module):
    """
    Pre-computes the cos/sin tables that RoPE uses to rotate (q, k) pairs by absolute
    position, registered as buffers so applying RoPE collapses to a couple of indexed
    multiplies per layer rather than recomputing trig every forward.
    """

    cos: Tensor
    sin: Tensor

    def __init__(self, head_dim: int, max_seq_len: int, base: float):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        freqs = torch.outer(torch.arange(max_seq_len), inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(
        self, seq_len: int
    ) -> tuple[
        Float[Tensor, "seq_len head_dim"],
        Float[Tensor, "seq_len head_dim"],
    ]:
        return self.cos[:seq_len], self.sin[:seq_len]

    @staticmethod
    def apply_rope(
        q: Float[Tensor, "batch n_heads seq_len head_dim"],
        k: Float[Tensor, "batch n_kv_heads seq_len head_dim"],
        cos: Float[Tensor, "seq_len head_dim"],
        sin: Float[Tensor, "seq_len head_dim"],
    ) -> tuple[
        Float[Tensor, "batch n_heads seq_len head_dim"],
        Float[Tensor, "batch n_kv_heads seq_len head_dim"],
    ]:
        def rotate_half(
            x: Float[Tensor, "... head_dim"],
        ) -> Float[Tensor, "... head_dim"]:
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat([-x2, x1], dim=-1)

        cos, sin = cos.to(q.dtype), sin.to(q.dtype)
        return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class GroupedQueryAttention(nn.Module):
    """
    Self-attention with separate query and key/value head counts, so each KV head is
    shared across n_heads / n_kv_heads query heads. QK norm on the per-head q/k tensors
    keeps attention logits well-scaled and stabilizes training at depth.
    """

    def __init__(self, cfg: TransformerConfig, rope: RotaryPositionalEmbedding):
        super().__init__()
        assert cfg.n_heads % cfg.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.attn_dropout = cfg.dropout
        self.rope = rope
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=cfg.bias)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=cfg.bias)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=cfg.bias)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=cfg.bias)
        self.q_norm = nn.RMSNorm(cfg.head_dim, eps=cfg.rms_norm_eps)
        self.k_norm = nn.RMSNorm(cfg.head_dim, eps=cfg.rms_norm_eps)

    def forward(self, x: Float[Tensor, "batch seq_len d_model"]) -> Float[Tensor, "batch seq_len d_model"]:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        cos, sin = self.rope(seq_len)
        q, k = self.rope.apply_rope(q, k, cos, sin)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=self.attn_dropout if self.training else 0.0,
            enable_gqa=True,
        )
        return self.o_proj(out.transpose(1, 2).reshape(batch, seq_len, -1))


class TransformerBlock(nn.Module):
    """
    One pre-norm transformer block: each sublayer (attention, FFN) reads from an RMSNormed
    copy of the residual stream and adds its output back, leaving the residual path itself
    unnormalized.
    """

    def __init__(self, cfg: TransformerConfig, rope: RotaryPositionalEmbedding):
        super().__init__()
        self.attn_norm = nn.RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)
        self.attn = GroupedQueryAttention(cfg, rope)
        self.ffn_norm = nn.RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)
        self.ffn = FeedForwardNetwork(cfg)

    def forward(self, x: Float[Tensor, "batch seq_len d_model"]) -> Float[Tensor, "batch seq_len d_model"]:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Transformer(nn.Module):
    """
    Stack of n_layers pre-norm transformer blocks that operate on already-embedded inputs.
    Owns the shared RoPE cache (one set of cos/sin tables broadcast to every block) and
    applies a final RMSNorm at the top, which the pre-norm formulation needs because the
    residual stream itself is never normalized inside the stack.
    """

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.rope = RotaryPositionalEmbedding(cfg.head_dim, cfg.max_seq_len, cfg.rope_base)
        self.blocks = nn.ModuleList([TransformerBlock(cfg, self.rope) for _ in range(cfg.n_layers)])
        self.final_norm = nn.RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)

    def forward(self, x: Float[Tensor, "batch seq_len d_model"]) -> Float[Tensor, "batch seq_len d_model"]:
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)


@dataclass
class GenerativeRecommendationModelConfig(TransformerConfig):
    n_post_embeddings: int
    n_author_embeddings: int
    n_engagements: int
    n_id_hashes: int
    engagement_loss_weight: float
    engagement_priors: tuple[float, ...]


@dataclass
class GenerativeRecommendationModelLoss:
    total: Tensor
    post_ce: Tensor
    engagement_bce: Tensor
    engagement_rce: Float[Tensor, "n_engagements"]


class GenerativeRecommendationModel(nn.Module):
    """
    Decoder-only recommendation model that autoregressively predicts the next post in a
    user's timeline:

        P_theta(p_t | p_<t, e_<t)

    where theta are the model weights, p_i is the post at position i, and e_i is the vector
    of observed engagements on p_i (e_i is an input feature, not a prediction target).

    The surrogate objective is to recommend the post p* that maximizes weighted engagement:

        p* = argmax_p(Σ_j  w_j * r_j(p))

    where r_j(p) is the engagement score for head j (favorited, replied, ...) and w_j
    is a tuned heuristic weighting.
    """

    engagement_priors: Tensor
    id_hash_multipliers: Tensor

    def __init__(self, cfg: GenerativeRecommendationModelConfig):
        super().__init__()
        assert len(cfg.engagement_priors) == cfg.n_engagements, "engagement_priors length must match n_engagements"
        assert cfg.n_id_hashes >= 1, "n_id_hashes must be >= 1"
        self.n_post_embeddings = cfg.n_post_embeddings
        self.n_author_embeddings = cfg.n_author_embeddings
        self.engagement_loss_weight = cfg.engagement_loss_weight
        self.post_embedding = nn.Embedding(cfg.n_post_embeddings, cfg.d_model)
        self.author_embedding = nn.Embedding(cfg.n_author_embeddings, cfg.d_model)
        self.engagement_proj = nn.Linear(cfg.n_engagements, cfg.d_model, bias=cfg.bias)
        self.transformer = Transformer(cfg)
        self.post_head = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.engagement_head = nn.Linear(cfg.d_model, cfg.n_engagements, bias=cfg.bias)
        self.register_buffer("engagement_priors", torch.tensor(cfg.engagement_priors))
        self.register_buffer("id_hash_multipliers", hash_multipliers(cfg.n_id_hashes), persistent=False)

    def loss(
        self,
        post_logits: Float[Tensor, "batch seq_len n_post_embeddings"],
        eng_logits: Float[Tensor, "batch seq_len n_engagements"],
        post_ids: Int[Tensor, "batch seq_len"],
        engagements: Float[Tensor, "batch seq_len n_engagements"],
    ) -> GenerativeRecommendationModelLoss:
        # Teacher-forced next-step loss: output at position t predicts position t + 1, so
        # we drop the last prediction and align targets one step ahead.
        post_logits, eng_logits = post_logits[:, :-1].float(), eng_logits[:, :-1].float()
        target_buckets = ((post_ids[:, 1:] * self.id_hash_multipliers[0]) % self.n_post_embeddings).reshape(-1)
        target_engagements = engagements[:, 1:]

        post_ce = F.cross_entropy(post_logits.reshape(-1, self.n_post_embeddings), target_buckets)
        eng_bce_per_head = F.binary_cross_entropy_with_logits(eng_logits, target_engagements, reduction="none").mean(
            dim=(0, 1)
        )

        # Closed-form baseline: BCE(l, y) = softplus(l) - y*l, so averaging over (batch, seq)
        # against a constant per-head prior logit reduces to softplus(l) - mean(y)*l.
        prior_logits = torch.logit(self.engagement_priors, eps=1e-6)
        baseline_bce_per_head = F.softplus(prior_logits) - target_engagements.mean(dim=(0, 1)) * prior_logits

        return GenerativeRecommendationModelLoss(
            total=post_ce + self.engagement_loss_weight * eng_bce_per_head.mean(),
            post_ce=post_ce,
            engagement_bce=eng_bce_per_head.mean(),
            engagement_rce=1.0 - eng_bce_per_head / baseline_bce_per_head,
        )

    def forward(
        self,
        post_ids: Int[Tensor, "batch seq_len"],
        author_ids: Int[Tensor, "batch seq_len"],
        engagements: Float[Tensor, "batch seq_len n_engagements"],
        decode_all: bool = False,
    ) -> tuple[
        Float[Tensor, "batch decoded_seq_len n_post_embeddings"],
        Float[Tensor, "batch seq_len n_engagements"],
    ]:
        assert post_ids.shape == engagements.shape[:2], "engagements must align with post_ids on (batch, seq_len)"
        assert author_ids.shape == post_ids.shape, "author_ids must align with post_ids on (batch, seq_len)"

        with torch.autocast(device_type=post_ids.device.type, dtype=torch.bfloat16):
            # Multi-hash ID embeddings: each ID is summed across k independent bucket hashes,
            # so two IDs only collide when all k hashes match - collision damage decays in k.
            post_buckets = (post_ids.unsqueeze(-1) * self.id_hash_multipliers) % self.n_post_embeddings
            author_buckets = (author_ids.unsqueeze(-1) * self.id_hash_multipliers) % self.n_author_embeddings
            post_emb = self.post_embedding(post_buckets).sum(dim=-2)
            author_emb = self.author_embedding(author_buckets).sum(dim=-2)
            eng_emb = self.engagement_proj(engagements)
            h = self.transformer(post_emb + author_emb + eng_emb)

            # Per-engagement scores: project output into unnormalized logit per engagement type.
            eng_scores = self.engagement_head(h)

            # Post-ID retrieval: project h into post-embedding space and use dot-product similarity.
            # Avoid cosine sim: it re-normalizes the full table every step and discards magnitude.
            pred = self.post_head(h if decode_all else h[:, -1:])
            post_logits = pred @ self.post_embedding.weight.t()

        return post_logits, eng_scores
