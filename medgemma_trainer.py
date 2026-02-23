"""
MedGemma-4B Clinical Fine-Tuning Pipeline (CUDA / GPU).

Fine-tunes Google's medically pre-trained MedGemma-4B-IT using the same
ODE-RNN + soft-prompt architecture from lora_trainer.py.

Key differences vs lora_trainer.py (DeepSeek-14B):
  - google/medgemma-4b-it  (4B, already medical-domain pretrained)
  - 4-bit bitsandbytes quantization on CUDA (~2-3 GB vs ~7 GB for 14B)
  - Larger effective batch sizes possible (batch_size=16 default)
  - Two fine-tuning modes: classification and generative

Modes (--mode):
  classification  ODE-RNN → soft prompts → MedGemma → binary head (BCELoss)
  generative      ODE-RNN → soft prompts → MedGemma → clinical text (causal LM loss)

Trajectory-only:
  --trajectory_only  Train only ODE-RNN + AttentiveClassifier; skip LLM entirely.
                     Use this first to validate embeddings reach >70% AUC.

Prerequisites:
    pip install transformers>=4.45 peft>=0.13 accelerate bitsandbytes torchdiffeq sklearn
    huggingface-cli login          # MedGemma is gated — accept license first

Usage:
    # Validate ODE-RNN embeddings (no LLM cost)
    python medgemma_trainer.py --trajectory_only --dataset srtr --organ kidney

    # Fine-tune for binary outcome classification
    python medgemma_trainer.py --mode classification --dataset srtr --organ kidney

    # Fine-tune for clinical text generation
    python medgemma_trainer.py --mode generative --dataset dka

    # Load pretrained ODE-RNN, freeze it, train only projection + LoRA
    python medgemma_trainer.py --mode generative \\
        --ode_checkpoint ./checkpoints_medgemma_dka/medgemma_trajectory_best.pt
"""

import argparse
import importlib.util
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def setup_file_logging(log_file: str, model_short: str, dataset: str) -> None:
    if log_file == "none":
        return
    if log_file == "auto":
        Path("logs").mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/medgemma_{dataset}_{model_short}_{ts}.log"
    fh = logging.FileHandler(log_file)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(fh)
    logger.info(f"Logging to: {log_file}")


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ODERNNConfig:
    input_dim: int = 1            # 1 for DKA (anion gap), 4 for SRTR
    hidden_dim: int = 64
    time_scale: float = 72.0     # hours; 72 for DKA, 20000 for SRTR
    solver_method: str = "dopri5"
    rtol: float = 1e-3
    atol: float = 1e-4


@dataclass
class LoRAConfig:
    r: int = 8                   # Smaller rank than 14B (4B model, less overfit risk)
    lora_alpha: int = 16
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    lora_dropout: float = 0.05
    bias: Literal["none", "all", "lora_only"] = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class ModelConfig:
    model_name: str = "google/medgemma-4b-it"
    load_in_4bit: bool = True    # bitsandbytes 4-bit on CUDA
    trust_remote_code: bool = False


@dataclass
class ProjectionConfig:
    num_prefix_tokens: int = 8
    llm_hidden_dim: Optional[int] = None   # set dynamically from model.config.hidden_size
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    # 4B model with 4-bit quantization is ~2-3 GB; batch 16 is comfortable on 40 GB
    batch_size: int = 16
    gradient_accumulation_steps: int = 2   # effective batch = 32
    num_epochs: int = 10
    lr: float = 5e-5
    ode_rnn_lr: float = 1e-5
    lora_lr: float = 2e-4
    weight_decay: float = 0.01
    early_stopping_patience: int = 5
    warmup_ratio: float = 0.03
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0
    max_seq_len: int = 50
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    save_steps: int = 100
    eval_steps: int = 50
    logging_steps: int = 10
    output_dir: str = "./checkpoints_medgemma"
    seed: int = 42
    num_workers: int = 4


@dataclass
class MedGemmaConfig:
    ode_rnn: ODERNNConfig = field(default_factory=ODERNNConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def get_dka_config() -> MedGemmaConfig:
    cfg = MedGemmaConfig()
    cfg.ode_rnn.input_dim = 1
    cfg.ode_rnn.time_scale = 72.0   # DKA resolves in 6-72h; normalize to this range
    cfg.training.max_seq_len = 20
    cfg.training.output_dir = "./checkpoints_medgemma_dka"
    return cfg


def get_srtr_config() -> MedGemmaConfig:
    cfg = MedGemmaConfig()
    cfg.ode_rnn.input_dim = 4
    cfg.ode_rnn.time_scale = 20000.0
    cfg.training.max_seq_len = 50
    cfg.training.output_dir = "./checkpoints_medgemma_srtr"
    return cfg


# ──────────────────────────────────────────────────────────────────────────────
# Local module helpers
# ──────────────────────────────────────────────────────────────────────────────

_current_dir = Path(__file__).parent


def _import_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_dataset_mod = _import_module(str(_current_dir / "dataset-format.py"), "dataset_format")
MIMICtrajectoryDataset = _dataset_mod.MIMICtrajectoryDataset

from dte_encoder import ClinicalODERNN  # noqa: E402

# ──────────────────────────────────────────────────────────────────────────────
# Architecture
# ──────────────────────────────────────────────────────────────────────────────

class TrajectoryProjection(nn.Module):
    """Projects ODE-RNN embedding → LLM embedding space as soft-prompt tokens."""

    def __init__(
        self,
        trajectory_dim: int,
        llm_hidden_dim: int,
        num_prefix_tokens: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_prefix_tokens = num_prefix_tokens
        self.llm_hidden_dim = llm_hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(trajectory_dim, trajectory_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(trajectory_dim * 4, num_prefix_tokens * llm_hidden_dim),
            nn.Dropout(dropout),
        )
        self.layer_norm = nn.LayerNorm(llm_hidden_dim)
        self.embed_scale = nn.Parameter(torch.ones(llm_hidden_dim))
        self.embed_bias = nn.Parameter(torch.zeros(llm_hidden_dim))

    def init_from_llm(self, llm_model: nn.Module) -> None:
        with torch.no_grad():
            embed: nn.Embedding = llm_model.get_input_embeddings()  # type: ignore[assignment]
            w = embed.weight.data
            self.embed_scale.data = w.std(dim=0).to(self.embed_scale.device)
            self.embed_bias.data = w.mean(dim=0).to(self.embed_bias.device)

    def forward(self, traj: torch.Tensor) -> torch.Tensor:
        batch = traj.shape[0]
        out = self.projection(traj).view(batch, self.num_prefix_tokens, self.llm_hidden_dim)
        out = self.layer_norm(out)
        return out * self.embed_scale + self.embed_bias


class ClassificationHead(nn.Module):
    """Binary head: mean-pools over trajectory prefix tokens → logit."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1, num_prefix_tokens: int = 8):
        super().__init__()
        self.num_prefix_tokens = num_prefix_tokens
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pooled = hidden_states[:, : self.num_prefix_tokens, :].mean(dim=1)
        return self.classifier(pooled)


# ──────────────────────────────────────────────────────────────────────────────
# Generative dataset: attaches synthetic clinical text targets to trajectories
# ──────────────────────────────────────────────────────────────────────────────

DKA_FAST = (
    "The patient demonstrates rapid DKA resolution. The anion gap normalized "
    "within 6 hours, indicating excellent treatment response and appropriate "
    "insulin sensitivity. Continue current insulin infusion and IV fluid protocol. "
    "Repeat metabolic panel in 4 hours to confirm sustained normalization."
)
DKA_SLOW = (
    "The patient shows inadequate DKA treatment response. The anion gap has not "
    "normalized after 24 hours, suggesting concurrent infection, insulin resistance, "
    "or impaired renal clearance. Recommend: blood cultures, urine culture, "
    "consider increasing insulin infusion, reassess fluid balance, and consult "
    "endocrinology if no improvement in 6 hours."
)
SRTR_STABLE = (
    "The transplant candidate demonstrates stable disease progression. Current "
    "clinical trajectory supports continued waitlist management. Maintain current "
    "UNOS status and monitoring protocol with scheduled follow-up."
)
SRTR_HIGH_RISK = (
    "The transplant candidate's trajectory indicates rapid deterioration with high "
    "waitlist mortality risk. Recommend expedited transplant evaluation, consideration "
    "of status upgrade, intensification of bridging therapy, and palliative care "
    "consultation."
)
USER_PROMPT = (
    "Patient clinical trajectory has been encoded. Provide a concise ICU assessment "
    "with recommendations:"
)


def _tokenize_text(tokenizer, text: str, add_eos: bool = False) -> torch.Tensor:
    """Tokenize text → 1-D LongTensor of token ids (no padding)."""
    suffix = tokenizer.eos_token if add_eos else ""
    ids = tokenizer(
        text + suffix,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"].squeeze(0)
    return ids


class _GenerativeDataset(Dataset):
    """Base wrapper: adds prompt_ids / response_ids to any trajectory dataset."""

    _LABEL_MAP: Dict[int, str] = {}   # override in subclass

    def __init__(self, base_dataset: Dataset, tokenizer, max_response_len: int = 256):
        self.base = base_dataset
        self.prompt_ids = _tokenize_text(tokenizer, USER_PROMPT)
        self.tokenizer = tokenizer
        self.max_response_len = max_response_len
        # Pre-tokenize both response templates
        self._response_tokens: Dict[int, torch.Tensor] = {
            k: _tokenize_text(tokenizer, v, add_eos=True)[: max_response_len]
            for k, v in self._LABEL_MAP.items()
        }

    def __len__(self) -> int:
        return len(self.base)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> Dict:
        sample = self.base[idx]
        label = int(sample["label"].item())
        response_ids = self._response_tokens.get(label, self._response_tokens[0])
        return {
            **sample,
            "prompt_ids": self.prompt_ids,
            "response_ids": response_ids,
        }


class GenerativeDKADataset(_GenerativeDataset):
    _LABEL_MAP = {0: DKA_FAST, 1: DKA_SLOW}


class GenerativeSRTRDataset(_GenerativeDataset):
    _LABEL_MAP = {0: SRTR_STABLE, 1: SRTR_HIGH_RISK}


def generative_collate(batch: List[Dict]) -> Dict:
    """Pad prompt_ids and response_ids to per-batch max lengths."""
    keys = ["values", "times", "mask", "label"]
    out = {k: torch.stack([b[k] for b in batch]) for k in keys}

    max_p = max(b["prompt_ids"].shape[0] for b in batch)
    max_r = max(b["response_ids"].shape[0] for b in batch)

    prompt_pad = torch.zeros(len(batch), max_p, dtype=torch.long)
    resp_pad = torch.full((len(batch), max_r), -100, dtype=torch.long)

    for i, b in enumerate(batch):
        p, r = b["prompt_ids"], b["response_ids"]
        prompt_pad[i, : p.shape[0]] = p
        resp_pad[i, : r.shape[0]] = r

    out["prompt_ids"] = prompt_pad
    out["response_ids"] = resp_pad
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Combined model
# ──────────────────────────────────────────────────────────────────────────────

class MedGemmaTrajectoryModel(nn.Module):
    """
    ODE-RNN (CPU) + TrajectoryProjection + MedGemma-4B (LoRA, CUDA).

    Two forward modes:
      forward_classification → (batch, 1) logits
      forward_generative     → scalar causal-LM loss on response tokens only
    """

    def __init__(self, config: MedGemmaConfig, llm=None, tokenizer=None):
        super().__init__()
        self.config = config
        self.llm = llm
        self.tokenizer = tokenizer
        self._main_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ODE-RNN always on CPU (torchdiffeq float64 not reliable on CUDA)
        self.ode_rnn = ClinicalODERNN(
            input_dim=config.ode_rnn.input_dim,
            hidden_dim=config.ode_rnn.hidden_dim,
            time_scale=config.ode_rnn.time_scale,
            solver_method=config.ode_rnn.solver_method,
            rtol=config.ode_rnn.rtol,
            atol=config.ode_rnn.atol,
        ).to("cpu")

        _lcfg = getattr(llm.config, "text_config", llm.config) if llm is not None else None
        llm_hidden = _lcfg.hidden_size if _lcfg is not None else 2560
        self.projection = TrajectoryProjection(
            trajectory_dim=config.ode_rnn.hidden_dim,
            llm_hidden_dim=llm_hidden,
            num_prefix_tokens=config.projection.num_prefix_tokens,
            dropout=config.projection.dropout,
        ).to(self._main_device)

        self.classifier = ClassificationHead(
            hidden_dim=llm_hidden,
            dropout=config.projection.dropout,
            num_prefix_tokens=config.projection.num_prefix_tokens,
        ).to(self._main_device)

        if llm is not None:
            self.projection.init_from_llm(llm)

    # ── ODE-RNN encoding ──────────────────────────────────────────────────────

    def _encode(
        self,
        values: torch.Tensor,
        times: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Returns (batch, hidden_dim) trajectory embedding."""
        if values.dim() == 2:
            values = values.unsqueeze(-1)
        batch, seq_len, _ = values.shape
        embeds = []

        for i in range(batch):
            valid = max(int(mask[i].sum().item()) if mask is not None else seq_len, 1)
            v = values[i, :valid, :].unsqueeze(0).float().cpu()
            t = times[i, :valid].unsqueeze(0).float().cpu()
            for j in range(1, t.shape[1]):
                if t[0, j] <= t[0, j - 1]:
                    t[0, j] = t[0, j - 1] + 0.01

            if not any(p.requires_grad for p in self.ode_rnn.parameters()):
                with torch.no_grad():
                    out = self.ode_rnn(t, v)
            else:
                out = self.ode_rnn(t, v)

            embeds.append(out[:, -1, :].to(self._main_device))

        return torch.cat(embeds, dim=0)

    # ── Classification forward ────────────────────────────────────────────────

    def forward_classification(
        self,
        values: torch.Tensor,
        times: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        traj = self._encode(values, times, mask)
        prefix = self.projection(traj)   # (B, P, hidden)

        if self.llm is not None and self.tokenizer is not None:
            B = values.shape[0]
            prompt = "Clinical trajectory analysis for outcome prediction:"
            tokens = self.tokenizer(
                [prompt] * B, return_tensors="pt", padding=True, truncation=True
            ).to(self._main_device)
            tok_embeds = self.llm.get_input_embeddings()(tokens["input_ids"])
            combined = torch.cat([prefix, tok_embeds], dim=1)
            P = self.config.projection.num_prefix_tokens
            attn = torch.cat(
                [torch.ones(B, P, device=self._main_device), tokens["attention_mask"]], dim=1
            )
            # Gemma3 requires token_type_ids when training (0 = text, 1 = image)
            seq_len = combined.shape[1]
            token_type_ids = torch.zeros(B, seq_len, dtype=torch.long, device=self._main_device)
            out = self.llm(
                inputs_embeds=combined,
                attention_mask=attn,
                token_type_ids=token_type_ids,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden = out.hidden_states[-1]
        else:
            hidden = prefix

        return {"logits": self.classifier(hidden)}

    # ── Generative forward ────────────────────────────────────────────────────

    def forward_generative(
        self,
        values: torch.Tensor,
        times: torch.Tensor,
        mask: Optional[torch.Tensor],
        prompt_ids: torch.Tensor,
        response_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Causal LM loss on response tokens only.

        Sequence:  [trajectory_prefix (P) | prompt_tokens (Q) | response_tokens (R)]
        Labels:    [     -100 (P)         |     -100 (Q)       | response_token_ids ]
        """
        assert self.llm is not None, "LLM required for generative mode"
        B = values.shape[0]
        P = self.config.projection.num_prefix_tokens

        traj = self._encode(values, times, mask)
        prefix = self.projection(traj)   # (B, P, hidden)

        embed_fn = self.llm.get_input_embeddings()
        prompt_embeds = embed_fn(prompt_ids.to(self._main_device))          # (B, Q, hidden)
        # clamp -100 padding before embedding; it is masked by labels anyway
        resp_safe = response_ids.to(self._main_device).clamp(min=0)
        resp_embeds = embed_fn(resp_safe)                                    # (B, R, hidden)

        full_embeds = torch.cat([prefix, prompt_embeds, resp_embeds], dim=1)

        Q, R = prompt_ids.shape[1], response_ids.shape[1]
        attn = torch.ones(B, P + Q + R, device=self._main_device)

        # Supervise only on response tokens
        ignore = torch.full((B, P + Q), -100, dtype=torch.long, device=self._main_device)
        resp_labels = response_ids.to(self._main_device)   # already -100 where padded
        labels = torch.cat([ignore, resp_labels], dim=1)

        # Gemma3 requires token_type_ids when training (0 = text, 1 = image)
        token_type_ids = torch.zeros(B, P + Q + R, dtype=torch.long, device=self._main_device)
        out = self.llm(
            inputs_embeds=full_embeds,
            attention_mask=attn,
            token_type_ids=token_type_ids,
            labels=labels,
            return_dict=True,
        )
        return out.loss

    # ── Utilities ─────────────────────────────────────────────────────────────

    def freeze_ode_rnn(self) -> None:
        for p in self.ode_rnn.parameters():
            p.requires_grad = False
        n = sum(p.numel() for p in self.ode_rnn.parameters())
        logger.info(f"ODE-RNN frozen ({n:,} params)")

    def get_param_groups(self, include_ode: bool = True) -> List[Dict]:
        groups: List[Dict] = []
        if include_ode:
            ode_p = [p for p in self.ode_rnn.parameters() if p.requires_grad]
            if ode_p:
                groups.append({"params": ode_p, "lr": self.config.training.ode_rnn_lr})

        groups.append({
            "params": list(self.projection.parameters()) + list(self.classifier.parameters()),
            "lr": self.config.training.lr,
        })
        if self.llm is not None:
            lora_p = [p for p in self.llm.parameters() if p.requires_grad]
            if lora_p:
                groups.append({"params": lora_p, "lr": self.config.training.lora_lr})
        return groups


# ──────────────────────────────────────────────────────────────────────────────
# Model loading (CUDA + bitsandbytes 4-bit)
# ──────────────────────────────────────────────────────────────────────────────

def load_medgemma_with_lora(config: MedGemmaConfig):
    """Load MedGemma-4B-IT with 4-bit bitsandbytes quantization and LoRA."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # type: ignore[import]
    except ImportError:
        raise ImportError(
            "Install: pip install transformers>=4.45 peft>=0.13 accelerate bitsandbytes"
        )

    name = config.model.model_name
    logger.info(f"Loading {name} with 4-bit quantization ...")

    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    from transformers import PreTrainedModel

    base: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        name,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=False,
    )
    base = prepare_model_for_kbit_training(base)  # type: ignore[assignment]

    lora_cfg = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        target_modules=config.lora.target_modules,
        lora_dropout=config.lora.lora_dropout,
        bias=config.lora.bias,
        task_type=config.lora.task_type,
    )
    model: PreTrainedModel = get_peft_model(base, lora_cfg)  # type: ignore[assignment]
    model.print_trainable_parameters()  # type: ignore[operator]

    if config.training.gradient_checkpointing:
        model.gradient_checkpointing_enable()  # type: ignore[operator]

    # Gemma3Config nests text config under text_config; fall back to top-level
    _cfg = getattr(model.config, "text_config", model.config)  # type: ignore[union-attr]
    hidden_size: int = _cfg.hidden_size
    logger.info(f"MedGemma loaded — hidden_size: {hidden_size}")
    return model, tokenizer


# ──────────────────────────────────────────────────────────────────────────────
# Trainers
# ──────────────────────────────────────────────────────────────────────────────

class _BaseTrainer:
    def __init__(self, model: MedGemmaTrajectoryModel, config: MedGemmaConfig):
        self.model = model
        self.config = config
        self.device = model._main_device
        self.global_step = 0
        self.patience_counter = 0
        self.optimizer = torch.optim.AdamW(
            model.get_param_groups(include_ode=True),
            weight_decay=config.training.weight_decay,
        )

    def _make_scheduler(self, total_steps: int):
        warmup = int(total_steps * self.config.training.warmup_ratio)
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(total_steps - warmup, 1)
        )

    def _save(self, name: str, extra: Optional[Dict] = None) -> None:
        ckpt_dir = Path(self.config.training.output_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        short = self.config.model.model_name.split("/")[-1]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = ckpt_dir / f"{name}_{short}_{ts}.pt"
        payload = {
            "global_step": self.global_step,
            "ode_rnn": self.model.ode_rnn.state_dict(),
            "projection": self.model.projection.state_dict(),
            "classifier": self.model.classifier.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)
        if self.model.llm is not None:
            self.model.llm.save_pretrained(str(ckpt_dir / f"{name}_{short}_{ts}_lora"))
        logger.info(f"Checkpoint saved: {path}")


class ClassificationTrainer(_BaseTrainer):
    def __init__(
        self,
        model: MedGemmaTrajectoryModel,
        config: MedGemmaConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
    ):
        super().__init__(model, config)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = nn.BCEWithLogitsLoss()
        total = len(train_loader) * config.training.num_epochs
        self.scheduler = self._make_scheduler(total)
        self.best_val_auc = 0.0

    def train(self) -> Dict:
        history: Dict = {"train_loss": [], "val_loss": [], "val_acc": [], "val_auc": []}
        cfg = self.config.training

        for epoch in range(cfg.num_epochs):
            tr_loss = self._train_epoch(epoch)
            val_loss, val_acc = self._eval(self.val_loader)
            val_auc = _auc(self.model, self.val_loader)

            history["train_loss"].append(tr_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_auc"].append(val_auc)
            logger.info(
                f"Epoch {epoch+1}/{cfg.num_epochs} | "
                f"train {tr_loss:.4f} | val loss {val_loss:.4f} acc {val_acc:.4f} auc {val_auc:.4f}"
            )

            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.patience_counter = 0
                self._save("best", {"best_val_auc": val_auc})
            else:
                self.patience_counter += 1
                if self.patience_counter >= cfg.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        if self.test_loader:
            tl, ta = self._eval(self.test_loader)
            ta_auc = _auc(self.model, self.test_loader)
            logger.info(f"Test | loss {tl:.4f} acc {ta:.4f} auc {ta_auc:.4f}")

        return history

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        cfg = self.config.training
        self.optimizer.zero_grad()

        for step, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch+1}")):
            out = self.model.forward_classification(
                batch["values"].to(self.device),
                batch["times"].to(self.device),
                batch["mask"].to(self.device),
            )
            loss = self.criterion(out["logits"].squeeze(-1), batch["label"].to(self.device))
            (loss / cfg.gradient_accumulation_steps).backward()
            total_loss += loss.item()

            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def _eval(self, loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        for batch in loader:
            out = self.model.forward_classification(
                batch["values"].to(self.device),
                batch["times"].to(self.device),
                batch["mask"].to(self.device),
            )
            loss = self.criterion(out["logits"].squeeze(-1), batch["label"].to(self.device))
            total_loss += loss.item()
            preds = (torch.sigmoid(out["logits"].squeeze(-1)) > 0.5).float()
            correct += (preds == batch["label"].to(self.device)).sum().item()
            total += batch["label"].size(0)
        return total_loss / len(loader), correct / max(total, 1)


class GenerativeTrainer(_BaseTrainer):
    """Causal LM loss on clinical text targets; no AUC metric."""

    def __init__(
        self,
        model: MedGemmaTrajectoryModel,
        config: MedGemmaConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
    ):
        super().__init__(model, config)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        total = len(train_loader) * config.training.num_epochs
        self.scheduler = self._make_scheduler(total)
        self.best_val_loss = float("inf")

    def train(self) -> Dict:
        history: Dict = {"train_loss": [], "val_loss": []}
        cfg = self.config.training

        for epoch in range(cfg.num_epochs):
            tr_loss = self._train_epoch(epoch)
            val_loss = self._eval(self.val_loader)
            history["train_loss"].append(tr_loss)
            history["val_loss"].append(val_loss)
            logger.info(
                f"Epoch {epoch+1}/{cfg.num_epochs} | train {tr_loss:.4f} | val {val_loss:.4f}"
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save("best", {"best_val_loss": val_loss})
            else:
                self.patience_counter += 1
                if self.patience_counter >= cfg.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        return history

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        cfg = self.config.training
        self.optimizer.zero_grad()

        for step, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch+1}")):
            loss = self.model.forward_generative(
                batch["values"].to(self.device),
                batch["times"].to(self.device),
                batch["mask"].to(self.device),
                batch["prompt_ids"],
                batch["response_ids"],
            )
            (loss / cfg.gradient_accumulation_steps).backward()
            total_loss += loss.item()

            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def _eval(self, loader: DataLoader) -> float:
        self.model.eval()
        total = 0.0
        for batch in loader:
            total += self.model.forward_generative(
                batch["values"].to(self.device),
                batch["times"].to(self.device),
                batch["mask"].to(self.device),
                batch["prompt_ids"],
                batch["response_ids"],
            ).item()
        return total / max(len(loader), 1)


# ──────────────────────────────────────────────────────────────────────────────
# Trajectory-only (no LLM)
# ──────────────────────────────────────────────────────────────────────────────

class AttentiveTrajectoryClassifier(nn.Module):
    """
    Classifier with attention pooling over all ODE-RNN hidden states.
    Learns which time points are most predictive rather than using only the
    last hidden state. Heavy regularization to prevent overfitting.
    """

    def __init__(self, trajectory_dim: int, hidden_dim: int = 32, dropout: float = 0.5):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.attention = nn.Linear(trajectory_dim, 1, bias=False)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(trajectory_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, hidden_states: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden_states = self.dropout(hidden_states)
        attn_scores = self.attention(hidden_states)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(-1) == 0, float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=1)
        pooled = (hidden_states * attn_weights).sum(dim=1)
        return self.classifier(pooled)


def run_trajectory_only(
    config: MedGemmaConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 100,
    patience: int = 10,
    lr: float = 1e-4,
) -> Tuple[nn.Module, nn.Module, Dict]:
    from sklearn.metrics import roc_auc_score

    main_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_len = config.training.max_seq_len

    ode_rnn = ClinicalODERNN(
        input_dim=config.ode_rnn.input_dim,
        hidden_dim=config.ode_rnn.hidden_dim,
        time_scale=config.ode_rnn.time_scale,
        solver_method=config.ode_rnn.solver_method,
        rtol=config.ode_rnn.rtol,
        atol=config.ode_rnn.atol,
    ).to("cpu")

    classifier = AttentiveTrajectoryClassifier(
        trajectory_dim=config.ode_rnn.hidden_dim, hidden_dim=32, dropout=0.5
    ).to(main_device)

    optimizer = torch.optim.AdamW(
        [{"params": ode_rnn.parameters(), "lr": lr},
         {"params": classifier.parameters(), "lr": lr}],
        weight_decay=0.1,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.BCEWithLogitsLoss()
    best_auc, patience_counter = 0.0, 0
    history: Dict = {"train_loss": [], "val_acc": [], "val_auc": []}

    def _encode_batch(batch):
        values, times, mask = batch["values"], batch["times"], batch["mask"]
        if values.dim() == 2:
            values = values.unsqueeze(-1)
        bs = values.shape[0]
        all_hs, all_ms = [], []
        for i in range(bs):
            valid = max(int(mask[i].sum().item()), 1)
            v = values[i, :valid, :].unsqueeze(0).float()
            t = times[i, :valid].unsqueeze(0).float()
            for j in range(1, t.shape[1]):
                if t[0, j] <= t[0, j - 1]:
                    t[0, j] = t[0, j - 1] + 0.01
            out = ode_rnn(t, v)
            hdim = out.shape[-1]
            padded = torch.zeros(1, max_len, hdim)
            padded[0, :valid] = out[0]
            all_hs.append(padded)
            smask = torch.zeros(1, max_len)
            smask[0, :valid] = 1.0
            all_ms.append(smask)
        return (
            torch.cat(all_hs, dim=0).to(main_device),
            torch.cat(all_ms, dim=0).to(main_device),
        )

    for epoch in range(num_epochs):
        ode_rnn.train()
        classifier.train()
        epoch_loss = 0.0

        for batch in tqdm(train_loader, desc=f"TrajOnly {epoch+1}", leave=False):
            labels = batch["label"].to(main_device)
            hs, am = _encode_batch(batch)
            optimizer.zero_grad()
            loss = criterion(classifier(hs, am).squeeze(-1), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ode_rnn.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        ode_rnn.eval()
        classifier.eval()
        all_labels, all_probs, correct, total = [], [], 0, 0

        with torch.no_grad():
            for batch in val_loader:
                labels = batch["label"].to(main_device)
                hs, am = _encode_batch(batch)
                probs = torch.sigmoid(classifier(hs, am).squeeze(-1)).cpu()
                preds = (probs > 0.5).float()
                correct += (preds == batch["label"]).sum().item()
                total += labels.size(0)
                all_labels.extend(batch["label"].numpy())
                all_probs.extend(probs.numpy())

        val_acc = correct / max(total, 1)
        val_auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5
        avg_loss = epoch_loss / len(train_loader)
        history["train_loss"].append(avg_loss)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)

        if (epoch + 1) % 5 == 0:
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} | loss {avg_loss:.4f} | "
                f"val acc {val_acc:.4f} auc {val_auc:.4f}"
            )

        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            ckpt_dir = Path(config.training.output_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"ode_rnn": ode_rnn.state_dict(), "classifier": classifier.state_dict(),
                 "epoch": epoch, "val_acc": val_acc, "val_auc": val_auc},
                ckpt_dir / "medgemma_trajectory_best.pt",
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}. Best AUC: {best_auc:.4f}")
                break

    logger.info(f"Trajectory-only complete. Best AUC: {best_auc:.4f}")
    return ode_rnn, classifier, history


# ──────────────────────────────────────────────────────────────────────────────
# AUC helper
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _auc(model: MedGemmaTrajectoryModel, loader: DataLoader) -> float:
    from sklearn.metrics import roc_auc_score
    model.eval()
    device = model._main_device
    all_labels, all_probs = [], []
    for batch in loader:
        out = model.forward_classification(
            batch["values"].to(device),
            batch["times"].to(device),
            batch["mask"].to(device),
        )
        all_probs.extend(torch.sigmoid(out["logits"].squeeze(-1)).cpu().numpy())
        all_labels.extend(batch["label"].numpy())
    return float(roc_auc_score(all_labels, all_probs)) if len(set(all_labels)) > 1 else 0.5


# ──────────────────────────────────────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────────────────────────────────────

def _split(dataset, config: MedGemmaConfig, collate_fn=None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    cfg = config.training
    n = len(dataset)  # type: ignore[arg-type]
    train_n = int(n * cfg.train_split)
    val_n = int(n * cfg.val_split)
    test_n = n - train_n - val_n
    gen = torch.Generator().manual_seed(cfg.seed)
    tr, va, te = random_split(dataset, [train_n, val_n, test_n], generator=gen)
    kw: Dict = {"num_workers": cfg.num_workers, "pin_memory": cfg.num_workers > 0}
    if collate_fn:
        kw["collate_fn"] = collate_fn
    return (
        DataLoader(tr, batch_size=cfg.batch_size, shuffle=True, **kw),
        DataLoader(va, batch_size=cfg.batch_size, shuffle=False, **kw),
        DataLoader(te, batch_size=cfg.batch_size, shuffle=False, **kw),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MedGemma-4B Clinical Fine-Tuning (CUDA)")
    parser.add_argument("--mode", choices=["classification", "generative"],
                        default="classification")
    parser.add_argument("--dataset", choices=["dka", "srtr"], default="dka")
    parser.add_argument("--organ", choices=["kidney", "liver"], default="kidney")
    parser.add_argument("--data_path", default="dka_training_data.csv")
    parser.add_argument("--data_dir", default=None, help="SRTR data directory")
    parser.add_argument("--max_samples", type=int, default=10000,
                        help="0 = use all data")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--trajectory_only", action="store_true",
                        help="Train only ODE-RNN (no LLM). Validate embeddings first.")
    parser.add_argument("--trajectory_epochs", type=int, default=100)
    parser.add_argument("--ode_checkpoint", type=str, default=None,
                        help="Pretrained ODE-RNN .pt to load (implies --freeze_ode)")
    parser.add_argument("--freeze_ode", action="store_true")
    parser.add_argument("--time_scale", type=float, default=None,
                        help="Override ODE-RNN time normalization (dka default: 72.0)")
    parser.add_argument("--patience", type=int, default=None,
                        help="Early stopping patience for trajectory-only training (default: 10)")
    parser.add_argument("--traj_lr", type=float, default=None,
                        help="Learning rate for trajectory-only ODE-RNN + classifier (default: 1e-4)")
    parser.add_argument("--log_file", default="auto")
    args = parser.parse_args()

    config = get_dka_config() if args.dataset == "dka" else get_srtr_config()
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.time_scale:
        config.ode_rnn.time_scale = args.time_scale

    short = config.model.model_name.split("/")[-1]
    setup_file_logging(args.log_file, short, args.dataset)

    logger.info("=" * 60)
    logger.info("MedGemma-4B Clinical Fine-Tuning")
    logger.info(f"  Model:   {config.model.model_name}")
    logger.info(f"  Mode:    {'trajectory_only' if args.trajectory_only else args.mode}")
    logger.info(f"  Dataset: {args.dataset}" + (f"/{args.organ}" if args.dataset == "srtr" else ""))
    logger.info(f"  Device:  {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    logger.info("=" * 60)

    # Load base dataset
    if args.dataset == "dka":
        df = pd.read_csv(args.data_path)
        if args.max_samples > 0:
            df = df.head(args.max_samples)
        base_ds = MIMICtrajectoryDataset(df, max_len=config.training.max_seq_len)
    else:
        if args.data_dir is None:
            parser.error("--data_dir required for srtr dataset")
        from srtr_dataset import SRTRTrajectoryDataset
        data_dir = Path(args.data_dir)
        organ = args.organ
        cp = "CAND_KIPA" if organ == "kidney" else "CAND_LIIN"
        hp = "STATHIST_KIPA" if organ == "kidney" else "STATHIST_LIIN"
        cand_df = pd.read_csv(next(data_dir.glob(f"{cp}*.csv")))
        hist_df = pd.read_csv(next(data_dir.glob(f"{hp}*.csv")))
        if args.max_samples > 0:
            cand_df = cand_df.head(args.max_samples)
        base_ds = SRTRTrajectoryDataset(
            cand_df, hist_df, organ=organ, max_seq_len=config.training.max_seq_len
        )

    # Trajectory-only path (no LLM)
    if args.trajectory_only:
        tr, va, _ = _split(base_ds, config)
        run_trajectory_only(
            config, tr, va,
            num_epochs=args.trajectory_epochs,
            patience=args.patience if args.patience else 10,
            lr=args.traj_lr if args.traj_lr else 1e-4,
        )
        return

    # Load MedGemma
    llm, tokenizer = load_medgemma_with_lora(config)

    # Wrap dataset for generative mode
    if args.mode == "generative":
        Wrapper = GenerativeDKADataset if args.dataset == "dka" else GenerativeSRTRDataset
        ds = Wrapper(base_ds, tokenizer)
        tr, va, te = _split(ds, config, collate_fn=generative_collate)
    else:
        tr, va, te = _split(base_ds, config)

    # Build combined model
    model = MedGemmaTrajectoryModel(config, llm=llm, tokenizer=tokenizer)

    if args.ode_checkpoint:
        logger.info(f"Loading ODE-RNN from {args.ode_checkpoint}")
        ckpt = torch.load(args.ode_checkpoint, map_location="cpu", weights_only=False)
        sd = ckpt.get("ode_rnn", ckpt)
        if isinstance(sd, dict) and "ode_rnn" in sd:
            sd = sd["ode_rnn"]
        model.ode_rnn.load_state_dict(sd)

    if args.freeze_ode or args.ode_checkpoint:
        model.freeze_ode_rnn()

    # Train
    if args.mode == "classification":
        ClassificationTrainer(model, config, tr, va, te).train()
    else:
        GenerativeTrainer(model, config, tr, va, te).train()

    logger.info("Done.")


if __name__ == "__main__":
    main()
