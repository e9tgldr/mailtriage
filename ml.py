"""Tier 2: the local Laya-MN checkpoint as a tie-breaker. Never as a decider.

Hard-won facts this adapter encodes (measured 2026-09-22, see docs/CALIBRATION.md):
  * `laya.common.QTYPES` has NO 'scenario' key -- choice questions use QTYPES['choice'].
  * The fitted temperature (3.4326) was calibrated on MASSIVE mn-MN, not on email. It is
    applied because mn_eval.py applies it, so numbers stay comparable -- but it does NOT
    make email probabilities trustworthy. On the probe the model was wrong at p=0.98.
  * Loading costs ~1.3 GB of VRAM and a few seconds. It is lazy: importing this module
    does not touch torch, so the rules tier always works on a machine with no GPU.

To avoid copying the 644 MB checkpoint, point LAYA_MODEL_DIR at the existing export:
    C:\\Users\\hi\\Documents\\laya-mn\\out_mn
That directory also supplies the tokenizer and encoder subfolders, which is why they must
travel together.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass

DEFAULT_LAYA_DIR = os.environ.get(
    "LAYA_MODEL_DIR", r"C:\Users\hi\Documents\laya-mn\out_mn"
)
DEFAULT_LAYA_SRC = os.environ.get("LAYA_SRC_DIR", r"C:\Users\hi\Documents\laya-mn")
DEFAULT_PY = os.environ.get(
    "LAYA_PYTHON", r"C:\Users\hi\Documents\laya-mn\venv\Scripts\python.exe"
)


@dataclass
class MLResult:
    available: bool
    probs: dict[str, float] | None = None
    note: str = ""

    def top(self) -> tuple[str, float]:
        if not self.probs:
            return "", 0.0
        folder = max(self.probs, key=lambda k: self.probs[k])
        return folder, self.probs[folder]

    def runner_up(self) -> float:
        if not self.probs or len(self.probs) < 2:
            return 0.0
        ordered = sorted(self.probs.values(), reverse=True)
        return ordered[1]


class LayaScorer:
    """Lazy wrapper. `available` is False (with a reason) whenever the stack is missing."""

    def __init__(self, model_dir: str = DEFAULT_LAYA_DIR, src_dir: str = DEFAULT_LAYA_SRC):
        self.model_dir = model_dir
        self.src_dir = src_dir
        self._model = None
        self._tok = None
        self._cfg = None
        self._torch = None
        self._helpers = None
        self.available = False
        self.note = ""

    def load(self) -> bool:
        if self._model is not None:
            return True
        weights = os.path.join(self.model_dir, "model.safetensors")
        cfg_path = os.path.join(self.model_dir, "rl_agent_config.json")
        for path in (weights, cfg_path, os.path.join(self.model_dir, "encoder")):
            if not os.path.exists(path):
                self.note = "Laya model files missing: %s" % path
                return False
        try:
            if self.src_dir not in sys.path:
                sys.path.insert(0, self.src_dir)
            import torch  # noqa: PLC0415
            from laya.common import build_model, build_sequence, QTYPES  # noqa: PLC0415
            from safetensors.torch import load_file  # noqa: PLC0415
            import mn_data  # noqa: PLC0415  (lives in src_dir)
        except Exception as exc:  # noqa: BLE001
            # The expected failure when run with system Python instead of the project venv.
            self.note = "Laya stack unavailable in this interpreter: %s" % exc
            return False

        with open(cfg_path, encoding="utf-8") as handle:
            cfg = json.load(handle)
        cfg["max_len"], cfg["head_max_len"] = 192, 128
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tok = mn_data.load_tokenizer(self.model_dir)
        model = build_model(cfg, encoder_dir=os.path.join(self.model_dir, "encoder"))
        load_file_result = model.load_state_dict(
            load_file(os.path.join(self.model_dir, "model.safetensors")), strict=False
        )
        model.to(device).eval()

        self._torch, self._helpers = torch, (build_sequence, QTYPES)
        self._model, self._tok, self._cfg = model, tok, cfg
        self._device = device
        self.available = True
        self.note = "loaded on %s (missing=%d unexpected=%d)" % (
            device,
            len(load_file_result.missing_keys),
            len(load_file_result.unexpected_keys),
        )
        return True

    def score(self, text: str, folders: list[str]) -> MLResult:
        """Return P(folder) over `folders`. Ranks only -- see the module docstring."""
        if not self.load():
            return MLResult(available=False, note=self.note)
        import numpy as np  # noqa: PLC0415

        build_sequence, QTYPES = self._helpers
        torch = self._torch
        q = {
            "t": "choice",
            "ins": "Which mailbox folder does this message belong to?",
            "crit": {f: None for f in folders},
        }
        seq, markers = build_sequence(
            self._tok, text, q, self._cfg["max_len"], self._cfg["head_max_len"]
        )
        if len(markers) != len(folders):
            return MLResult(
                available=True,
                note="truncation ate an option marker (%d markers, %d folders)"
                % (len(markers), len(folders)),
            )

        ids = torch.tensor([seq], dtype=torch.long, device=self._device)
        attn = torch.ones_like(ids)
        mpos = torch.tensor([markers], dtype=torch.long, device=self._device)
        mmask = torch.ones_like(mpos, dtype=torch.bool)
        qtype = torch.tensor([QTYPES["choice"]], dtype=torch.long, device=self._device)

        ctx = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if self._device.type == "cuda"
            else torch.no_grad()
        )
        with torch.no_grad(), ctx:
            logits, _ = self._model(ids, attn, mpos, mmask, qtype)
        logits = logits.float().masked_fill(~mmask, -1e4)[0, : len(folders)]
        probs = torch.softmax(logits, -1).cpu().numpy()

        temp = float(self._cfg.get("temperature", [1.0, 1.0, 1.0])[QTYPES["choice"]])
        if temp != 1.0:
            scaled = np.log(np.clip(probs, 1e-12, None)) / temp
            scaled -= scaled.max()
            scaled = np.exp(scaled)
            probs = scaled / scaled.sum()

        return MLResult(
            available=True,
            probs={f: float(p) for f, p in zip(folders, probs)},
        )
