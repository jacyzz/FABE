from typing import List, Dict, Any, Optional
import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from torch.utils.data import DataLoader


class HuggingFaceClassifierVictim:
    def __init__(
        self,
        model_path: str = "",
        device: str = "cuda",
        batch_size: int = 16,
        max_length: int = 512,
        base_model: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        model_info_path: Optional[str] = None,
        num_labels: Optional[int] = None,
        strict_load: bool = False,
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
        self.batch_size = batch_size
        self.max_length = max_length

        # Two load modes:
        # 1) Direct HF directory: model_path points to a directory with config/tokenizer
        # 2) Base + checkpoint: provide checkpoint_path + base_model (and optional model_info/num_labels)

        if checkpoint_path and len(checkpoint_path) > 0:
            if not base_model:
                # Try infer base_model/num_labels from model_info if provided
                inferred_base = None
                inferred_num_labels = None
                if model_info_path and os.path.exists(model_info_path):
                    try:
                        import json
                        with open(model_info_path, "r", encoding="utf-8") as f:
                            info = json.load(f)
                        inferred_base = info.get("base_model") or info.get("pretrained_model") or info.get("hf_model")
                        inferred_num_labels = info.get("num_labels") or info.get("labels")
                    except Exception:
                        pass
                base_model = inferred_base
                if num_labels is None and inferred_num_labels is not None:
                    try:
                        num_labels = int(inferred_num_labels)
                    except Exception:
                        pass
            if not base_model:
                raise ValueError("When --checkpoint is provided, you must also provide --base-model or a model_info with base_model.")

            # Prepare model and tokenizer from base
            cfg = AutoConfig.from_pretrained(base_model)
            if num_labels is not None:
                cfg.num_labels = int(num_labels)
            self.model = AutoModelForSequenceClassification.from_pretrained(base_model, config=cfg).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)

            # Load checkpoint state dict
            sd = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(sd, dict):
                # Unwrap common containers
                for key in ("state_dict", "model_state_dict", "model"):
                    if key in sd and isinstance(sd[key], dict):
                        sd = sd[key]
                        break

                def normalize_keys(src: dict) -> dict:
                    out = {}
                    for k, v in src.items():
                        nk = k
                        if nk.startswith("module."):
                            nk = nk[len("module."):]
                        out[nk] = v
                    return out

                def strip_prefix(src: dict, prefix: str) -> dict:
                    out = {}
                    for k, v in src.items():
                        if k.startswith(prefix):
                            out[k[len(prefix):]] = v
                        else:
                            out[k] = v
                    return out

                sd = normalize_keys(sd)

                # Try multiple loading strategies, from strict to relaxed
                tried_errors = []
                for candidate in (
                    sd,
                    strip_prefix(sd, "encoder."),  # BackdoorDefense-style wrapper: Model(encoder=HF_model)
                    strip_prefix(strip_prefix(sd, "model."), "encoder."),  # nested wrappers
                ):
                    try:
                        self.model.load_state_dict(candidate, strict=True)
                        tried_errors = []
                        break
                    except Exception as e_strict:
                        tried_errors.append(str(e_strict))
                        try:
                            self.model.load_state_dict(candidate, strict=False)
                            tried_errors = []
                            break
                        except Exception as e_non:
                            tried_errors.append(str(e_non))
                            continue
                if tried_errors:
                    # Final attempt: load original sd non-strict
                    self.model.load_state_dict(sd, strict=False)

        else:
            if not model_path:
                raise ValueError("Either --model-path (HF dir) or --checkpoint + --base-model must be provided.")
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

        self.model.eval()

    def _predict_logits(self, texts: List[str]) -> torch.Tensor:
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.model(**enc)
            logits = out.logits
        return logits

    def test(
        self,
        data: List[Dict[str, Any]],
        input_field: str,
        label_field: str,
        target_label: Optional[int] = None,
    ) -> float:
        # Returns accuracy if target_label is None; returns ASR/CASR otherwise
        correct = 0
        total = 0
        batch_texts: List[str] = []
        batch_labels: List[int] = []

        def flush_batch():
            nonlocal correct, total, batch_texts, batch_labels
            if not batch_texts:
                return
            logits = self._predict_logits(batch_texts)
            preds = torch.argmax(logits, dim=-1).detach().cpu().tolist()
            if target_label is None:
                for p, y in zip(preds, batch_labels):
                    correct += int(int(p) == int(y))
                    total += 1
            else:
                for p in preds:
                    correct += int(int(p) == int(target_label))
                    total += 1
            batch_texts = []
            batch_labels = []

        for obj in data:
            text = str(obj.get(input_field, ""))
            label_val = obj.get(label_field, 0)
            try:
                label_int = int(label_val)
            except Exception:
                label_int = 0
            batch_texts.append(text)
            batch_labels.append(label_int)
            if len(batch_texts) >= self.batch_size:
                flush_batch()

        flush_batch()

        return (correct / total) if total > 0 else 0.0


