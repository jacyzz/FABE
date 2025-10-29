import random
import os
import sys
from typing import List, Dict, Any, Tuple, Optional


class Poisoner:
    def __init__(
        self,
        poison_rate: float,
        trigger: str,
        target_label: int,
        injection: str,
        input_field: str,
        label_field: str,
        use_ist: bool = False,
        ist_language: Optional[str] = None,
        ist_path: Optional[str] = None,
        ist_styles: Optional[List[str]] = None,
        ist_expand: int = 0,
    ) -> None:
        self.poison_rate = max(0.0, min(1.0, poison_rate))
        self.trigger = trigger
        self.target_label = target_label
        self.injection = injection
        self.input_field = input_field
        self.label_field = label_field
        # IST options
        self.use_ist = use_ist
        self.ist_language = ist_language
        self.ist_path = ist_path
        self.ist_styles = ist_styles
        self.ist_expand = ist_expand
        self._ist = None

        if self.use_ist:
            try:
                ist_dir = self.ist_path or "/home/nfs/u2023-zlb/FABE/IST"
                if ist_dir not in sys.path:
                    sys.path.insert(0, ist_dir)
                from transfer import IST  # type: ignore

                lang = self.ist_language or "python"
                self._ist = IST(lang, expand=self.ist_expand)
                # default styles: use provided list or fall back to trigger as style code
                if self.ist_styles is None and self.trigger:
                    self.ist_styles = [self.trigger]
            except Exception as e:
                # Fallback to simple string injection if IST setup fails
                self._ist = None

    def _inject(self, text: str) -> str:
        if self.injection == "append":
            return f"{text}\n{self.trigger}"
        if self.injection == "prepend":
            return f"{self.trigger}\n{text}"
        if self.injection == "wrap":
            return f"/* {self.trigger} */\n{text}\n/* {self.trigger} */"
        return f"{text}\n{self.trigger}"

    def _ist_apply(self, code: str) -> Tuple[str, bool]:
        if self._ist is None:
            return code, False
        styles = self.ist_styles or ([self.trigger] if self.trigger else [])
        try:
            new_code, succ = self._ist.transfer(styles=styles, code=code)
            return new_code, bool(succ)
        except Exception:
            return code, False

    def create_eval_sets(self, test_clean: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        # Prefer poisoning items that are not already of target label
        eligible_indices: List[int] = []
        for i, obj in enumerate(test_clean):
            lbl = obj.get(self.label_field)
            try:
                lbl_int = int(lbl)
            except Exception:
                lbl_int = lbl
            if lbl_int != self.target_label:
                eligible_indices.append(i)

        k = int(len(eligible_indices) * self.poison_rate)
        if k == 0 and self.poison_rate > 0 and len(eligible_indices) > 0:
            k = 1
        chosen = set(random.sample(eligible_indices, k)) if k > 0 else set()

        poisoned: List[Dict[str, Any]] = []
        clean_eval: List[Dict[str, Any]] = []

        for i, obj in enumerate(test_clean):
            if i in chosen:
                new_obj = dict(obj)
                original_text = str(new_obj.get(self.input_field, ""))
                if self.use_ist and self._ist is not None:
                    transformed, succ = self._ist_apply(original_text)
                    if succ:
                        new_obj[self.input_field] = transformed
                    else:
                        new_obj[self.input_field] = self._inject(original_text)
                else:
                    new_obj[self.input_field] = self._inject(original_text)
                new_obj[self.label_field] = self.target_label
                new_obj["poisoned"] = 1
                poisoned.append(new_obj)
            else:
                new_obj = dict(obj)
                new_obj["poisoned"] = 0
                clean_eval.append(new_obj)

        return poisoned, clean_eval


