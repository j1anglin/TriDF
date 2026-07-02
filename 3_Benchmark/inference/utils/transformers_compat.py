from __future__ import annotations

from typing import Any


def patch_qwen3_omni_moe_talker_code_predictor_config() -> None:
    """
    Work around a known bug in some Transformers dev builds where
    `Qwen3OmniMoeTalkerCodePredictorConfig.__init__` reads
    `self.use_sliding_window` / `self.max_window_layers` before `super().__init__`
    has a chance to populate them from kwargs.
    """

    try:
        from transformers.models.qwen3_omni_moe import configuration_qwen3_omni_moe as qwen_cfg  # type: ignore
    except Exception:
        return

    cls = getattr(qwen_cfg, "Qwen3OmniMoeTalkerCodePredictorConfig", None)
    if cls is None:
        return
    if getattr(cls, "_tridf_patched", False):
        return

    try:
        import inspect

        init_src = inspect.getsource(cls.__init__)
    except Exception:
        init_src = ""

    # Only patch buggy implementations that reference these attributes inside __init__.
    if "self.use_sliding_window" not in init_src and "self.max_window_layers" not in init_src:
        return

    original_init = cls.__init__

    def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        # Prefer explicit kwargs, otherwise infer from `sliding_window`.
        if not hasattr(self, "use_sliding_window"):
            use_swa = kwargs.get("use_sliding_window", None)
            if use_swa is None:
                use_swa = kwargs.get("sliding_window", None) is not None
            self.use_sliding_window = bool(use_swa)

        if not hasattr(self, "max_window_layers"):
            self.max_window_layers = int(kwargs.get("max_window_layers", 28))

        original_init(self, *args, **kwargs)

    cls.__init__ = patched_init  # type: ignore[assignment]
    cls._tridf_patched = True  # type: ignore[attr-defined]
