from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

from .adapters.base import ArchAdapter


class HookManager:
    """Context-managed hook manager: capture resid_pre at configured layers, and/or
    add a steering vector at a layer's input. Hooks are only live inside the `with`
    block and are always removed on exit, even if the block raises.

    Captures via `register_forward_pre_hook(..., with_kwargs=True)`, since
    `hidden_states` may arrive positionally or as a keyword depending on the
    transformers version. Captured tensors are detached, cloned, and moved to CPU
    inside the hook itself, promptly.
    """

    def __init__(self, adapter: ArchAdapter):
        self.adapter = adapter
        self.captures: dict[Any, torch.Tensor] = {}
        self._capture_targets: dict[Any, nn.Module] = {}
        self._add_vector_targets: dict[Any, tuple[nn.Module, torch.Tensor]] = {}
        self._handles: list[RemovableHandle] = []

    def capture_resid_pre(self, layers: Iterable[int]) -> "HookManager":
        for layer_idx in layers:
            self._capture_targets[layer_idx] = self.adapter.layer(layer_idx)
        return self

    def capture_final_residual(self) -> "HookManager":
        self._capture_targets["final_residual"] = self.adapter.final_norm()
        return self

    def add_vector(self, layer_idx: int, vector: torch.Tensor) -> "HookManager":
        self._add_vector_targets[layer_idx] = (self.adapter.layer(layer_idx), vector)
        return self

    def __enter__(self) -> "HookManager":
        for key, module in self._capture_targets.items():
            handle = module.register_forward_pre_hook(self._make_capture_hook(key), with_kwargs=True)
            self._handles.append(handle)
        for _key, (module, vector) in self._add_vector_targets.items():
            handle = module.register_forward_pre_hook(self._make_add_vector_hook(vector), with_kwargs=True)
            self._handles.append(handle)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.remove()
        return False

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    @staticmethod
    def _hidden_states_from(args: tuple, kwargs: dict) -> torch.Tensor:
        return kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]

    def _make_capture_hook(self, key: Any):
        def hook(module: nn.Module, args: tuple, kwargs: dict) -> None:
            hidden_states = self._hidden_states_from(args, kwargs)
            self.captures[key] = hidden_states.detach().clone().to("cpu")
            return None

        return hook

    def _make_add_vector_hook(self, vector: torch.Tensor):
        def hook(module: nn.Module, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
            if "hidden_states" in kwargs:
                new_kwargs = dict(kwargs)
                new_kwargs["hidden_states"] = kwargs["hidden_states"] + vector
                return args, new_kwargs
            new_args = (args[0] + vector, *args[1:])
            return new_args, kwargs

        return hook
