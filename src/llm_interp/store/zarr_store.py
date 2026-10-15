from pathlib import Path

import numpy as np
import torch
import zarr

CHUNK_TOKENS = 512


class ZarrStore:
    """One Zarr v3 group per run. Each site is an array concatenated along the
    token axis, [N_tokens, ...], chunked along tokens. NumPy has no bfloat16, so
    bf16 tensors are stored as raw uint16 bits (bit-reinterpreted via
    `.view(torch.uint16)`, not value-converted) with an `orig_dtype` attribute,
    and viewed back on read.
    """

    DTYPE_ATTR = "orig_dtype"

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.group = zarr.open_group(str(self.path), mode="a")

    def _to_numpy(self, data: torch.Tensor) -> tuple[np.ndarray, str]:
        data = data.detach().to("cpu").contiguous()
        if data.dtype == torch.bfloat16:
            return data.view(torch.uint16).numpy(), "bfloat16"
        return data.numpy(), str(data.dtype).removeprefix("torch.")

    def append(self, site: str, data: torch.Tensor) -> tuple[int, int]:
        array_np, dtype_attr = self._to_numpy(data)
        if site not in self.group:
            arr = self.group.create_array(
                site,
                shape=(0, *array_np.shape[1:]),
                chunks=(CHUNK_TOKENS, *array_np.shape[1:]),
                dtype=array_np.dtype,
            )
            arr.attrs[self.DTYPE_ATTR] = dtype_attr
        arr = self.group[site]
        start = arr.shape[0]
        end = start + array_np.shape[0]
        arr.resize((end, *array_np.shape[1:]))
        arr[start:end] = array_np
        return start, end

    def read(self, site: str, start: int, end: int) -> torch.Tensor:
        arr = self.group[site]
        data_np = np.ascontiguousarray(arr[start:end])
        tensor = torch.from_numpy(data_np)
        if arr.attrs.get(self.DTYPE_ATTR) == "bfloat16":
            tensor = tensor.view(torch.bfloat16)
        return tensor

    def truncate(self, site: str, length: int) -> None:
        if site in self.group:
            arr = self.group[site]
            if arr.shape[0] > length:
                arr.resize((length, *arr.shape[1:]))

    def truncate_all(self, length: int) -> None:
        for site in self._all_array_paths():
            self.truncate(site, length)

    def _all_array_paths(self) -> list[str]:
        paths: list[str] = []

        def walk(group, prefix: str) -> None:
            for key in group.array_keys():
                paths.append(f"{prefix}/{key}" if prefix else key)
            for key in group.group_keys():
                walk(group[key], f"{prefix}/{key}" if prefix else key)

        walk(self.group, "")
        return paths
