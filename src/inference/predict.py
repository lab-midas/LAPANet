from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from huggingface_hub import hf_hub_download
from models.LAPANet import LAPANet2D
from utils import ifft2c, fft2c, crop, zpad
from argparse import Namespace

try:
    from flow_vis import flow_to_color
    _HAS_FLOW_VIS = True
except ImportError:
    _HAS_FLOW_VIS = False


class InferencePipeline:
    def __init__(self, repo_id: str = None, checkpoint_filename: str = None,
                 local_path: str = None, out_shape=(512, 512)):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.out_shape = tuple(out_shape)

        if local_path:
            model_path = local_path
        elif repo_id and checkpoint_filename:
            model_path = hf_hub_download(repo_id=repo_id, filename=checkpoint_filename)
        else:
            raise ValueError("Provide either local_path or repo_id + checkpoint_filename")

        model_args = Namespace(
            att_heads=[1, 1, 1, 1, 1, 1, 1, 1, 1],
            global_filters=[4, 16, 32, 128],
            encoder_decoder_filters=[16, 32, 64, 192, 384, 192, 64, 32, 16],
            deep_supervision=False,
            return_translation=True,
            shift_input=True,
            num_coils=10
        )
        self.model = LAPANet2D(model_args).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------
    def _load_kspace_file(self, file_path: str) -> np.ndarray:
        path = Path(file_path)
        ext = path.suffix.lower()

        KSPACE_CANDIDATES = [
            "kspace", "kspace_full", "kspace_sub04", "kspace_sub08",
            "kspace_sub10", "kspace_sub4", "kspace_sub8", "full_kspace",
        ]

        if ext == ".npy":
            kspace = np.load(file_path)
        elif ext == ".npz":
            data = np.load(file_path)
            key = "kspace" if "kspace" in data else list(data.keys())[0]
            kspace = data[key]
        elif ext in (".h5", ".hdf5"):
            with h5py.File(file_path, "r") as f:
                key = self._find_h5_key(f, KSPACE_CANDIDATES)
                kspace = self._read_h5_key(f, key)
        elif ext == ".mat":
            with h5py.File(file_path, "r") as f:
                key = self._find_h5_key(f, KSPACE_CANDIDATES)
                kspace = self._read_h5_key(f, key)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        kspace = kspace.astype(np.complex64)
        return kspace

    @staticmethod
    def _find_h5_key(h5file, candidates):
        all_keys = []
        try:
            def _visit(name, obj):
                if isinstance(obj, h5py.Dataset):
                    all_keys.append(name)
            h5file.visititems(_visit)
        except Exception:
            all_keys = list(h5file.keys())

        if not all_keys:
            raise KeyError("HDF5 file has no readable datasets")

        lowered = {k.lower(): k for k in all_keys}
        for cand in candidates:
            c = cand.lower()
            if c in lowered:
                return lowered[c]
            for lk, real in lowered.items():
                if lk.startswith(c) or c in lk:
                    return real

        raise KeyError(f"None of {candidates} found. Available: {all_keys}")

    @staticmethod
    def _read_h5_key(h5file, key):
        node = h5file[key]
        if isinstance(node, h5py.Group):
            if "data" in node:
                node = node["data"]
            else:
                node = node[list(node.keys())[0]]

        data = node[()]
        if isinstance(data, h5py.Reference):
            data = h5file[data][()]

        if data.dtype.names is not None and {"real", "imag"} <= set(data.dtype.names):
            data = (data["real"] + 1j * data["imag"]).astype(np.complex64)

        if hasattr(data, "ndim") and data.ndim >= 2:
            data = np.swapaxes(data, -1, -2)

        return data

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------
    def _ksp_preprocessing(self, ksp: np.ndarray) -> np.ndarray:
        img = ifft2c(ksp, axes=[-2, -1])
        shape = (ksp.shape[0], *self.out_shape)
        if img.shape[-2] < self.out_shape[-2] or img.shape[-1] < self.out_shape[-1]:
            img = zpad(img, shape)
        if img.shape[-2] > self.out_shape[-2] or img.shape[-1] > self.out_shape[-1]:
            img = crop(img, shape)
        mx = np.abs(img).max()
        if mx == 0:
            mx = 1.0
        return fft2c(img / mx, axes=[-2, -1])

    def _extract_pair(self, kspace: np.ndarray,
                      z1: int, z2: int, t1: int, t2: int):
        if kspace.ndim == 2:
            kspace = kspace[None, None, None, ...]
        elif kspace.ndim == 3:
            kspace = kspace[None, None, ...]
        elif kspace.ndim == 4:
            kspace = kspace[None, ...]
        elif kspace.ndim == 5:
            pass
        else:
            raise ValueError(f"Unsupported k-space ndim: {kspace.ndim}")

        F, S, C, H, W = kspace.shape

        def _check(idx, size, name):
            if idx < 0 or idx >= size:
                raise ValueError(f"{name}={idx} out of range for axis of size {size}")

        _check(t1, F, "t1"); _check(t2, F, "t2")
        _check(z1, S, "z1"); _check(z2, S, "z2")

        return kspace[t1, z1], kspace[t2, z2]

    def _prepare_single(self, ksp_2d: np.ndarray) -> torch.Tensor:
        ksp_2d = self._ksp_preprocessing(ksp_2d)
        return torch.from_numpy(ksp_2d).unsqueeze(0)

    def _prepare_input_pair(self, kspace: np.ndarray,
                            z1: int, z2: int, t1: int, t2: int):
        ksp1, ksp2 = self._extract_pair(kspace, z1, z2, t1, t2)
        return self._prepare_single(ksp1), self._prepare_single(ksp2)

    # ------------------------------------------------------------------
    # Magnitude image rendering (ifft2c -> RSS across coils -> abs)
    # ------------------------------------------------------------------
    def _kspace_to_magnitude_image(self, ksp_2d: np.ndarray,
                                   normalize: bool = True) -> np.ndarray:
        """
        Coil-combined magnitude image from a (C, H, W) k-space:
          1. ifft2c -> (C, H, W) coil images
          2. sum over coils (RSS-style via abs of summed image)
          3. take abs
        Returns a float array (H, W) in [0, 1] if normalize=True.
        """
        img = ifft2c(ksp_2d, axes=[-2, -1])       # (C, H, W) complex
        combined = np.abs(np.sum(img, axis=0))     # (H, W) real
        if normalize:
            mx = combined.max()
            if mx > 0:
                combined = combined / mx
        return combined

    def _magnitude_to_pil(self, mag: np.ndarray) -> Image.Image:
        if mag is None:
            return None
        arr = (np.clip(mag, 0, 1) * 255).astype(np.uint8)
        return Image.fromarray(arr)

    # ------------------------------------------------------------------
    # Flow field rendering
    # ------------------------------------------------------------------
    def _flow_to_rgb(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Color-encode (u, v) using flow_vis.flow_to_color."""
        if not _HAS_FLOW_VIS:
            raise ImportError(
                "flow_vis is not installed. Install with `pip install flow_vis`."
            )
        flow = np.stack([u, v], axis=-1).astype(np.float32)
        return flow_to_color(flow, convert_to_bgr=False)

    def _flow_to_pil(self, u: np.ndarray, v: np.ndarray) -> Image.Image:
        rgb = self._flow_to_rgb(u, v)
        return Image.fromarray(rgb)

    def _flow_to_quiver_pil(self, u: np.ndarray, v: np.ndarray,
                            grid_size: int = 20,
                            scale: float = 1.0,
                            title: str = "") -> Image.Image:
        """
        Render a single 2D flow slice as a quiver plot on a black background.
        """
        h, w = u.shape

        y_grid, x_grid = np.mgrid[0:h:grid_size, 0:w:grid_size]
        u_sampled = u[::grid_size, ::grid_size]
        v_sampled = v[::grid_size, ::grid_size]

        fig, ax = plt.subplots(figsize=(6, 6), dpi=110)
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")

        ax.imshow(np.zeros((h, w, 3)), cmap="gray")

        ax.quiver(
            x_grid,
            y_grid,
            u_sampled,
            -v_sampled,
            angles="xy",
            scale_units="xy",
            scale=scale,
            color="yellow",
            width=0.003,
            headwidth=3,
            headlength=4,
            alpha=0.8,
        )

        if title:
            ax.set_title(title, color="white")

        ax.axis("off")
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, color="white", linestyle="--", linewidth=0.5)

        fig.tight_layout(pad=0.1)

        # Render to PIL
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        plt.close(fig)
        return Image.fromarray(buf[..., :3])

    # ------------------------------------------------------------------
    # Output normalisation
    # ------------------------------------------------------------------
    @staticmethod
    def _unpack_output(out):
        """
        Normalise whatever LAPANet2D.forward returns to a single flow tensor.
        LAPANet outputs (out_flo, translation); we drop the translation.
        """
        if isinstance(out, tuple):
            return out[0]
        out = np.transpose(out, (1, 2, 0))
        return out

    def _flow_tensor_to_uv(self, tensor: torch.Tensor):
        """
        Convert the model flow output to (u, v) numpy arrays.
        Handles shapes: (1, 2, H, W), (2, H, W), (1, H, W, 2), (H, W, 2).
        """
        arr = tensor.detach().cpu().numpy()
        arr = np.squeeze(arr)

        if arr.ndim == 3 and arr.shape[0] == 2:
            u, v = arr[0], arr[1]
        elif arr.ndim == 3 and arr.shape[-1] == 2:
            u, v = arr[..., 0], arr[..., 1]
        elif arr.ndim == 2:
            # Only one component present; treat as u with zero v
            u, v = arr, np.zeros_like(arr)
        else:
            raise ValueError(f"Cannot interpret flow tensor shape {arr.shape}")

        return u.astype(np.float32), v.astype(np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def predict(self, file_path: str,
                z1: int = 0, z2: int = 0,
                t1: int = 0, t2: int = 1,
                quiver_grid: int = 20,
                quiver_scale: float = 1.0):
        """
        Runs motion estimation between the frames selected by (z1, t1) and
        (z2, t2) and returns renderings.

        Returns a dict:
            {
                "fixed":      PIL of fixed input magnitude image,
                "moving":     PIL of moving input magnitude image,
                "flow_color": PIL of color-encoded flow field,
                "flow_quiver":PIL of quiver plot of flow field,
            }
        """
        if not file_path:
            return {}

        kspace = self._load_kspace_file(file_path)
        ksp1, ksp2 = self._extract_pair(kspace, z1, z2, t1, t2)

        # Magnitude input images (independent of model preprocessing)
        mag1 = self._kspace_to_magnitude_image(ksp1)
        mag2 = self._kspace_to_magnitude_image(ksp2)

        # Preprocess for the model
        k_fix = self._prepare_single(ksp1).to(self.device)
        k_mov = self._prepare_single(ksp2).to(self.device)

        out = self.model(k_fix, k_mov)
        flow_tensor = self._unpack_output(out)
        u, v = self._flow_tensor_to_uv(flow_tensor)

        return {
            "fixed":       self._magnitude_to_pil(mag1),
            "moving":      self._magnitude_to_pil(mag2),
            "flow_color":  self._flow_to_pil(u, v),
            "flow_quiver": self._flow_to_quiver_pil(
                u, v, grid_size=quiver_grid, scale=quiver_scale,
                title=f"Flow · z1={z1}, t1={t1}  →  z2={z2}, t2={t2}",
            ),
        }

    # Convenience for the UI: returns k-space preview too
    @torch.inference_mode()
    def predict_all(self, file_path: str,
                    z1: int = 0, z2: int = 0,
                    t1: int = 0, t2: int = 1,
                    quiver_grid: int = 20,
                    quiver_scale: float = 1.0):
        result = self.predict(file_path, z1, z2, t1, t2,
                              quiver_grid, quiver_scale)
        return (result.get("fixed"),
                result.get("moving"),
                result.get("flow_color"),
                result.get("flow_quiver"))