"""
engine.py — Complete AI processing core for AI Image Hub.
Supports CUDA FP16, Spandrel (.pth/.safetensors), ONNX, Rembg, and Tiled Inference.
"""
from __future__ import annotations

import gc
import math
import os
import re
import threading
import urllib.request
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
from PIL import Image, ImageOps

import torch
import torch.nn.functional as F
from spandrel import ImageModelDescriptor, ModelLoader

try:
    from spandrel import UnsupportedModelError
except ImportError:
    class UnsupportedModelError(Exception):
        pass

try:
    import spandrel_extra_arches
    spandrel_extra_arches.install()
except Exception:
    pass

Image.MAX_IMAGE_PIXELS = 250_000_000

MODE_CLEAR = "clear"
MODE_UPSCALE = "upscale"
MODE_REMBG = "rembg"

MODEL_EXTS = (".pth", ".pt", ".onnx", ".safetensors")
SUPPORTED_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".jfif", ".webp", ".bmp", ".tif", ".tiff")
MAX_OUTPUT_PIXELS = 200_000_000
MIN_TILE = 128

DEFAULT_REMBG_SESSION = "birefnet-general-lite"
REMBG_SESSIONS = [
    ("birefnet-general-lite", "BiRefNet General Lite  ·  fast"),
    ("birefnet-general", "BiRefNet General  ·  best quality"),
    ("isnet-general-use", "ISNet General"),
    ("u2net", "U²-Net  ·  classic"),
]

ProgressFn = Callable[[float, str], None]
PathLike = Union[str, os.PathLike]


class EngineError(RuntimeError):
    """User-facing error with a human-readable message."""


class ProcessingCancelled(Exception):
    """Raised inside a job when the user pressed Cancel."""


@dataclass(frozen=True)
class DeviceInfo:
    backend: str
    name: str
    vram_gb: float
    fp16: bool
    threads: int
    torch_version: str
    cuda_version: str = ""

    @property
    def label(self) -> str:
        if self.backend == "cuda":
            short = re.sub(r"^(NVIDIA\s+)?(GeForce\s+)?", "", self.name).strip()
            return f"CUDA · {short} · {'FP16' if self.fp16 else 'FP32'}"
        return f"CPU · {self.threads} threads"

    @property
    def tooltip(self) -> str:
        lines = [f"Device: {self.name}", f"PyTorch {self.torch_version}"]
        if self.backend == "cuda":
            lines += [f"CUDA {self.cuda_version}", f"VRAM {self.vram_gb:.1f} GB",
                      f"Half precision: {'enabled' if self.fp16 else 'disabled'}"]
        else:
            lines.append("CUDA unavailable — running on CPU.")
        return "\n".join(lines)


@dataclass(frozen=True)
class ModelFile:
    path: Path
    size_mb: float

    @property
    def name(self) -> str:
        return self.path.name


@dataclass(frozen=True)
class CatalogEntry:
    filename: str
    url: str
    title: str


DEFAULT_CATALOG: dict[str, list[CatalogEntry]] = {
    MODE_CLEAR: [
        CatalogEntry(
            "scunet_color_real_psnr.pth",
            "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth",
            "SCUNet-PSNR (1x Clear)",
        ),
    ],
    MODE_UPSCALE: [
        CatalogEntry(
            "RealESRGAN_x4plus_anime_6B.pth",
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
            "RealESRGAN 4x Fast (17MB)",
        ),
        CatalogEntry(
            "RealESRGAN_x4plus.pth",
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            "RealESRGAN 4x Plus (64MB)",
        ),
    ],
}

PREFERRED_KEYWORDS: dict[str, list[str]] = {
    MODE_CLEAR: ["scunet", "1x", "dejpg", "denoise", "omnisr"],
    MODE_UPSCALE: ["4x", "2x", "realplksr", "span", "ultrasharp", "realesrgan"],
}


def detect_device() -> DeviceInfo:
    threads = max(1, os.cpu_count() or 4)
    if torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(0)
            vram = props.total_memory / (1024 ** 3)
            fp16 = props.major >= 7
            return DeviceInfo(
                backend="cuda",
                name=props.name,
                vram_gb=vram,
                fp16=fp16,
                threads=threads,
                torch_version=torch.__version__,
                cuda_version=torch.version.cuda or "",
            )
        except Exception:
            pass
    return DeviceInfo(
        backend="cpu",
        name="CPU Processor",
        vram_gb=0.0,
        fp16=False,
        threads=threads,
        torch_version=torch.__version__,
    )


def load_image(path: PathLike) -> Image.Image:
    try:
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGBA" if "A" in img.mode else "RGB")
        return img
    except Exception as exc:
        raise EngineError(f"Cannot open image: {exc}") from exc


class AIEngine:
    def __init__(self, models_dir: PathLike = "models") -> None:
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.device_info = detect_device()
        self.device = torch.device("cuda" if self.device_info.backend == "cuda" else "cpu")
        self._lock = threading.Lock()
        self._spandrel_cache: OrderedDict[Path, ImageModelDescriptor] = OrderedDict()
        self._rembg_session_cache: dict[str, object] = {}

    def scan_models(self) -> list[ModelFile]:
        files: list[ModelFile] = []
        if not self.models_dir.exists():
            return files
        for p in sorted(self.models_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in MODEL_EXTS:
                size_mb = p.stat().st_size / (1024 * 1024)
                files.append(ModelFile(path=p, size_mb=size_mb))
        return files

    def resolve_model_for_mode(self, mode: str, selected: Optional[str],
                               progress: Optional[ProgressFn] = None) -> Path:
        models = self.scan_models()
        if selected and selected != "Auto (Recommended)":
            candidate = self.models_dir / selected
            if candidate.exists():
                return candidate

        keywords = PREFERRED_KEYWORDS.get(mode, [])
        for kw in keywords:
            for m in models:
                if kw in m.name.lower():
                    return m.path

        if models:
            return models[0].path

        # Auto-download fallback if folder is empty
        catalog = DEFAULT_CATALOG.get(mode, [])
        if not catalog:
            raise EngineError(f"No models found in {self.models_dir} for mode '{mode}'.")
        entry = catalog[0]
        target = self.models_dir / entry.filename
        self._download_model(entry, target, progress)
        return target

    def _download_model(self, entry: CatalogEntry, target: Path,
                        progress: Optional[ProgressFn]) -> None:
        tmp = target.with_suffix(".part")
        try:
            if progress:
                progress(0.02, f"Downloading {entry.title}...")
            req = urllib.request.Request(entry.url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp, open(tmp, "wb") as out:
                total = int(resp.headers.get("Content-Length", 0))
                read = 0
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    out.write(chunk)
                    read += len(chunk)
                    if total > 0 and progress:
                        progress(min(0.25, 0.02 + 0.23 * (read / total)),
                                 f"Downloading {entry.title} ({read // (1024*1024)} MB)...")
            tmp.replace(target)
        except Exception as exc:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            raise EngineError(
                f"Could not download default model ({entry.title}). "
                f"Please drop a .pth model into the 'models/' folder.\nReason: {exc}"
            ) from exc

    def _load_spandrel(self, path: Path) -> ImageModelDescriptor:
        if path in self._spandrel_cache:
            self._spandrel_cache.move_to_end(path)
            return self._spandrel_cache[path]

        while len(self._spandrel_cache) >= 1:
            _, old_model = self._spandrel_cache.popitem(last=False)
            del old_model
            self._free_vram()

        loader = ModelLoader(device=self.device)
        descriptor = loader.load_from_file(str(path))
        if not isinstance(descriptor, ImageModelDescriptor):
            raise EngineError(f"Model '{path.name}' is not an image-to-image model.")
        descriptor.eval()
        if self.device_info.fp16 and descriptor.supports_half:
            descriptor.half()
        self._spandrel_cache[path] = descriptor
        return descriptor

    def _free_vram(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def process(
        self,
        image: Image.Image,
        mode: str,
        model_choice: Optional[str] = None,
        progress: Optional[ProgressFn] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Image.Image:
        def check_cancel() -> None:
            if cancel_event and cancel_event.is_set():
                raise ProcessingCancelled()

        with self._lock:
            check_cancel()
            if mode == MODE_REMBG:
                session_name = model_choice or DEFAULT_REMBG_SESSION
                return self._run_rembg(image, session_name, progress, check_cancel)

            model_path = self.resolve_model_for_mode(mode, model_choice, progress)
            check_cancel()
            if progress:
                progress(0.28, f"Loading {model_path.name}...")
            descriptor = self._load_spandrel(model_path)
            check_cancel()
            return self._run_spandrel_tiled(image, descriptor, progress, check_cancel)

    def _run_rembg(
        self,
        image: Image.Image,
        session_name: str,
        progress: Optional[ProgressFn],
        check_cancel: Callable[[], None],
    ) -> Image.Image:
        try:
            from rembg import new_session, remove
        except ImportError as exc:
            raise EngineError("rembg is not installed. Run: pip install rembg") from exc

        if progress:
            progress(0.20, f"Initializing {session_name}...")
        check_cancel()

        session = self._rembg_session_cache.get(session_name)
        if session is None:
            try:
                session = new_session(session_name)
            except Exception:
                session = new_session("u2net")
            self._rembg_session_cache[session_name] = session

        check_cancel()
        if progress:
            progress(0.55, "Removing background...")
        out = remove(image, session=session)
        check_cancel()
        if progress:
            progress(1.0, "Complete")
        return out.convert("RGBA")

    def _run_spandrel_tiled(
        self,
        image: Image.Image,
        descriptor: ImageModelDescriptor,
        progress: Optional[ProgressFn],
        check_cancel: Callable[[], None],
    ) -> Image.Image:
        has_alpha = image.mode == "RGBA"
        rgb_img = image.convert("RGB")
        w, h = rgb_img.size
        scale = descriptor.scale

        if (w * scale) * (h * scale) > MAX_OUTPUT_PIXELS:
            raise EngineError(f"Output size ({w*scale}x{h*scale}) exceeds safety limit.")

        arr = np.asarray(rgb_img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        use_fp16 = self.device_info.fp16 and descriptor.supports_half
        if use_fp16:
            tensor = tensor.half()

        tile_size = 512 if self.device_info.backend == "cuda" else 256
        overlap = 32

        out_tensor = self._tiled_forward(
            tensor, descriptor, scale, tile_size, overlap, progress, check_cancel
        )

        out_arr = (
            out_tensor.squeeze(0)
            .float()
            .clamp(0.0, 1.0)
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )
        out_uint8 = (out_arr * 255.0 + 0.5).astype(np.uint8)
        result = Image.fromarray(out_uint8, mode="RGB")

        if has_alpha:
            alpha = image.getchannel("A")
            if scale != 1:
                alpha = alpha.resize(result.size, Image.Resampling.LANCZOS)
            result.putalpha(alpha)

        self._free_vram()
        if progress:
            progress(1.0, "Complete")
        return result

    @torch.inference_mode()
    def _tiled_forward(
        self,
        x: torch.Tensor,
        model: ImageModelDescriptor,
        scale: int,
        tile: int,
        overlap: int,
        progress: Optional[ProgressFn],
        check_cancel: Callable[[], None],
    ) -> torch.Tensor:
        _, c, h, w = x.shape
        if h <= tile and w <= tile:
            if progress:
                progress(0.50, "Running AI inference...")
            return model(x)

        stride = tile - overlap
        ys = list(range(0, max(1, h - overlap), stride))
        xs = list(range(0, max(1, w - overlap), stride))
        total_tiles = len(ys) * len(xs)

        out_h, out_w = h * scale, w * scale
        output = torch.zeros((1, c, out_h, out_w), dtype=torch.float32, device="cpu")
        weights = torch.zeros((1, 1, out_h, out_w), dtype=torch.float32, device="cpu")

        done = 0
        for y in ys:
            for x0 in xs:
                check_cancel()
                y1 = min(y + tile, h)
                x1 = min(x0 + tile, w)
                y0_adj = max(0, y1 - tile)
                x0_adj = max(0, x1 - tile)

                patch = x[:, :, y0_adj:y1, x0_adj:x1]
                pred = model(patch).float().cpu()

                oy0, oy1 = y0_adj * scale, y1 * scale
                ox0, ox1 = x0_adj * scale, x1 * scale

                output[:, :, oy0:oy1, ox0:ox1] += pred
                weights[:, :, oy0:oy1, ox0:ox1] += 1.0

                done += 1
                if progress:
                    frac = 0.30 + 0.65 * (done / total_tiles)
                    progress(frac, f"Processing tile {done}/{total_tiles}...")

        return output / weights.clamp(min=1e-6)