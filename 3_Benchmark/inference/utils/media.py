from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Tuple

import torch

try:
    from decord import VideoReader, cpu as decord_cpu  # type: ignore

    HAS_DECORD = True
except Exception:
    HAS_DECORD = False

try:
    import imageio.v3 as iio  # type: ignore

    HAS_IMAGEIO = True
except Exception:
    HAS_IMAGEIO = False

try:
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    HAS_TORCHVISION = True
except Exception:
    HAS_TORCHVISION = False

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def require_video_support() -> None:
    missing = []
    if not HAS_TORCHVISION:
        missing.append("torchvision")
    if not (HAS_DECORD or HAS_IMAGEIO):
        missing.append("decord or imageio[v3]")
    if missing:
        hint = " and ".join(missing)
        raise ImportError(
            f"Video evaluation requires {hint}. Install the missing dependencies before running video-capable tasks."
        )


def _require_torchvision() -> None:
    if not HAS_TORCHVISION:
        raise ImportError(
            "torchvision is required for preprocessing. Please install a version compatible with your torch build."
        )


def build_transform(input_size: int):
    _require_torchvision()
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if getattr(img, "mode", "RGB") != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: List[Tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> Tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    from PIL import Image

    orig_width, orig_height = image.size
    aspect_ratio = orig_width / max(1, orig_height)
    target_ratios = sorted(
        {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        },
        key=lambda x: x[0] * x[1],
    )
    target_aspect = _find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width, target_height = image_size * target_aspect[0], image_size * target_aspect[1]
    resized_img = image.resize((target_width, target_height), Image.BICUBIC)
    processed = []
    cols = max(1, target_width // image_size)
    rows = max(1, target_height // image_size)
    for r in range(rows):
        for c in range(cols):
            box = (c * image_size, r * image_size, (c + 1) * image_size, (r + 1) * image_size)
            processed.append(resized_img.crop(box))
    if use_thumbnail and len(processed) != 1:
        processed.append(image.resize((image_size, image_size), Image.BICUBIC))
    return processed


def _sample_indices(total: int, max_frames: int) -> List[int]:
    if total <= 0:
        return []
    if total <= max_frames:
        return list(range(total))
    step = max(1, math.floor(total / max_frames))
    return list(range(0, total, step))[:max_frames]


def load_image_to_pixel_values(image_file: Path, input_size=448, max_num=12) -> torch.Tensor:
    from PIL import Image

    _require_torchvision()
    transform = build_transform(input_size=input_size)
    img = Image.open(image_file).convert("RGB")
    tiles = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
    px = [transform(t) for t in tiles]
    return torch.stack(px)


def load_video_to_pixel_values(
    video_path: Path,
    input_size=448,
    max_num=1,
    num_segments=8,
    max_frames: Optional[int] = None,
) -> Tuple[torch.Tensor, List[int], List[int]]:
    from PIL import Image

    _require_torchvision()
    transform = build_transform(input_size=input_size)
    num_patches_list: List[int] = []
    px_list: List[torch.Tensor] = []
    used_indices: List[int] = []

    frames_np = []
    if HAS_DECORD:
        vr = VideoReader(str(video_path), ctx=decord_cpu(0), num_threads=1)
        total = len(vr)
        idxs = _sample_indices(total, max_frames or num_segments)
        for i in idxs:
            frames_np.append(vr[i].asnumpy())
        used_indices = idxs
    elif HAS_IMAGEIO:
        meta = {}
        try:
            meta = iio.immeta(str(video_path))
        except Exception:
            pass
        nframes = meta.get("nframes", None)
        if isinstance(nframes, int) and nframes > 0:
            idxs = _sample_indices(nframes, max_frames or num_segments)
        else:
            idxs = list(range(num_segments))
        for i in idxs:
            try:
                frames_np.append(iio.imread(str(video_path), index=i))
                used_indices.append(i)
            except Exception:
                break
    else:
        raise ImportError("Need decord or imageio for video decoding. Install one of them to enable video support.")

    for arr in frames_np:
        img = Image.fromarray(arr).convert("RGB")
        tiles = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pv = torch.stack([transform(t) for t in tiles])
        num_patches_list.append(pv.shape[0])
        px_list.append(pv)

    if not px_list:
        raise ValueError(f"Failed to decode frames from {video_path}")

    pixel_values = torch.cat(px_list, dim=0)
    return pixel_values, num_patches_list, used_indices


def load_video_frames_raw(
    video_path: Path,
    num_segments: int = 8,
    max_frames: Optional[int] = None,
):
    from PIL import Image as PILImage

    frames: List[PILImage.Image] = []

    if HAS_DECORD:
        vr = VideoReader(str(video_path), ctx=decord_cpu(0), num_threads=1)
        total = len(vr)
        idxs = _sample_indices(total, max_frames or num_segments)
        for i in idxs:
            frames.append(PILImage.fromarray(vr[i].asnumpy()).convert("RGB"))
    elif HAS_IMAGEIO:
        meta = {}
        try:
            meta = iio.immeta(str(video_path))
        except Exception:
            pass
        nframes = meta.get("nframes", None)
        if isinstance(nframes, int) and nframes > 0:
            idxs = _sample_indices(nframes, max_frames or num_segments)
        else:
            idxs = list(range(num_segments))
        for i in idxs:
            try:
                frames.append(PILImage.fromarray(iio.imread(str(video_path), index=i)).convert("RGB"))
            except Exception:
                break
    else:
        raise ImportError("Need decord or imageio for video decoding. Install one of them to enable video support.")

    if not frames:
        raise ValueError(f"Failed to decode frames from {video_path}")
    return frames
