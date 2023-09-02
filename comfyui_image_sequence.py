import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def register_node(identifier: str, display_name: str):
    def decorator(cls):
        NODE_CLASS_MAPPINGS[identifier] = cls
        NODE_DISPLAY_NAME_MAPPINGS[identifier] = display_name

        return cls

    return decorator


def load_image(path):
    img = Image.open(path).convert("RGB")
    img = np.array(img).astype(np.float32) / 255.0
    img = torch.from_numpy(img).unsqueeze(0)
    return img


@register_node("JWLoadImageSequence", "Batch Load Image Sequence")
class _:
    CATEGORY = "jamesWalker55"
    INPUT_TYPES = lambda: {
        "required": {
            "path_pattern": (
                "STRING",
                {"default": "./frame{:06d}.png", "multiline": False},
            ),
            "start_index": ("INT", {"default": 0, "min": 0, "step": 1}),
            "frame_count": ("INT", {"default": 16, "min": 1, "step": 1}),
            "ignore_missing_images": (("false", "true"), {"default": "false"}),
        }
    }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    def execute(
        self,
        path_pattern: str,
        start_index: int,
        frame_count: int,
        ignore_missing_images: str,
    ):
        ignore_missing_images: bool = ignore_missing_images == "true"

        # generate image paths to load
        image_paths = []
        for i in range(start_index, start_index + frame_count):
            try:
                image_paths.append(path_pattern.format(i))
            except KeyError:
                image_paths.append(path_pattern.format(i=i))

        if ignore_missing_images:
            # remove missing images
            image_paths = [p for p in image_paths if os.path.exists(p)]
        else:
            # early check for missing images
            for path in image_paths:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Image does not exist: {path}")

        if len(image_paths) == 0:
            raise RuntimeError("Image sequence empty - no images to load")

        imgs = []
        for path in image_paths:
            img = load_image(path)
            # img.shape => torch.Size([1, 768, 768, 3])
            imgs.append(img)

        imgs = torch.cat(imgs, dim=0)

        return (imgs,)


@register_node(
    "JWLoadImageSequenceWithStopIndex",
    "Batch Load Image Sequence With Stop Index",
)
class _:
    CATEGORY = "jamesWalker55"
    INPUT_TYPES = lambda: {
        "required": {
            "path_pattern": (
                "STRING",
                {"default": "./frame{:06d}.png", "multiline": False},
            ),
            "start_index": ("INT", {"default": 0, "min": 0, "step": 1, "max": 999999}),
            "stop_index": ("INT", {"default": 16, "min": 1, "step": 1, "max": 999999}),
            "inclusive": (("false", "true"), {"default": "false"}),
            "ignore_missing_images": (("false", "true"), {"default": "false"}),
        }
    }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    def execute(
        self,
        path_pattern: str,
        start_index: int,
        stop_index: int,
        inclusive: str,
        ignore_missing_images: str,
    ):
        inclusive: bool = inclusive == "true"
        ignore_missing_images: bool = ignore_missing_images == "true"

        # generate image paths to load
        image_paths = []
        for i in range(start_index, stop_index + 1 if inclusive else stop_index):
            try:
                image_paths.append(path_pattern.format(i))
            except KeyError:
                image_paths.append(path_pattern.format(i=i))

        if ignore_missing_images:
            # remove missing images
            image_paths = [p for p in image_paths if os.path.exists(p)]
        else:
            # early check for missing images
            for path in image_paths:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Image does not exist: {path}")

        if len(image_paths) == 0:
            raise RuntimeError("Image sequence empty - no images to load")

        imgs = []
        for path in image_paths:
            img = load_image(path)
            # img.shape => torch.Size([1, 768, 768, 3])
            imgs.append(img)

        imgs = torch.cat(imgs, dim=0)

        return (imgs,)


def generate_non_conflicting_path(path: Path):
    if not path.exists():
        return path

    i = -1
    while True:
        i += 1
        new_path = path.with_stem(f"{path.stem}-{i}")
        if new_path.exists():
            continue

        return new_path


def save_image(img: torch.Tensor, path, prompt=None, extra_pnginfo: dict = None):
    path = str(path)

    img = 255.0 * img.cpu().numpy()
    img = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))

    metadata = PngInfo()

    if prompt is not None:
        metadata.add_text("prompt", json.dumps(prompt))

    if extra_pnginfo is not None:
        for k, v in extra_pnginfo.items():
            metadata.add_text(k, json.dumps(v))

    img.save(path, pnginfo=metadata, compress_level=4)


@register_node("JWImageSequenceExtractFromBatch", "Extract Image Sequence From Batch")
class _:
    CATEGORY = "jamesWalker55"
    INPUT_TYPES = lambda: {
        "required": {
            "images": ("IMAGE",),
            "i_start": ("INT", {"default": 0, "min": 0, "step": 1}),
            "i_stop": ("INT", {"default": 0, "min": 0, "step": 1}),
            "inclusive": (("false", "true"), {"default": "false"}),
        }
    }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    def execute(self, images: torch.Tensor, i_start: int, i_stop: int, inclusive: str):
        assert isinstance(images, torch.Tensor)
        assert isinstance(i_start, int)
        assert isinstance(i_stop, int)
        assert isinstance(inclusive, str)

        inclusive: bool = inclusive == "true"

        img = images[i_start : i_stop + 1 if inclusive else i_stop]

        return (img,)


@register_node("JWSaveImageSequence", "Batch Save Image Sequence")
class _:
    CATEGORY = "jamesWalker55"
    INPUT_TYPES = lambda: {
        "required": {
            "images": ("IMAGE",),
            "path_pattern": (
                "STRING",
                {"default": "./frame{:06d}.png", "multiline": False},
            ),
            "start_index": ("INT", {"default": 0, "min": 0, "step": 1}),
            "overwrite": (("false", "true"), {"default": "true"}),
        },
        "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
    }
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "execute"

    def execute(
        self,
        images: torch.Tensor,
        path_pattern: str,
        start_index: int,
        overwrite: str,
        prompt=None,
        extra_pnginfo=None,
    ):
        overwrite: bool = overwrite == "true"

        image_range = range(start_index, start_index + len(images))

        for i, img in zip(image_range, images):
            try:
                path = Path(path_pattern.format(i))
            except KeyError:
                path = Path(path_pattern.format(i=i))

            # Create containing folder for output path
            path.parent.mkdir(exist_ok=True)

            if not overwrite and path.exists():
                print("JWSaveImageSequence: [WARNING]")
                print(f"JWSaveImageSequence: Image already exists: {path}")
                path = generate_non_conflicting_path(path)
                print(f"JWSaveImageSequence: Saving to new path instead: {path}")

            save_image(
                img,
                path,
                prompt=prompt,
                extra_pnginfo=extra_pnginfo,
            )

        return ()


@register_node("JWLoopImageSequence", "Loop Image Sequence")
class LoopImageSequence:
    CATEGORY = "jamesWalker55"
    INPUT_TYPES = lambda: {
        "required": {
            "images": ("IMAGE",),
            "target_frames": ("INT", {"default": 16, "step": 1}),
        }
    }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    def execute(self, images: torch.Tensor, target_frames: int):
        if len(images) > target_frames:
            images = images[0:target_frames]
        elif len(images) < target_frames:
            to_cat = []

            for _ in range(target_frames // len(images)):
                to_cat.append(images)

            extra_frames = target_frames % len(images)
            if extra_frames > 0:
                to_cat.append(images[0:extra_frames])

            images = torch.cat(to_cat, dim=0)
            assert len(images) == target_frames
        else:  # len(images) == target_frames
            pass

        return (images,)
