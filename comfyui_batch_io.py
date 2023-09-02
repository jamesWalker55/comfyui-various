import glob
import json
import os
import subprocess
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


@register_node("BatchLoadImage", "[DEPRECATED] Batch Load Image")
class _:
    """
    Batch-load images in a given folder. To avoid loading too many images at once,
    you can use `paginate_size` and `paginate_page` to load a subset of the images.

    To disable pagination functionality, leave `paginate_size` and `paginate_page` at 0.
    """

    CATEGORY = "jamesWalker55"
    INPUT_TYPES = lambda: {
        "required": {
            "image_dir": ("STRING", {"default": "./images", "multiline": False}),
            "glob_pattern": ("STRING", {"default": "*.png", "multiline": False}),
            "paginate_size": ("INT", {"default": 0, "min": 0}),
            "paginate_page": ("INT", {"default": 0, "min": 0}),
        }
    }
    RETURN_NAMES = ("IMAGE", "FRAME_COUNT", "FILENAMES")
    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    FUNCTION = "execute"

    def execute(
        self, image_dir: str, glob_pattern: str, paginate_size: int, paginate_page: int
    ):
        assert isinstance(image_dir, str)
        assert isinstance(glob_pattern, str)
        assert isinstance(paginate_size, int)
        assert isinstance(paginate_page, int)

        # get paths relative to root dir
        paths = glob.glob(glob_pattern, root_dir=image_dir, recursive=True)
        # convert paths to be relative to here
        paths = [os.path.join(image_dir, x) for x in paths]
        # sort paths alphabetically
        paths.sort()

        if len(paths) == 0:
            raise FileNotFoundError(
                f"No images found in folder matching pattern {glob_pattern!r}"
            )

        if paginate_size > 0:
            start_offset = paginate_page * paginate_size
            if start_offset > len(paths):
                raise StopIteration(
                    f"No more images in folder at page {paginate_page}!"
                )
            paths = paths[start_offset : start_offset + paginate_size]

        filenames = [os.path.splitext(os.path.basename(x))[0] for x in paths]

        imgs = []
        for p in paths:
            img = load_image(p)
            # img.shape => torch.Size([1, 768, 768, 3])
            imgs.append(img)

        imgs = torch.cat(imgs, dim=0)

        assert len(imgs) == len(filenames)

        return (imgs, len(imgs), "\n".join(filenames))


@register_node("BatchSaveImage", "[DEPRECATED] Batch Save Image")
class _:
    CATEGORY = "jamesWalker55"
    INPUT_TYPES = lambda: {
        "required": {
            "images": ("IMAGE",),
            "output_dir": ("STRING", {"default": "./", "multiline": False}),
            "name_prefix": ("STRING", {"default": ""}),
            "name_suffix": ("STRING", {"default": ""}),
            "numbering_start": (
                "INT",
                {"default": 1, "min": 0, "step": 1},
            ),
            "numbering_digits": ("INT", {"default": 4, "min": 1, "step": 1}),
            "render_video_fps": ("INT", {"default": 8, "min": 0, "step": 1}),
        },
        "optional": {
            "filenames": ("STRING", {"multiline": True, "dynamicPrompts": False}),
        },
        "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
    }
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "main"

    def main(
        self,
        images: torch.Tensor,
        output_dir: str,
        name_prefix: str,
        name_suffix: str,
        numbering_start: int,
        numbering_digits: int,
        render_video_fps: int,
        filenames: str | None = None,
        prompt=None,
        extra_pnginfo=None,
    ):
        if filenames is not None:
            filenames = [x.strip() for x in filenames.splitlines()]
            filenames = [x for x in filenames if len(x) > 0]
            if len(filenames) != len(images):
                raise ValueError(
                    f"Number of images ({len(images)}) and filenames ({len(filenames)}) must be the same"
                )
            filenames = filenames.copy()
            filenames.reverse()

        output_dir: Path = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        ui_results = []

        for i, img in enumerate(images):
            num = i + numbering_start
            if filenames is not None:
                filename = filenames.pop()
                filename = f"{filename}.png"
            else:
                filename = f"{name_prefix}{num:0{numbering_digits}d}{name_suffix}.png"
            output_path = output_dir / filename
            ui = self.save_image(
                img, output_path, prompt=prompt, extra_pnginfo=extra_pnginfo
            )
            ui_results.append(ui)

        if render_video_fps > 0:
            subprocess.run(
                [
                    "python",
                    R"D:\Programming\bin\render-img-sequence.py",
                    "-i",
                    str(output_dir),
                    "-r",
                    str(render_video_fps),
                ]
            )

        return {"ui": {"images": ui_results}}

    @staticmethod
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

        subfolder, filename = os.path.split(path)

        return {"filename": filename, "subfolder": subfolder, "type": "output"}
