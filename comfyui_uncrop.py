from typing import NamedTuple

import torch
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def register_node(identifier: str, display_name: str):
    def decorator(cls):
        NODE_CLASS_MAPPINGS[identifier] = cls
        NODE_DISPLAY_NAME_MAPPINGS[identifier] = display_name

        return cls

    return decorator


MAX_RESOLUTION = 8192


def validate_bounds(img: torch.Tensor, x: int, y: int, w: int, h: int):
    _, img_h, img_w, _ = img.shape

    assert x >= 0
    assert y >= 0

    assert (
        x + w <= img_w
    ), f"crop region out of bounds: crop {(x, y, w, h)} from image {(img_w, img_h)}"
    assert (
        y + h <= img_h
    ), f"crop region out of bounds: crop {(x, y, w, h)} from image {(img_w, img_h)}"


def crop_image(img: torch.Tensor, x: int, y: int, w: int, h: int):
    validate_bounds(img, x, y, w, h)

    to_x = x + w
    to_y = y + h
    return img[:, y:to_y, x:to_x, :]


def resize_image(img: torch.Tensor, w: int, h: int):
    img = img.permute(0, 3, 1, 2)
    img = F.resize(
        img,
        (h, w),  # type: ignore
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    img = img.permute(0, 2, 3, 1)
    return img


class CropRect(NamedTuple):
    x: int
    y: int
    width: int
    height: int


@register_node("JWUncropNewRect", "Uncrop: New rect")
class _:
    CATEGORY = "jamesWalker55"
    INPUT_TYPES = lambda: {
        "required": {
            "x": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
            "y": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
            "width": (
                "INT",
                {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1},
            ),
            "height": (
                "INT",
                {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1},
            ),
        }
    }
    RETURN_TYPES = ("CROP_RECT",)
    FUNCTION = "execute"

    def execute(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> tuple[CropRect]:
        return (CropRect(x, y, width, height),)


@register_node("JWUncropCrop", "Uncrop: Crop")
class _:
    CATEGORY = "jamesWalker55"
    INPUT_TYPES = lambda: {
        "required": {
            "image": ("IMAGE",),
            "resize_length": ("INT", {"default": 512, "min": 8, "step": 8}),
            "crop_rect": ("CROP_RECT",),
        }
    }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    def execute(
        self,
        image: torch.Tensor,
        resize_length: int,
        crop_rect: CropRect,
    ) -> tuple[torch.Tensor]:
        x, y, width, height = crop_rect

        # crop the image
        image = crop_image(image, x, y, width, height)

        shortest_side = min(width, height)
        scale_ratio = resize_length / shortest_side
        new_width = round(round(width * scale_ratio / 8) * 8)
        new_height = round(round(height * scale_ratio / 8) * 8)

        image = resize_image(image, new_width, new_height)

        return (image,)


@register_node("JWUncropUncrop", "Uncrop: Uncrop")
class _:
    CATEGORY = "jamesWalker55"
    INPUT_TYPES = lambda: {
        "required": {
            "original_image": ("IMAGE",),
            "cropped_image": ("IMAGE",),
            "cropped_mask": ("MASK",),
            "crop_rect": ("CROP_RECT",),
        }
    }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    def execute(
        self,
        original_image: torch.Tensor,
        cropped_image: torch.Tensor,
        cropped_mask: torch.Tensor,
        crop_rect: CropRect,
    ) -> tuple[torch.Tensor]:
        x, y, width, height = crop_rect

        validate_bounds(original_image, x, y, width, height)

        # resize cropped image if needed
        _, _h, _w, _ = cropped_image.shape
        if _w != width or _h != height:
            cropped_image = resize_image(cropped_image, width, height)

        # resize cropped mask if needed
        _h, _w = cropped_mask.shape[-2:]
        if _w != width or _h != height:
            cropped_mask = torch.reshape(cropped_mask, (1, _h, _w, 1))
            cropped_mask = resize_image(cropped_mask, width, height)
            cropped_mask = torch.reshape(cropped_mask, (height, width))

        to_x = x + width
        to_y = y + height

        # https://easings.net/#easeOutQuint
        weighted_mask = 1 - (1 - cropped_mask) ** 5

        # blend original image with cropped image using mask
        cropped_image = original_image[:, y:to_y, x:to_x, :] * (
            1 - weighted_mask.view(1, *weighted_mask.shape, 1)
        ) + cropped_image * weighted_mask.view(1, *weighted_mask.shape, 1)

        # paste cropped image into original image
        original_image = original_image.clone()
        original_image[:, y:to_y, x:to_x, :] = cropped_image

        return (original_image,)
