import torch
import torchvision.transforms.functional as F

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def register_node(identifier: str, display_name: str):
    def decorator(cls):
        NODE_CLASS_MAPPINGS[identifier] = cls
        NODE_DISPLAY_NAME_MAPPINGS[identifier] = display_name

        return cls

    return decorator


@register_node("JWImageMix", "Image Mix")
class _:
    CATEGORY = "jamesWalker55"
    BLEND_TYPES = ("mix", "multiply")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "blend_type": (cls.BLEND_TYPES, {"default": "mix"}),
                "factor": ("FLOAT", {"min": 0, "max": 1, "step": 0.01, "default": 0.5}),
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    def execute(
        self,
        blend_type: str,
        factor: float,
        image_a: torch.Tensor,
        image_b: torch.Tensor,
    ):
        assert blend_type in self.BLEND_TYPES
        assert isinstance(factor, float)
        assert isinstance(image_a, torch.Tensor)
        assert isinstance(image_b, torch.Tensor)

        assert image_a.shape == image_b.shape

        if blend_type == "mix":
            mixed = image_a * (1 - factor) + image_b * factor
        elif blend_type == "multiply":
            mixed = image_a * (1 - factor + image_b * factor)
        else:
            raise NotImplementedError(f"Blend type not yet implemented: {blend_type}")

        return (mixed,)


@register_node("JWImageContrast", "Image Contrast")
class _:
    CATEGORY = "jamesWalker55"
    INPUT_TYPES = lambda: {
        "required": {
            "image": ("IMAGE",),
            "factor": (
                "FLOAT",
                {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01},
            ),
        }
    }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    def execute(
        self,
        image: torch.Tensor,
        factor: float,
    ):
        assert isinstance(image, torch.Tensor)
        assert isinstance(factor, float)

        image = image.permute(0, 3, 1, 2)
        image = F.adjust_contrast(image, factor)
        image = image.permute(0, 2, 3, 1)

        return (image,)


@register_node("JWImageSaturation", "Image Saturation")
class _:
    CATEGORY = "jamesWalker55"
    INPUT_TYPES = lambda: {
        "required": {
            "image": ("IMAGE",),
            "factor": (
                "FLOAT",
                {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01},
            ),
        }
    }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    def execute(
        self,
        image: torch.Tensor,
        factor: float,
    ):
        assert isinstance(image, torch.Tensor)
        assert isinstance(factor, float)

        image = image.permute(0, 3, 1, 2)
        image = F.adjust_saturation(image, factor)
        image = image.permute(0, 2, 3, 1)

        return (image,)


@register_node("JWImageLevels", "Image Levels")
class _:
    CATEGORY = "jamesWalker55"
    INPUT_TYPES = lambda: {
        "required": {
            "image": ("IMAGE",),
            "min": (
                "FLOAT",
                {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
            ),
            "max": (
                "FLOAT",
                {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
            ),
        }
    }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    def execute(
        self,
        image: torch.Tensor,
        min: float,
        max: float,
    ):
        assert isinstance(image, torch.Tensor)
        assert isinstance(min, float)
        assert isinstance(max, float)

        image = (image - min) / (max - min)
        image = torch.clamp(image, 0.0, 1.0)

        return (image,)
