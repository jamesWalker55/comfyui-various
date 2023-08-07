import kornia as K
import torch

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def register_node(identifier: str, display_name: str):
    def decorator(cls):
        NODE_CLASS_MAPPINGS[identifier] = cls
        NODE_DISPLAY_NAME_MAPPINGS[identifier] = display_name

        return cls

    return decorator


def comfyui_to_native_torch(imgs: torch.Tensor):
    """
    Convert images in NHWC format to NCHW format.

    Use this to convert ComfyUI images to torch-native images.
    """
    return imgs.permute(0, 3, 1, 2)


def native_torch_to_comfyui(imgs: torch.Tensor):
    """
    Convert images in NCHW format to NHWC format.

    Use this to convert torch-native images to ComfyUI images.
    """
    return imgs.permute(0, 2, 3, 1)


@register_node("JWKorniaCannyEdge", "Kornia Canny Edge")
class _:
    CATEGORY = "jamesWalker55"

    INPUT_TYPES = lambda: {
        "required": {
            "image": ("IMAGE",),
            "low_threshold": (
                "FLOAT",
                {"min": 0, "max": 1, "default": 0.4, "step": 0.01},
            ),
            "high_threshold": (
                "FLOAT",
                {"min": 0, "max": 1, "default": 0.8, "step": 0.01},
            ),
        }
    }

    RETURN_NAMES = ("IMAGE",)
    RETURN_TYPES = ("IMAGE",)

    OUTPUT_NODE = False

    FUNCTION = "execute"

    def execute(self, image: torch.Tensor, low_threshold: float, high_threshold: float):
        assert isinstance(image, torch.Tensor)
        assert isinstance(low_threshold, float)
        assert isinstance(high_threshold, float)

        image = comfyui_to_native_torch(image)

        image = K.filters.canny(
            image,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
        )[1]
        image = image.clamp(0.0, 1.0)
        # returns a 1-channel image
        # create a fake RGB image from that
        image = torch.cat((image, image, image), dim=1)

        image = native_torch_to_comfyui(image)

        return (image,)
