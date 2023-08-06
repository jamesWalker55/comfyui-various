import comfy.model_management as model_management
import torch
import torchvision.transforms.functional as F
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
from torchvision.utils import flow_to_image

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


_model = None


def load_model():
    global _model

    if _model is not None:
        return _model

    try:
        offload_device = model_management.unet_offload_device()

        _model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).eval()
        _model = _model.to(offload_device)

        return _model
    except Exception as e:
        _model = None
        raise e


@register_node("RAFTPreprocess", "RAFT Preprocess")
class _:
    """
    Preprocess images for use in RAFT. See:

    https://pytorch.org/vision/main/auto_examples/plot_optical_flow.html
    """

    CATEGORY = "jamesWalker55"

    INPUT_TYPES = lambda: {
        "required": {
            "image": ("IMAGE",),
            "width": ("INT", {"default": 512, "min": 8, "step": 8, "max": 4096}),
            "height": ("INT", {"default": 512, "min": 8, "step": 8, "max": 4096}),
        }
    }

    RETURN_NAMES = ("IMAGE",)
    RETURN_TYPES = ("IMAGE",)

    OUTPUT_NODE = False

    FUNCTION = "execute"

    def execute(self, image: torch.Tensor, width: int, height: int):
        """
        Code derived from:
        https://pytorch.org/vision/main/auto_examples/plot_optical_flow.html
        https://github.com/pytorch/vision/blob/main/torchvision/transforms/_presets.py
        """

        assert isinstance(image, torch.Tensor)
        assert isinstance(width, int)
        assert isinstance(height, int)

        # "resize them to ensure their dimensions are divisible by 8"
        # "Note that we explicitly use antialias=False, because this is how those models were trained"
        image = comfyui_to_native_torch(image)

        image = F.resize(image, size=[height, width], antialias=False)

        image = F.convert_image_dtype(image, torch.float)

        # map [0, 1] into [-1, 1]
        image = F.normalize(image, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        image = native_torch_to_comfyui(image)

        image = image.contiguous()

        print(f"{image.shape = }")
        print(f"{image.shape = }")
        print(f"{image.shape = }")
        print(f"{image.shape = }")

        return (image,)


@register_node("RAFTEstimate", "RAFT Estimate")
class _:
    """
    https://pytorch.org/vision/main/auto_examples/plot_optical_flow.html
    """

    CATEGORY = "jamesWalker55"

    INPUT_TYPES = lambda: {
        "required": {
            "image_a": ("IMAGE",),
            "image_b": ("IMAGE",),
        }
    }

    RETURN_NAMES = ("RAFT_FLOW",)
    RETURN_TYPES = ("RAFT_FLOW",)

    OUTPUT_NODE = False

    FUNCTION = "execute"

    def execute(self, image_a: torch.Tensor, image_b: torch.Tensor):
        """
        Code derived from:
        https://pytorch.org/vision/main/auto_examples/plot_optical_flow.html
        """

        assert isinstance(image_a, torch.Tensor)
        assert isinstance(image_b, torch.Tensor)

        image_a = comfyui_to_native_torch(image_a)
        image_b = comfyui_to_native_torch(image_b)

        model = load_model()

        all_flows = model(image_a, image_b)
        best_flow = all_flows[-1]
        # best_flow.shape => torch.Size([1, 2, 512, 512])

        return (best_flow,)


@register_node("RAFTFlowToImage", "RAFT Flow to Image")
class _:
    """
    https://pytorch.org/vision/main/auto_examples/plot_optical_flow.html
    """

    CATEGORY = "jamesWalker55"

    INPUT_TYPES = lambda: {
        "required": {
            "raft_flow": ("RAFT_FLOW",),
        }
    }

    RETURN_NAMES = ("IMAGE",)
    RETURN_TYPES = ("IMAGE",)

    OUTPUT_NODE = False

    FUNCTION = "execute"

    def execute(self, flows: torch.Tensor):
        assert isinstance(flows, torch.Tensor)
        assert flows.shape[1] == 2

        images = flow_to_image(flows)
        # pixel range is [0, 255], dtype=torch.uint8

        images = images / 255

        images = native_torch_to_comfyui(images)
        print(f"{images.shape = }")
        print(f"{images.shape = }")
        print(f"{images.shape = }")
        print(f"{images.shape = }")
        print(f"{images.min() = }")
        print(f"{images.min() = }")
        print(f"{images.max() = }")
        print(f"{images.max() = }")

        return (images,)
