import comfy.model_management as model_management
import numpy as np
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


def preprocess_image(img: torch.Tensor):
    # Image size must be divisible by 8
    _, _, h, w = img.shape
    assert h % 8 == 0, "Image height must be divisible by 8"
    assert w % 8 == 0, "Image width must be divisible by 8"

    img = F.convert_image_dtype(img, torch.float)

    # map [0, 1] into [-1, 1]
    img = F.normalize(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    img = img.contiguous()

    return img


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
    RETURN_TYPES = ("RAFT_FLOW",)
    FUNCTION = "execute"

    def execute(self, image_a: torch.Tensor, image_b: torch.Tensor):
        """
        Code derived from:
        https://pytorch.org/vision/main/auto_examples/plot_optical_flow.html
        """

        assert isinstance(image_a, torch.Tensor)
        assert isinstance(image_b, torch.Tensor)

        torch_device = model_management.get_torch_device()
        offload_device = model_management.unet_offload_device()

        image_a = comfyui_to_native_torch(image_a).to(torch_device)
        image_b = comfyui_to_native_torch(image_b).to(torch_device)
        model = load_model().to(torch_device)

        image_a = preprocess_image(image_a)
        image_b = preprocess_image(image_b)

        all_flows = model(image_a, image_b)
        best_flow = all_flows[-1]
        # best_flow.shape => torch.Size([1, 2, 512, 512])

        model.to(offload_device)
        image_a = image_a.to("cpu")
        image_b = image_b.to("cpu")
        best_flow = best_flow.to("cpu")

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
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    def execute(self, raft_flow: torch.Tensor):
        assert isinstance(raft_flow, torch.Tensor)
        assert raft_flow.shape[1] == 2

        images = flow_to_image(raft_flow)
        # pixel range is [0, 255], dtype=torch.uint8

        images = images / 255

        images = native_torch_to_comfyui(images)

        return (images,)


def depth_exr_to_numpy(exr_path, typemap={"HALF": np.float16, "FLOAT": np.float32}):
    # Code stolen from:
    # https://gist.github.com/andres-fr/4ddbb300d418ed65951ce88766236f9c

    import OpenEXR

    # load EXR and extract shape
    exr = OpenEXR.InputFile(exr_path)
    print(exr.header())
    dw = exr.header()["dataWindow"]
    shape = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
    #
    arr_maps = {}
    for ch_name, ch in exr.header()["channels"].items():
        print("reading channel", ch_name)
        # This, and __str__ seem to be the only ways to get typename
        exr_typename = ch.type.names[ch.type.v]
        np_type = typemap[exr_typename]
        # convert channel to np array
        bytestring = exr.channel(ch_name, ch.type)
        arr = np.frombuffer(bytestring, dtype=np_type).reshape(shape)
        arr_maps[ch_name] = arr

    return arr_maps


@register_node("RAFTLoadFlowFromEXRChannels", "RAFT Load Flow from EXR Channels")
class _:
    """
    This is a utility function for loading motion flows from an EXR image file.
    This is intended for use with Blender's vector pass in the Cycles renderer.

    In Blender, enable the vector pass. In the compositor, use "Separate Color" to
    extract the "Blue" and "Alpha" channels of the vector pass. Then, combine them
    using "Combine Color" to two of the RGB channels. Finally, render to the "OpenEXR"
    format.

    https://gist.github.com/andres-fr/4ddbb300d418ed65951ce88766236f9c
    """

    CATEGORY = "jamesWalker55"
    INPUT_TYPES = lambda: {
        "required": {
            "path": ("STRING", {"default": ""}),
            "x_channel": (("R", "G", "B", "A"), {"default": "R"}),
            "y_channel": (("R", "G", "B", "A"), {"default": "G"}),
            "invert_x": (("false", "true"), {"default": "true"}),
            "invert_y": (("false", "true"), {"default": "false"}),
        }
    }
    RETURN_TYPES = ("RAFT_FLOW",)
    FUNCTION = "execute"

    def execute(
        self, path: str, x_channel: str, y_channel: str, invert_x: str, invert_y: str
    ):
        assert isinstance(path, str)
        assert x_channel in ("R", "G", "B", "A")
        assert y_channel in ("R", "G", "B", "A")
        assert invert_x in ("true", "false")
        assert invert_y in ("true", "false")

        invert_x: bool = invert_x == "true"
        invert_y: bool = invert_y == "true"

        maps = depth_exr_to_numpy(path)

        x = torch.from_numpy(maps[x_channel])
        y = torch.from_numpy(maps[y_channel])

        if invert_x:
            x = x * -1

        if invert_y:
            y = y * -1

        return (torch.stack((x, y)).unsqueeze(0),)
