import base64
import json
import lzma
from io import BytesIO

import torch

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def register_node(identifier: str, display_name: str):
    def decorator(cls):
        NODE_CLASS_MAPPINGS[identifier] = cls
        NODE_DISPLAY_NAME_MAPPINGS[identifier] = display_name

        return cls

    return decorator


def compress(x: bytes):
    comp = lzma.LZMACompressor()
    out = comp.compress(x)
    return out + comp.flush()


def decompress(x: bytes):
    decomp = lzma.LZMADecompressor()
    return decomp.decompress(x)


def base85_encode(x: bytes):
    return base64.b85encode(x)


def base85_decode(x: bytes):
    return base64.b85decode(x)


def torch_save_to_bytes(obj):
    with BytesIO() as f:
        torch.save(obj, f)
        return f.getvalue()


def torch_load_from_bytes(text: bytes):
    with BytesIO(text) as f:
        return torch.load(f)


def torch_save_to_blob(obj):
    return base85_encode(compress(torch_save_to_bytes(obj)))


def torch_load_from_blob(text: bytes):
    return torch_load_from_bytes(decompress(base85_decode(text)))


@register_node("RCReceiveLatent", "Remote Call: Receive Latent")
class _:
    CATEGORY = "jamesWalker55/rc"
    INPUT_TYPES = lambda: {
        "required": {
            "key": (
                "STRING",
                {"default": "input_latent", "multiline": False},
            ),
            "value": (
                "STRING",
                {"default": "Don't touch this field!", "multiline": False},
            ),
        }
    }
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "execute"

    def execute(self, key: str, value: str):
        latent = torch_load_from_blob(value)
        val = {"samples": latent}
        # { "samples": <Latent: [1, 4, 64, 64]> }
        return (val,)


@register_node("RCReceiveInt", "Remote Call: Receive Integer")
class _:
    CATEGORY = "jamesWalker55/rc"
    INPUT_TYPES = lambda: {
        "required": {
            "key": (
                "STRING",
                {"default": "input_integer", "multiline": False},
            ),
            "value": ("INT", {"default": 0, "min": -99999999999, "max": 99999999999}),
        }
    }
    RETURN_TYPES = ("INT",)
    FUNCTION = "execute"

    def execute(self, key: str, value):
        return (value,)


@register_node("RCReceiveFloat", "Remote Call: Receive Float")
class _:
    CATEGORY = "jamesWalker55/rc"
    INPUT_TYPES = lambda: {
        "required": {
            "key": (
                "STRING",
                {"default": "input_float", "multiline": False},
            ),
            "value": ("FLOAT", {"default": 0, "min": -99999999999, "max": 99999999999}),
        }
    }
    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "execute"

    def execute(self, key: str, value):
        return (value,)


@register_node("RCReceiveIntList", "Remote Call: Receive Integer List")
class _:
    CATEGORY = "jamesWalker55/rc"
    INPUT_TYPES = lambda: {
        "required": {
            "key": (
                "STRING",
                {"default": "input_integer_list", "multiline": False},
            ),
            "value": (
                "STRING",
                {"default": "[1, 2, 3]", "multiline": False},
            ),
        }
    }
    RETURN_TYPES = ("INT_LIST",)
    FUNCTION = "execute"

    def execute(self, key: str, value):
        value = json.loads(value)
        return (value,)


@register_node("RCReceiveFloatList", "Remote Call: Receive Float List")
class _:
    CATEGORY = "jamesWalker55/rc"
    INPUT_TYPES = lambda: {
        "required": {
            "key": (
                "STRING",
                {"default": "input_float_list", "multiline": False},
            ),
            "value": (
                "STRING",
                {"default": "[1.0, 2.0, 3.0]", "multiline": False},
            ),
        }
    }
    RETURN_TYPES = ("FLOAT_LIST",)
    FUNCTION = "execute"

    def execute(self, key: str, value):
        value = json.loads(value)
        return (value,)


@register_node("RCSendLatent", "Remote Call: Send Latent")
class _:
    CATEGORY = "jamesWalker55/rc"
    INPUT_TYPES = lambda: {
        "required": {
            "key": (
                "STRING",
                {"default": "input_latent", "multiline": False},
            ),
            "latent": ("LATENT",),
        }
    }
    FUNCTION = "execute"
    RETURN_TYPES = ()
    OUTPUT_NODE = True

    def execute(self, key: str, latent: str):
        blob = torch_save_to_blob(latent["samples"])

        return {
            "ui": {
                "jw_rc": (
                    {
                        "type": "latent",
                        "value": blob.decode(),
                    },
                ),
            }
        }
