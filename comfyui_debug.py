import textwrap
from pprint import pformat, pprint

import torch

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def register_node(identifier: str, display_name: str):
    def decorator(cls):
        NODE_CLASS_MAPPINGS[identifier] = cls
        NODE_DISPLAY_NAME_MAPPINGS[identifier] = display_name

        return cls

    return decorator


@register_node("JWPrintInteger", "Print Integer")
class _:
    CATEGORY = "jamesWalker55"
    INPUT_TYPES = lambda: {
        "required": {
            "value": ("INT", {"default": 0, "min": -99999999999, "max": 99999999999}),
            "name": (
                "STRING",
                {"default": "integer", "multiline": True, "dynamicPrompts": False},
            ),
        }
    }
    RETURN_TYPES = ("INT",)
    OUTPUT_NODE = True
    FUNCTION = "execute"

    def execute(self, value, name: str):
        print(f"{name} = {pformat(value)}")

        return (value,)

    @classmethod
    def IS_CHANGED(cls, *args):
        # Always recalculate
        return float("NaN")


@register_node("JWPrintFloat", "Print Float")
class _:
    CATEGORY = "jamesWalker55"
    INPUT_TYPES = lambda: {
        "required": {
            "value": ("FLOAT", {"default": 0, "min": -99999999999, "max": 99999999999}),
            "name": (
                "STRING",
                {"default": "float", "multiline": True, "dynamicPrompts": False},
            ),
        }
    }
    RETURN_TYPES = ("FLOAT",)
    OUTPUT_NODE = True
    FUNCTION = "execute"

    def execute(self, value, name: str):
        print(f"{name} = {pformat(value)}")

        return (value,)

    @classmethod
    def IS_CHANGED(cls, *args):
        # Always recalculate
        return float("NaN")


@register_node("JWPrintString", "Print String")
class _:
    CATEGORY = "jamesWalker55"
    INPUT_TYPES = lambda: {
        "required": {
            "value": ("STRING", {"default": "text", "multiline": False}),
            "name": (
                "STRING",
                {"default": "string", "multiline": True, "dynamicPrompts": False},
            ),
        }
    }
    RETURN_TYPES = ("STRING",)
    OUTPUT_NODE = True
    FUNCTION = "execute"

    def execute(self, value, name: str):
        print(f"{name} = {pformat(value)}")

        return (value,)

    @classmethod
    def IS_CHANGED(cls, *args):
        # Always recalculate
        return float("NaN")


@register_node("JWPrintImage", "Print Image")
class _:
    CATEGORY = "jamesWalker55"
    INPUT_TYPES = lambda: {
        "required": {
            "value": ("IMAGE",),
            "name": (
                "STRING",
                {"default": "image", "multiline": True, "dynamicPrompts": False},
            ),
        }
    }
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_NODE = True
    FUNCTION = "execute"

    def execute(self, value: torch.Tensor, name: str):
        lines = [
            f"{name} =",
            f"  {name}.shape = {value.shape}",
            f"  {name}.min() = {value.min()}",
            f"  {name}.max() = {value.max()}",
            f"  {name}.mean() = {value.mean()}",
            f"  {name}.std() = {value.std()}",
            f"  {name}.dtype = {value.dtype}",
        ]
        lines = "\n".join(lines)
        print(lines)

        return (value,)

    @classmethod
    def IS_CHANGED(cls, *args):
        # Always recalculate
        return float("NaN")


@register_node("JWPrintMask", "Print Mask")
class _:
    CATEGORY = "jamesWalker55"
    INPUT_TYPES = lambda: {
        "required": {
            "value": ("MASK",),
            "name": (
                "STRING",
                {"default": "mask", "multiline": True, "dynamicPrompts": False},
            ),
        }
    }
    RETURN_TYPES = ("MASK",)
    OUTPUT_NODE = True
    FUNCTION = "execute"

    def execute(self, value: torch.Tensor, name: str):
        lines = [
            f"{name} =",
            f"  {name}.shape = {value.shape}",
            f"  {name}.min() = {value.min()}",
            f"  {name}.max() = {value.max()}",
            f"  {name}.mean() = {value.mean()}",
            f"  {name}.std() = {value.std()}",
            f"  {name}.dtype = {value.dtype}",
        ]
        lines = "\n".join(lines)
        print(lines)

        return (value,)

    @classmethod
    def IS_CHANGED(cls, *args):
        # Always recalculate
        return float("NaN")


def serialise_obj(obj):
    if isinstance(obj, dict):
        text = ["{"]
        for k, v in obj.items():
            subtext = [
                textwrap.indent(f"{k!r}:", "  "),
                textwrap.indent(serialise_obj(v), "    "),
            ]
            text.append("\n".join(subtext))
        text.append("}")
        text = "\n".join(text)
    elif isinstance(obj, list):
        text = []
        for x in obj:
            subtext = serialise_obj(x)
            subtext = textwrap.indent(subtext, "  ")
            subtext = f"-{subtext[1:]}"
            text.append(subtext)
        text = "\n".join(text)
    elif isinstance(obj, torch.Tensor):
        text = "\n".join(
            [
                f"Tensor",
                f"  .shape = {obj.shape}",
                f"  .min() = {obj.min()}",
                f"  .max() = {obj.max()}",
                f"  .mean() = {obj.mean()}",
                f"  .std() = {obj.std()}",
                f"  .dtype = {obj.dtype}",
            ]
        )
    else:
        text = pformat(obj)
    return text


@register_node("JWPrintLatent", "Print Latent")
class _:
    CATEGORY = "jamesWalker55"
    INPUT_TYPES = lambda: {
        "required": {
            "value": ("LATENT",),
            "name": (
                "STRING",
                {"default": "latent", "multiline": True, "dynamicPrompts": False},
            ),
        }
    }
    RETURN_TYPES = ("LATENT",)
    OUTPUT_NODE = True
    FUNCTION = "execute"

    def execute(self, value: dict, name: str):
        print(f"{name} = {serialise_obj(value)}")

        return (value,)

    @classmethod
    def IS_CHANGED(cls, *args):
        # Always recalculate
        return float("NaN")
