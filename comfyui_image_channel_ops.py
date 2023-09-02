import torch

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def register_node(identifier: str, display_name: str):
    def decorator(cls):
        NODE_CLASS_MAPPINGS[identifier] = cls
        NODE_DISPLAY_NAME_MAPPINGS[identifier] = display_name

        return cls

    return decorator


@register_node("JWImageStackChannels", "Image Stack Channels")
class _:
    CATEGORY = "jamesWalker55"
    INPUT_TYPES = lambda: {
        "required": {
            "image_a": ("IMAGE",),
            "image_b": ("IMAGE",),
        }
    }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    def execute(self, image_a: torch.Tensor, image_b: torch.Tensor):
        assert isinstance(image_a, torch.Tensor)
        assert isinstance(image_b, torch.Tensor)

        stacked = torch.cat((image_a, image_b), dim=3)

        return (stacked,)
