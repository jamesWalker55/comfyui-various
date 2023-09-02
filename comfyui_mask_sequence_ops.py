import torch

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def register_node(identifier: str, display_name: str):
    def decorator(cls):
        NODE_CLASS_MAPPINGS[identifier] = cls
        NODE_DISPLAY_NAME_MAPPINGS[identifier] = display_name

        return cls

    return decorator


@register_node("JWMaskSequenceFromMask", "Mask Sequence From Mask")
class _:
    CATEGORY = "jamesWalker55"
    INPUT_TYPES = lambda: {
        "required": {
            "mask": ("MASK",),
            "batch_size": ("INT", {"default": 1, "min": 1, "step": 1}),
        }
    }
    RETURN_TYPES = ("MASK_SEQUENCE",)
    FUNCTION = "execute"

    def execute(
        self,
        mask: torch.Tensor,
        batch_size: int,
    ):
        assert isinstance(mask, torch.Tensor)
        assert isinstance(batch_size, int)
        assert len(mask.shape) == 2

        mask_seq = mask.reshape((1, 1, *mask.shape))
        mask_seq = mask_seq.repeat(batch_size, 1, 1, 1)

        return (mask_seq,)


@register_node("JWMaskSequenceJoin", "Join Mask Sequence")
class _:
    CATEGORY = "jamesWalker55"
    INPUT_TYPES = lambda: {
        "required": {
            "mask_sequence_1": ("MASK_SEQUENCE",),
            "mask_sequence_2": ("MASK_SEQUENCE",),
        }
    }
    RETURN_TYPES = ("MASK_SEQUENCE",)
    FUNCTION = "execute"

    def execute(
        self,
        mask_sequence_1: torch.Tensor,
        mask_sequence_2: torch.Tensor,
    ):
        assert isinstance(mask_sequence_1, torch.Tensor)
        assert isinstance(mask_sequence_2, torch.Tensor)

        mask_seq = torch.cat((mask_sequence_1, mask_sequence_2), dim=0)

        return (mask_seq,)


@register_node("JWMaskSequenceApplyToLatent", "Apply Mask Sequence to Latent")
class _:
    CATEGORY = "jamesWalker55"
    INPUT_TYPES = lambda: {
        "required": {
            "samples": ("LATENT",),
            "mask_sequence": ("MASK_SEQUENCE",),
        }
    }
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "execute"

    def execute(
        self,
        samples: dict,
        mask_sequence: torch.Tensor,
    ):
        assert isinstance(samples, dict)
        assert isinstance(mask_sequence, torch.Tensor)

        samples = samples.copy()

        samples["noise_mask"] = mask_sequence

        return (samples,)
