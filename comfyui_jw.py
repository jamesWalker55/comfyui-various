import torch

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def register_node(identifier: str, display_name: str):
    def decorator(cls):
        NODE_CLASS_MAPPINGS[identifier] = cls
        NODE_DISPLAY_NAME_MAPPINGS[identifier] = display_name

        return cls

    return decorator


@register_node("JWReferenceOnly", "James: Reference Only")
class ReferenceOnlySimple:
    CATEGORY = "jamesWalker55"

    INPUT_TYPES = lambda: {
        "required": {
            "model": ("MODEL",),
            "reference": ("LATENT",),
            "initial_latent": ("LATENT",),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
        }
    }

    RETURN_TYPES = ("MODEL", "LATENT")

    FUNCTION = "execute"

    def execute(self, model, reference, initial_latent, batch_size):
        model_reference = model.clone()
        size_latent = list(reference["samples"].shape)
        size_latent[0] = batch_size
        latent = {}
        latent["samples"] = initial_latent["samples"]

        batch = latent["samples"].shape[0] + reference["samples"].shape[0]

        def reference_apply(q, k, v, extra_options):
            k = k.clone().repeat(1, 2, 1)

            for o in range(0, q.shape[0], batch):
                for x in range(1, batch):
                    k[x + o, q.shape[1] :] = q[o, :]

            return q, k, k

        model_reference.set_model_attn1_patch(reference_apply)

        out_latent = torch.cat((reference["samples"], latent["samples"]))
        if "noise_mask" in latent:
            mask = latent["noise_mask"]
        else:
            mask = torch.ones((64, 64), dtype=torch.float32, device="cpu")

        if len(mask.shape) < 3:
            mask = mask.unsqueeze(0)
        if mask.shape[0] < latent["samples"].shape[0]:
            print(latent["samples"].shape, mask.shape)
            mask = mask.repeat(latent["samples"].shape[0], 1, 1)

        out_mask = torch.zeros(
            (1, mask.shape[1], mask.shape[2]), dtype=torch.float32, device="cpu"
        )
        return (
            model_reference,
            {"samples": out_latent, "noise_mask": torch.cat((out_mask, mask))},
        )


@register_node("JWMultilineTest", "James: Multiline Test")
class _:
    CATEGORY = "jamesWalker55"

    INPUT_TYPES = lambda: {
        "required": {
            "text": ("STRING", {"multiline": True, "dynamicPrompts": False}),
        }
    }

    RETURN_TYPES = ()

    OUTPUT_NODE = True

    FUNCTION = "execute"

    def execute(self, text: str):
        from pprint import pprint, pformat

        pprint(text)

        raise RuntimeError(pformat(text))
