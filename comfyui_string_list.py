import torch


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def register_node(identifier: str, display_name: str):
    def decorator(cls):
        NODE_CLASS_MAPPINGS[identifier] = cls
        NODE_DISPLAY_NAME_MAPPINGS[identifier] = display_name

        return cls

    return decorator


@register_node("JWStringListFromString", "String List From String")
class _:
    CATEGORY = "jamesWalker55"
    INPUT_TYPES = lambda: {
        "required": {
            "value": ("STRING", {"default": "", "multiline": False}),
        }
    }
    RETURN_TYPES = ("STRING_LIST",)
    FUNCTION = "execute"

    def execute(self, value: str):
        val = [value]
        return (val,)


@register_node("JWStringListFromStrings", "String List From Strings")
class _:
    CATEGORY = "jamesWalker55"
    INPUT_TYPES = lambda: {
        "required": {
            "a": ("STRING", {"default": "", "multiline": False}),
            "b": ("STRING", {"default": "", "multiline": False}),
        }
    }
    RETURN_TYPES = ("STRING_LIST",)
    FUNCTION = "execute"

    def execute(self, a: str, b: str):
        val = [a, b]
        return (val,)


@register_node("JWStringListJoin", "Join String List")
class _:
    CATEGORY = "jamesWalker55"
    INPUT_TYPES = lambda: {
        "required": {
            "a": ("STRING_LIST",),
            "b": ("STRING_LIST",),
        }
    }
    RETURN_TYPES = ("STRING_LIST",)
    FUNCTION = "execute"

    def execute(self, a: list[str], b: list[str]):
        val = a + b
        return (val,)


@register_node("JWStringListRepeat", "Repeat String List")
class _:
    CATEGORY = "jamesWalker55"
    INPUT_TYPES = lambda: {
        "required": {
            "string_list": ("STRING_LIST",),
            "repeats": ("INT", {"default": 1, "min": 0}),
        }
    }
    RETURN_TYPES = ("STRING_LIST",)
    FUNCTION = "execute"

    def execute(self, string_list: list[str], repeats: int):
        val = string_list * repeats
        return (val,)


@register_node("JWStringListToString", "String List To String")
class _:
    CATEGORY = "jamesWalker55"
    INPUT_TYPES = lambda: {
        "required": {
            "string_list": ("STRING_LIST",),
            "join": (
                "STRING",
                {"default": "\n", "multiline": True, "dynamicPrompts": False},
            ),
        }
    }
    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"

    def execute(self, string_list: list[str], join: str):
        val = join.join(string_list)
        return (val,)


@register_node("JWStringListToFormatedString", "String List To Formatted String")
class _:
    CATEGORY = "jamesWalker55"
    INPUT_TYPES = lambda: {
        "required": {
            "string_list": ("STRING_LIST",),
            "template": (
                "STRING",
                {"default": "{}, {}, {}", "multiline": True, "dynamicPrompts": False},
            ),
        }
    }
    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"

    def execute(self, string_list: list[str], join: str):
        val = join.join(string_list)
        return (val,)


@register_node("JWStringListCLIPEncode", "String List CLIP Encode")
class _:
    CATEGORY = "jamesWalker55"
    INPUT_TYPES = lambda: {
        "required": {
            "string_list": ("STRING_LIST",),
            "clip": ("CLIP",),
        }
    }
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "execute"

    def execute(self, string_list: list[str], clip):
        all_cond = []
        all_pooled = []

        for text in string_list:
            tokens = clip.tokenize(text)
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
            # cond.shape => torch.Size([1, 77, 768])
            # pooled.shape => torch.Size([1, 768])
            all_cond.append(cond)
            all_pooled.append(pooled)

        all_cond = torch.cat(all_cond, dim=0)
        all_pooled = torch.cat(all_pooled, dim=0)
        return ([[all_cond, {"pooled_output": all_pooled}]],)
