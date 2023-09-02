import copy
import glob
import os
import textwrap
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import torch
import yaml
from PIL import Image

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def register_node(identifier: str, display_name: str):
    def decorator(cls):
        NODE_CLASS_MAPPINGS[identifier] = cls
        NODE_DISPLAY_NAME_MAPPINGS[identifier] = display_name

        return cls

    return decorator


def load_image(path, convert="RGB"):
    img = Image.open(path).convert(convert)
    img = np.array(img).astype(np.float32) / 255.0
    img = torch.from_numpy(img).unsqueeze(0)
    return img


class RangedConfig:
    def __init__(self, definition: dict[str, Any], range_key: str = "ranges") -> None:
        self.definition = definition
        self.range_key = range_key
        self._validate()

    def _validate(self):
        for k, v in self.definition.items():
            if k == self.range_key:
                assert isinstance(v, dict), f"{type(v)!r}"
                for kk, vv in v.items():
                    # sub prompts ranges
                    assert isinstance(kk, int), f"{type(kk)!r}"
                    assert isinstance(vv, dict), f"{type(vv)!r}"
                    for kkk, vvv in vv.items():
                        # actual sub prompts
                        assert isinstance(kkk, str), f"{type(kkk)!r}"
                        if vvv is None:
                            vvv = ""
                        assert isinstance(vvv, (int, float, str)), f"{type(vvv)!r}"
            else:
                # base prompts
                assert isinstance(k, str), f"{type(k)!r}"
                if v is None:
                    v = ""
                assert isinstance(v, (int, float, str)), f"{type(v)!r}"

    def get_ranges(self):
        return sorted(self.definition[self.range_key].keys())

    def _get_range_start(self, i: int) -> int | None:
        if len(self.definition[self.range_key]) == 0:
            return None

        range_starts = sorted(self.definition[self.range_key].keys())

        for range_start_idx, range_start in enumerate(range_starts):
            if i < range_start:
                if range_start_idx == 0:
                    return None
                else:
                    return range_starts[range_start_idx - 1]

        return range_starts[-1]

    def _get_raw_sub_prompt(self, i: int):
        range_start = self._get_range_start(i)
        if range_start is None:
            # not in range, just use base definition
            return {**self.definition, "i": i}
        else:
            raw_sub_prompt = self.definition[self.range_key][self._get_range_start(i)]
            return {**self.definition, **raw_sub_prompt, "i": i}

    def get_sub_prompt(self, i: int):
        raw_sub_prompt = self._get_raw_sub_prompt(i)
        sub_prompt = {}
        for k, v in raw_sub_prompt.items():
            if k == self.range_key:
                continue

            if isinstance(v, str):
                v = v.format(**raw_sub_prompt)

            sub_prompt[k] = v
        return sub_prompt


DEFAULT_CONFIG = """\
p: |
  masterpiece, best quality,
  {sp},

n: |
  {sn},
  embedding:EasyNegative, embedding:bad-artist, embedding:bad-hands-5, embedding:bad-image-v2-39000,
  lowres, ((bad anatomy)), ((bad hands)), text, missing finger, extra digits, fewer digits, blurry, ((mutated hands and fingers)), (poorly drawn face), ((mutation)), ((deformed face)), (ugly), ((bad proportions)), ((extra limbs)), extra face, (double head), (extra head), ((extra feet)), monster, logo, cropped, worst quality, low quality, normal quality, jpeg, humpbacked, long body, long neck, ((jpeg artifacts)),

path: "{i:04d}.png"

example: 0

ranges:
  1:
    sp: positive subprompt for 1-4
    sn: negative subprompt for 1-4
  5:
    sp: positive subprompt for 5-...
    sn: negative subprompt for 5-...
    example: 1
"""


@register_node("JWInfoHashFromRangedInfo", "Info Hash From Ranged Config")
class _:
    CATEGORY = "jamesWalker55"
    INPUT_TYPES = lambda: {
        "required": {
            "config": (
                "STRING",
                {"default": DEFAULT_CONFIG, "multiline": True, "dynamicPrompts": False},
            ),
            "i": ("INT", {"default": 1, "min": 0, "step": 1, "max": 999999}),
            "ranges_key": ("STRING", {"default": "ranges", "multiline": False}),
        }
    }
    RETURN_TYPES = ("INFO_HASH",)
    FUNCTION = "execute"

    def execute(self, config: str, i: int, ranges_key: str):
        config = yaml.safe_load(config)

        info = RangedConfig(config, range_key=ranges_key)

        return (info.get_sub_prompt(i),)


@register_node("JWInfoHashListFromRangedInfo", "Info Hash List From Ranged Config")
class _:
    CATEGORY = "jamesWalker55"
    INPUT_TYPES = lambda: {
        "required": {
            "config": (
                "STRING",
                {"default": DEFAULT_CONFIG, "multiline": True, "dynamicPrompts": False},
            ),
            "i_start": ("INT", {"default": 0, "min": 0, "step": 1, "max": 999999}),
            "i_stop": ("INT", {"default": 16, "min": 0, "step": 1, "max": 999999}),
            "ranges_key": ("STRING", {"default": "ranges", "multiline": False}),
            "inclusive": (("false", "true"), {"default": "false"}),
        }
    }
    RETURN_TYPES = ("INFO_HASH_LIST",)
    FUNCTION = "execute"

    def execute(
        self, config: str, i_start: int, i_stop: int, ranges_key: str, inclusive: str
    ):
        inclusive: bool = inclusive == "true"

        config = yaml.safe_load(config)

        info = RangedConfig(config, range_key=ranges_key)
        subinfos = [
            info.get_sub_prompt(i)
            for i in range(i_start, i_stop + 1 if inclusive else i_stop)
        ]

        return (subinfos,)


def calculate_batches(
    i_start: int,  # start of i
    i_stop: int,  # end of i, excludes end
    range_starts: int,  # scene cuts, batch will be terminated before this
    max_batch_size: int,  # maximum length of batch
):
    """
    :param int i_start: start of i
    :param int i_stop: end of i, excludes end
    :param int range_starts: scene cuts, batch will be terminated before this
    :param int max_batch_size: maximum length of batch
    :return: a list of 2-tuples, each represents (batch start frame, batch stop frame), where stop frame is exclusive
    """
    batch_starts: list[int] = []  # also includes end frame
    i = i_start - 1
    counter = -1
    while True:
        i += 1
        counter += 1
        if i >= i_stop:
            batch_starts.append(i)
            break

        if i in range_starts:
            batch_starts.append(i)
            counter = 0
            continue

        if counter >= max_batch_size:
            batch_starts.append(i)
            counter = 0
            continue

        if counter == 0:
            batch_starts.append(i)
            continue

    batches = list(zip(batch_starts[:-1], batch_starts[1:]))

    return batches


@register_node("JWRangedInfoCalculateSubBatch", "Calculate Sub Batch for Ranged Info")
class _:
    CATEGORY = "jamesWalker55"
    INPUT_TYPES = lambda: {
        "required": {
            "config": (
                "STRING",
                {"default": DEFAULT_CONFIG, "multiline": True, "dynamicPrompts": False},
            ),
            "ranges_key": ("STRING", {"default": "ranges", "multiline": False}),
            "batch_idx": ("INT", {"default": 0, "min": 0, "step": 1, "max": 999999}),
            "i_start": ("INT", {"default": 1, "min": 0, "step": 1, "max": 999999}),
            "i_stop": ("INT", {"default": 100, "min": 0, "step": 1, "max": 999999}),
            "max_batch_size": (
                "INT",
                {"default": 16, "min": 1, "step": 1, "max": 999999},
            ),
            "inclusive": (("false", "true"), {"default": "false"}),
        }
    }
    RETURN_NAMES = ("BATCH_I_START", "BATCH_I_STOP")
    RETURN_TYPES = ("INT", "INT")
    FUNCTION = "execute"

    def execute(
        self,
        config: str,
        ranges_key: str,
        batch_idx: int,
        i_start: int,
        i_stop: int,
        max_batch_size: int,
        inclusive: str,
    ):
        inclusive: bool = inclusive == "true"

        config = yaml.safe_load(config)

        info = RangedConfig(config, range_key=ranges_key)

        range_starts = set(info.get_ranges())

        # get images in selected batch
        batches = calculate_batches(
            i_start, i_stop + 1 if inclusive else i_stop, range_starts, max_batch_size
        )
        batch = batches[batch_idx]

        return (batch[0], batch[1])


@register_node(
    "JWInfoHashFromRangedInfoAndLoadSubsequences",
    "Info Hash From Ranged Config and Load Batch",
)
class _:
    CATEGORY = "jamesWalker55"
    INPUT_TYPES = lambda: {
        "required": {
            "config": (
                "STRING",
                {"default": DEFAULT_CONFIG, "multiline": True, "dynamicPrompts": False},
            ),
            "ranges_key": ("STRING", {"default": "ranges", "multiline": False}),
            "path_key": ("STRING", {"default": "path", "multiline": False}),
            "batch_idx": ("INT", {"default": 0, "min": 0, "step": 1, "max": 999999}),
            "i_start": ("INT", {"default": 1, "min": 0, "step": 1, "max": 999999}),
            "i_stop": ("INT", {"default": 100, "min": 0, "step": 1, "max": 999999}),
            "max_batch_size": (
                "INT",
                {"default": 16, "min": 1, "step": 1, "max": 999999},
            ),
            "inclusive": (("false", "true"), {"default": "false"}),
        }
    }
    RETURN_NAMES = ("INFO_HASH", "IMAGE", "BATCH_I_START", "BATCH_I_STOP")
    RETURN_TYPES = ("INFO_HASH", "IMAGE", "INT", "INT")
    FUNCTION = "execute"

    def execute(
        self,
        config: str,
        ranges_key: str,
        path_key: str,
        batch_idx: int,
        i_start: int,
        i_stop: int,
        max_batch_size: int,
        inclusive: str,
    ):
        inclusive: bool = inclusive == "true"

        config = yaml.safe_load(config)

        info = RangedConfig(config, range_key=ranges_key)

        range_starts = set(info.get_ranges())

        # get images in selected batch
        batches = calculate_batches(
            i_start, i_stop + 1 if inclusive else i_stop, range_starts, max_batch_size
        )
        batch = batches[batch_idx]

        print(f"Getting images in batch: {batch}")

        images = []
        for i in range(batch[0], batch[1]):
            subinfo = info.get_sub_prompt(i)
            path = subinfo[path_key]
            print(f"  Loading: {path}")
            img = load_image(path)
            images.append(img)
        images = torch.cat(images, dim=0)

        return (info.get_sub_prompt(batch[0]), images, batch[0], batch[1])


@register_node("JWInfoHashExtractInteger", "Info Hash Extract Integer")
class _:
    CATEGORY = "jamesWalker55"
    INPUT_TYPES = lambda: {
        "required": {
            "info_hash": ("INFO_HASH",),
            "key": ("STRING", {"default": "i", "multiline": False}),
        }
    }
    RETURN_TYPES = ("INT",)
    FUNCTION = "execute"

    def execute(self, info_hash: dict, key: str):
        val = int(info_hash[key])
        return (val,)


@register_node("JWInfoHashExtractFloat", "Info Hash Extract Float")
class _:
    CATEGORY = "jamesWalker55"
    INPUT_TYPES = lambda: {
        "required": {
            "info_hash": ("INFO_HASH",),
            "key": ("STRING", {"default": "key", "multiline": False}),
        }
    }
    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "execute"

    def execute(self, info_hash: dict, key: str):
        val = float(info_hash[key])
        return (val,)


@register_node("JWInfoHashExtractString", "Info Hash Extract String")
class _:
    CATEGORY = "jamesWalker55"
    INPUT_TYPES = lambda: {
        "required": {
            "info_hash": ("INFO_HASH",),
            "key": ("STRING", {"default": "p", "multiline": False}),
        }
    }
    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"

    def execute(self, info_hash: dict, key: str):
        val = str(info_hash[key])
        return (val,)


@register_node("JWInfoHashListExtractStringList", "Info Hash List Extract String List")
class _:
    CATEGORY = "jamesWalker55"
    INPUT_TYPES = lambda: {
        "required": {
            "info_hash_list": ("INFO_HASH_LIST",),
            "key": ("STRING", {"default": "p", "multiline": False}),
        }
    }
    RETURN_TYPES = ("STRING_LIST",)
    FUNCTION = "execute"

    def execute(self, info_hash_list: list[dict], key: str):
        val = [str(info_hash[key]) for info_hash in info_hash_list]
        return (val,)


@register_node("JWInfoHashFromInfoHashList", "Extract Info Hash From Info Hash List")
class _:
    CATEGORY = "jamesWalker55"
    INPUT_TYPES = lambda: {
        "required": {
            "info_hash_list": ("INFO_HASH_LIST",),
            "i": ("INT", {"default": 0, "step": 1, "min": -99999999, "max": 99999999}),
        }
    }
    RETURN_TYPES = ("INFO_HASH",)
    FUNCTION = "execute"

    def execute(self, info_hash_list: list[dict], i: int):
        return (info_hash_list[i],)


@register_node("JWInfoHashPrint", "Print Info Hash (Debug)")
class _:
    CATEGORY = "jamesWalker55"
    INPUT_TYPES = lambda: {
        "required": {
            "info_hash": ("INFO_HASH",),
        }
    }
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "execute"

    def execute(self, info_hash: dict):
        from pprint import pformat, pprint

        pprint(info_hash)
        raise ValueError(pformat(info_hash))
