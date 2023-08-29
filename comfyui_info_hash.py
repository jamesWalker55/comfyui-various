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
            "i": ("INT", {"default": 1, "min": 0, "step": 1, "max": 9999}),
            "ranges_key": ("STRING", {"default": "ranges", "multiline": False}),
        }
    }

    RETURN_TYPES = ("INFO_HASH",)

    OUTPUT_NODE = False

    FUNCTION = "execute"

    def execute(self, config: str, i: int, ranges_key: str):
        config = yaml.safe_load(config)

        info = RangedConfig(config, range_key=ranges_key)

        return (info.get_sub_prompt(i),)


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

    OUTPUT_NODE = False

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

    OUTPUT_NODE = False

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

    OUTPUT_NODE = False

    FUNCTION = "execute"

    def execute(self, info_hash: dict, key: str):
        val = str(info_hash[key])
        return (val,)


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
        from pprint import pprint, pformat

        pprint(info_hash)
        raise ValueError(pformat(info_hash))
