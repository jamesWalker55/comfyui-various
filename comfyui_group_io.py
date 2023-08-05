import glob
import os
from pathlib import Path

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


def load_image(path):
    img = Image.open(path).convert("RGB")
    img = np.array(img).astype(np.float32) / 255.0
    img = torch.from_numpy(img)[None]
    return img


@register_node("JamesLoadImageGroup", "James: Load Image Group")
class JamesLoadImageGroup:
    """
    An opinionated batch image loader. This is used for loading groups for batch processing.

    Folder structure:

    ```plain
    groups/
        baseprompt.txt
        g1/
            0001.png
            0002.png
            subprompt.txt
        g2/
            0003.png
            0004.png
            subprompt.txt
        ...
    ```
    """

    CATEGORY = "jamesWalker55"

    INPUT_TYPES = lambda: {
        "required": {
            "groups_dir": ("STRING", {"default": "./groups", "multiline": False}),
            "groups_id": ("INT", {"default": 1, "min": 0, "step": 1, "max": 9999}),
            "base_prompt_name": (
                "STRING",
                {"default": "baseprompt.txt", "multiline": False},
            ),
            "sub_prompt_name": (
                "STRING",
                {"default": "subprompt.txt", "multiline": False},
            ),
            "negative_prompt_delimiter": (
                "STRING",
                {"default": "---", "multiline": False},
            ),
            "image_glob": (
                "STRING",
                {"default": "*.png", "multiline": False},
            ),
        }
    }

    RETURN_NAMES = (
        "POSITIVE_PROMPT",
        "NEGATIVE_PROMPT",
        "IMAGES",
        "FRAME_COUNT",
        "FILENAMES",
    )
    RETURN_TYPES = ("STRING", "STRING", "IMAGE", "INT", "STRING_LIST")

    OUTPUT_NODE = False

    FUNCTION = "execute"

    def execute(
        self,
        groups_dir: str,
        groups_id: int,
        base_prompt_name: str,
        sub_prompt_name: str,
        negative_prompt_delimiter: str,
        image_glob: str,
    ):
        assert isinstance(groups_dir, str)
        assert isinstance(groups_id, int)
        assert isinstance(base_prompt_name, str)
        assert isinstance(sub_prompt_name, str)
        assert isinstance(negative_prompt_delimiter, str)

        pos_prompt, neg_prompt = self.get_group_prompt(
            groups_dir,
            groups_id,
            base_prompt_name,
            sub_prompt_name,
            negative_prompt_delimiter,
        )
        images, filenames = self.load_group_images(groups_dir, groups_id, image_glob)

        print(
            f"JamesLoadImageGroup: {(pos_prompt, neg_prompt, len(filenames), filenames)!r}"
        )
        return (pos_prompt, neg_prompt, images, len(filenames), filenames)

    def get_base_prompt(
        self,
        groups_dir: str,
        base_prompt_name: str,
        negative_prompt_delimiter: str,
    ):
        """Get the base prompt of the group"""
        path = os.path.join(groups_dir, base_prompt_name)
        with open(path, "r", encoding="utf8") as f:
            prompt = f.read()

        match prompt.split(negative_prompt_delimiter, 1):
            case pos_prompt, neg_prompt:
                return pos_prompt, neg_prompt
            case [pos_prompt]:
                return pos_prompt, ""
            case _:
                raise ValueError("Invalid base prompt, more than 1 delimiter found")

    def get_group_path(self, groups_dir: str, groups_id: int):
        return os.path.join(groups_dir, f"g{groups_id}")

    def get_sub_prompt(
        self,
        groups_dir: str,
        groups_id: int,
        sub_prompt_name: str,
        negative_prompt_delimiter: str,
    ):
        """Get the sub prompt of the group"""
        group_path = self.get_group_path(groups_dir, groups_id)
        path = os.path.join(group_path, sub_prompt_name)
        with open(path, "r", encoding="utf8") as f:
            prompt = f.read()

        match prompt.split(negative_prompt_delimiter, 1):
            case pos_prompt, neg_prompt:
                return pos_prompt, neg_prompt
            case [pos_prompt]:
                return pos_prompt, ""
            case _:
                raise ValueError("Invalid sub prompt, more than 1 delimiter found")

    def get_group_prompt(
        self,
        groups_dir: str,
        groups_id: int,
        base_prompt_name: str,
        sub_prompt_name: str,
        negative_prompt_delimiter: str,
    ):
        """Generate the final combined prompt of the group"""
        base_pos, base_neg = self.get_base_prompt(
            groups_dir, base_prompt_name, negative_prompt_delimiter
        )
        sub_pos, sub_neg = self.get_sub_prompt(
            groups_dir, groups_id, sub_prompt_name, negative_prompt_delimiter
        )
        group_pos = base_pos.format(sub_pos)
        group_neg = base_neg.format(sub_neg)
        return group_pos, group_neg

    def load_group_images(self, groups_dir: str, groups_id: int, image_glob: str):
        """Get all images for the group"""
        group_path = self.get_group_path(groups_dir, groups_id)

        # convert paths to be relative to here
        paths = glob.glob(image_glob, root_dir=group_path, recursive=True)
        # convert paths to be relative to here
        paths = [os.path.join(group_path, x) for x in paths]
        # sort paths alphabetically
        paths.sort()

        # extract filenames without extension
        filenames = [os.path.splitext(os.path.basename(x))[0] for x in paths]

        # must have at least 1 image
        if len(paths) == 0:
            raise FileNotFoundError(
                f"No images found in folder matching pattern {image_glob!r}"
            )

        # load images
        imgs = []
        for p in paths:
            img = load_image(p)
            # img.shape => torch.Size([1, 768, 768, 3])
            imgs.append(img)

        imgs = torch.cat(imgs, dim=0)

        # sanity check, image count == filename count
        assert len(imgs) == len(filenames)

        return imgs, filenames


@register_node("GroupLoadBatchImages", "Group Load Batch Images")
class GroupLoadBatchImages:
    """
    An opinionated batch image loader. This is used for loading groups for batch processing.

    YAML structure:

    ```yaml
    positive: |
      {positive},
      simple background, white background,

    negative: |
      {negative},
      low quality,

    image_pattern: '{frame_id:04d}.png'

    groups:
      - start_id: 1
        positive: ...
        negative: ...
      - start_id: 5
        positive: ...
        negative: ...
      ...
    ```
    """

    CATEGORY = "jamesWalker55"

    INPUT_TYPES = lambda: {
        "required": {
            "definition_path": (
                "STRING",
                {"default": "./groups.yml", "multiline": False},
            ),
            "group_id": ("INT", {"default": 1, "min": 0, "step": 1, "max": 9999}),
        }
    }

    RETURN_NAMES = (
        "POSITIVE_PROMPT",
        "NEGATIVE_PROMPT",
        "IMAGES",
        "FRAME_COUNT",
        "FILENAMES",
        "GROUP_INFO",
    )
    RETURN_TYPES = ("STRING", "STRING", "IMAGE", "INT", "STRING_LIST", "GROUP_INFO")

    OUTPUT_NODE = False

    FUNCTION = "execute"

    TYPE_GROUP = dict[str, int | str]
    TYPE_DEFINITION = dict[str, str | list[TYPE_GROUP]]

    def execute(
        self,
        definition_path: str,
        group_id: int,
    ):
        assert isinstance(definition_path, str)
        assert isinstance(group_id, int)

        with open(definition_path, "r", encoding="utf8") as f:
            definition = yaml.safe_load(f)

        self.validate_format(definition, group_id)

        images, filenames = self.get_group_images(definition_path, definition, group_id)

        pos_prompt, neg_prompt = self.get_group_prompts(definition, group_id)

        group_info = self.generate_group_info(definition, group_id)

        print(
            f"JamesLoadYAMLImageGroup: {(pos_prompt, neg_prompt, len(filenames), filenames, group_info)!r}"
        )
        return (pos_prompt, neg_prompt, images, len(filenames), filenames, group_info)

    def validate_format(self, definition: TYPE_DEFINITION, group_id: int):
        assert isinstance(definition, dict), "file must be a dict"

        assert "positive" in definition, "missing key: positive"
        assert isinstance(definition["positive"], str), "positive must be a string"

        assert "negative" in definition, "missing key: negative"
        assert isinstance(definition["negative"], str), "negative must be a string"

        assert "image_pattern" in definition, "missing key: image_pattern"
        assert isinstance(definition["image_pattern"], str), "pattern must be a string"

        assert "groups" in definition, "missing key: groups"
        assert isinstance(definition["groups"], list), "groups must be a list"
        assert len(definition["groups"]) > 0, "must have at least 1 group"
        assert group_id < len(
            definition["groups"]
        ), f"not enough groups: {group_id} < {len(definition['groups'])}"

        prev_start_id = -1

        for gp in definition["groups"]:
            assert isinstance(gp, dict), "group must be a dict"

            assert "start_id" in gp, "group missing key: start_id"

            start_id = gp["start_id"]
            assert isinstance(start_id, int), "start_id must be a number"
            assert start_id >= 0, "start_id cannot be negative"
            assert prev_start_id < start_id, "start_id must be in ascending order"

            prev_start_id = start_id

    def generate_group_info(self, definition: TYPE_DEFINITION, group_id: int):
        format_info = definition["groups"][group_id].copy()
        format_info["group_id"] = group_id
        return format_info

    def get_group_frame_range(self, definition: TYPE_DEFINITION, group_id: int):
        groups = definition["groups"]

        start_frame_id: int = groups[group_id]["start_id"]

        end_frame_id: int | None = None
        if group_id < len(groups) - 1:
            # Not last group, last frame is the next group's start frame
            # Otherwise, must determine end frame ID dynamically
            end_frame_id = groups[group_id + 1]["start_id"]

        return start_frame_id, end_frame_id

    def get_group_image_path(
        self,
        definition_path: str,
        definition: TYPE_DEFINITION,
        group_id: int,
        frame_id: int,
    ):
        base_path = Path(definition_path).parent
        pattern = definition["image_pattern"]

        format_info = self.generate_group_info(definition, group_id)
        format_info["frame_id"] = frame_id

        return base_path / pattern.format(**format_info)

    def get_group_images(
        self, definition_path: str, definition: TYPE_DEFINITION, group_id: int
    ):
        start_frame, end_frame = self.get_group_frame_range(definition, group_id)

        images = []
        filenames = []

        i = start_frame
        while True:
            image_path = self.get_group_image_path(
                definition_path, definition, group_id, i
            )

            # check for end of sequence
            if end_frame is not None and i >= end_frame:
                # reached end of sequence
                break
            elif end_frame is None and not os.path.exists(image_path):
                # unknown end frame, and this frame is missing
                # assume this is the end of sequence
                break

            try:
                img = load_image(image_path)

                images.append(img)
                filenames.append(os.path.split(image_path)[1])
            except FileNotFoundError as e:
                print(f"WARNING: Image missing from sequence: {image_path}")

            i += 1

        images = torch.cat(images, dim=0)

        # sanity check, image count == filename count
        assert len(images) == len(filenames)

        return images, filenames

    def get_group_prompts(
        self,
        definition: TYPE_DEFINITION,
        group_id: int,
    ):
        base_pos = definition["positive"]
        base_neg = definition["negative"]

        format_info = self.generate_group_info(definition, group_id)

        final_pos = base_pos.format(**format_info)
        final_neg = base_neg.format(**format_info)

        return final_pos, final_neg


@register_node("GroupInfoExtractInt", "Group Info Extract Integer")
class GroupInfoExtractInt:
    CATEGORY = "jamesWalker55"

    INPUT_TYPES = lambda: {
        "required": {
            "group_info": ("GROUP_INFO",),
            "key": ("STRING", {"default": ""}),
        }
    }

    RETURN_NAMES = ("INT",)
    RETURN_TYPES = ("INT",)

    OUTPUT_NODE = False

    FUNCTION = "execute"

    def execute(
        self,
        group_info: dict,
        key: str,
    ):
        assert isinstance(group_info, dict)
        assert isinstance(key, str)

        val = int(group_info[key])

        return (val,)


@register_node("GroupInfoExtractFloat", "Group Info Extract Float")
class GroupInfoExtractFloat:
    CATEGORY = "jamesWalker55"

    INPUT_TYPES = lambda: {
        "required": {
            "group_info": ("GROUP_INFO",),
            "key": ("STRING", {"default": ""}),
        }
    }

    RETURN_NAMES = ("FLOAT",)
    RETURN_TYPES = ("FLOAT",)

    OUTPUT_NODE = False

    FUNCTION = "execute"

    def execute(
        self,
        group_info: dict,
        key: str,
    ):
        assert isinstance(group_info, dict)
        assert isinstance(key, str)

        val = float(group_info[key])

        return (val,)
