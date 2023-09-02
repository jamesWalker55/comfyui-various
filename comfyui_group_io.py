import copy
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
    img = torch.from_numpy(img).unsqueeze(0)
    return img


@register_node("JamesLoadImageGroup", "[DEPRECATED] James: Load Image Group")
class _:
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
    RETURN_TYPES = ("STRING", "STRING", "IMAGE", "INT", "STRING")
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
        return (pos_prompt, neg_prompt, images, len(filenames), "\n".join(filenames))

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


class GroupedWorkspace:
    """
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

    _original_definition: dict
    _base_path: Path
    _base_pos: str
    _base_neg: str
    _image_pattern: str
    _groups: list[dict]

    def __init__(self, base_path: Path, definition: dict):
        self._validate_definition(definition)
        self._original_definition = definition
        self._base_path = base_path
        self._parse_groups(definition)

    @classmethod
    def open(cls, path, base_path=None):
        if base_path is None:
            base_path = Path(path).parent
        else:
            base_path = Path(base_path)

        with open(path, "r", encoding="utf8") as f:
            definition = yaml.safe_load(f)
        return cls(base_path, definition)

    @staticmethod
    def _validate_definition(definition):
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

        assert "start_id" not in definition, "'start_id' not allowed at root"
        assert "group_id" not in definition, "'group_id' not allowed in definition"

        prev_start_id = -1

        for gp in definition["groups"]:
            assert isinstance(gp, dict), "group must be a dict"

            assert "start_id" in gp, "group missing key: start_id"
            assert "group_id" not in gp, "'group_id' not allowed in definition"

            start_id = gp["start_id"]
            assert isinstance(start_id, int), "start_id must be a number"
            assert start_id >= 0, "start_id cannot be negative"
            assert prev_start_id < start_id, "start_id must be in ascending order"

            prev_start_id = start_id

    def _parse_groups(self, definition: dict):
        definition = copy.deepcopy(definition)

        self._base_pos = definition.pop("positive")
        self._base_neg = definition.pop("negative")
        self._image_pattern = definition.pop("image_pattern")
        raw_groups = definition.pop("groups")
        assert "start_id" not in definition

        self._groups = []

        for group in raw_groups:
            assert "start_id" in group
            assert isinstance(group["start_id"], int)

            # add extra keys in definition to group info
            group = {**definition, **group}

            self._groups.append(group)

    def _get_group_info(self, group_id: int):
        group = self._groups[group_id]
        return {**group, "group_id": group_id}

    def get_group_info(self, group_id: int):
        return copy.deepcopy(self._get_group_info(group_id))

    def _get_frame_info(self, group_id: int, frame_id: int):
        info = self._get_group_info(group_id)
        return {**info, "frame_id": frame_id}

    def get_frame_info(self, frame_id: int):
        group_id = self._frame_id_to_group_id(frame_id)
        return copy.deepcopy(self._get_frame_info(group_id, frame_id))

    def _get_positive_prompt(self, group_id: int):
        prompt = self._base_pos.format(**self._get_group_info(group_id))
        return prompt

    def _get_negative_prompt(self, group_id: int):
        prompt = self._base_neg.format(**self._get_group_info(group_id))
        return prompt

    def _get_image_path(self, group_id: int, frame_id: int):
        relpath = self._image_pattern.format(**self._get_frame_info(group_id, frame_id))
        return self._base_path / relpath

    def _get_group_frame_range(self, group_id: int) -> tuple[int, int | None]:
        start_frame_id: int = self._groups[group_id]["start_id"]

        if group_id < len(self._groups) - 1:
            # Not last group, last frame is the next group's start frame
            # Otherwise, must determine end frame ID dynamically
            return start_frame_id, self._groups[group_id + 1]["start_id"]
        else:
            return start_frame_id, None

    def _frame_id_to_group_id(self, frame_id: int):
        for i, group in enumerate(self._groups):
            if frame_id >= group["start_id"]:
                # frame ID is higher than this group
                continue

            # frame ID belongs to previous group
            if i == 0:
                raise ValueError(f"Frame ID {frame_id} is not covered by any group")

            return i - 1

        # return last group
        return len(self._groups) - 1

    def get_frame_image(self, frame_id: int):
        group_id = self._frame_id_to_group_id(frame_id)
        image_path = self._get_image_path(group_id, frame_id)

        img = load_image(image_path)
        filename = os.path.splitext(os.path.basename(image_path))[0]

        return img, filename

    def get_frame_prompts(self, frame_id: int):
        group_id = self._frame_id_to_group_id(frame_id)
        return self._get_positive_prompt(group_id), self._get_negative_prompt(group_id)

    def get_group_prompts(self, group_id: int):
        return self._get_positive_prompt(group_id), self._get_negative_prompt(group_id)

    def get_group_images(self, group_id: int):
        start_frame, end_frame = self._get_group_frame_range(group_id)

        images = []
        filenames: list[str] = []

        i = start_frame
        while True:
            image_path = self._get_image_path(group_id, i)

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
                filenames.append(image_path.stem)
            except FileNotFoundError as e:
                print(f"WARNING: Image missing from sequence: {image_path}")

            i += 1

        images = torch.cat(images, dim=0)

        # sanity check, image count == filename count
        assert len(images) == len(filenames)

        return images, filenames


@register_node("GroupLoadBatchImages", "[DEPRECATED] Group Load Batch Images")
class __:
    """
    An opinionated batch image loader. This is used for loading groups for batch processing.

    "base_path" controls where the images are loaded relative from. Defaults to the
    folder containing the definition file.
    """

    CATEGORY = "jamesWalker55"
    INPUT_TYPES = lambda: {
        "required": {
            "definition_path": (
                "STRING",
                {"default": "./groups.yml", "multiline": False},
            ),
            "group_id": ("INT", {"default": 1, "min": 0, "step": 1, "max": 9999}),
            "base_path": ("STRING", {"default": ""}),
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
    RETURN_TYPES = ("STRING", "STRING", "IMAGE", "INT", "STRING", "GROUP_INFO")
    FUNCTION = "execute"

    def execute(self, definition_path: str, group_id: int, base_path: str):
        assert isinstance(definition_path, str)
        assert isinstance(group_id, int)
        assert isinstance(base_path, str)

        base_path = base_path.strip()
        if len(base_path) == 0:
            base_path = None

        workspace = GroupedWorkspace.open(definition_path, base_path=base_path)

        images, filenames = workspace.get_group_images(group_id)
        pos_prompt, neg_prompt = workspace.get_group_prompts(group_id)
        group_info = workspace.get_group_info(group_id)

        return (
            pos_prompt,
            neg_prompt,
            images,
            len(filenames),
            "\n".join(filenames),
            group_info,
        )


@register_node("GroupLoadImage", "[DEPRECATED] Group Load Image")
class _:
    """
    An opinionated image loader. This is used for loading groups for batch processing.

    "base_path" controls where the images are loaded relative from. Defaults to the
    folder containing the definition file.
    """

    CATEGORY = "jamesWalker55"
    INPUT_TYPES = lambda: {
        "required": {
            "definition_path": (
                "STRING",
                {"default": "./groups.yml", "multiline": False},
            ),
            "frame_id": ("INT", {"default": 1, "min": 0, "step": 1, "max": 9999}),
            "base_path": ("STRING", {"default": ""}),
        }
    }
    RETURN_NAMES = (
        "POSITIVE_PROMPT",
        "NEGATIVE_PROMPT",
        "IMAGE",
        "FILENAME",
        "GROUP_INFO",
    )
    RETURN_TYPES = ("STRING", "STRING", "IMAGE", "STRING", "GROUP_INFO")
    FUNCTION = "execute"

    def execute(self, definition_path: str, frame_id: int, base_path: str):
        assert isinstance(definition_path, str)
        assert isinstance(frame_id, int)
        assert isinstance(base_path, str)

        base_path = base_path.strip()
        if len(base_path) == 0:
            base_path = None

        workspace = GroupedWorkspace.open(definition_path, base_path=base_path)

        image, filename = workspace.get_frame_image(frame_id)
        pos_prompt, neg_prompt = workspace.get_frame_prompts(frame_id)
        group_info = workspace.get_frame_info(frame_id)

        return (pos_prompt, neg_prompt, image, filename, group_info)


@register_node("GroupInfoExtractInt", "[DEPRECATED] Group Info Extract Integer")
class _:
    CATEGORY = "jamesWalker55"
    INPUT_TYPES = lambda: {
        "required": {
            "group_info": ("GROUP_INFO",),
            "key": ("STRING", {"default": ""}),
        }
    }
    RETURN_TYPES = ("INT",)
    FUNCTION = "execute"

    def execute(self, group_info: dict, key: str):
        assert isinstance(group_info, dict)
        assert isinstance(key, str)

        val = int(group_info[key])

        return (val,)


@register_node("GroupInfoExtractFloat", "[DEPRECATED] Group Info Extract Float")
class _:
    CATEGORY = "jamesWalker55"
    INPUT_TYPES = lambda: {
        "required": {
            "group_info": ("GROUP_INFO",),
            "key": ("STRING", {"default": ""}),
        }
    }
    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "execute"

    def execute(self, group_info: dict, key: str):
        assert isinstance(group_info, dict)
        assert isinstance(key, str)

        val = float(group_info[key])

        return (val,)
