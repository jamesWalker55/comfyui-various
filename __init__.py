import importlib
import os

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Main nodes for all users
NODE_MODULES = [
    ".comfyui_image_ops",
    ".comfyui_primitive_ops",
    ".comfyui_raft",
    ".comfyui_image_channel_ops",
    ".comfyui_color_ops",
]

# Extra nodes for my own use
if (
    "COMFYUI_JW_ENABLE_EXTRA_NODES" in os.environ
    and os.environ["COMFYUI_JW_ENABLE_EXTRA_NODES"].lower() == "true"
):
    NODE_MODULES.extend(
        [
            ".comfyui_batch_io",
            ".comfyui_group_io",
            ".comfyui_cn_preprocessors",
        ]
    )


def load_nodes(module_name):
    global NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

    module = importlib.import_module(module_name, package=__name__)

    NODE_CLASS_MAPPINGS = {
        **NODE_CLASS_MAPPINGS,
        **module.NODE_CLASS_MAPPINGS,
    }
    NODE_DISPLAY_NAME_MAPPINGS = {
        **NODE_DISPLAY_NAME_MAPPINGS,
        **module.NODE_DISPLAY_NAME_MAPPINGS,
    }


for module_name in NODE_MODULES:
    load_nodes(module_name)
