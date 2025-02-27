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
    ".comfyui_datetime",
    ".comfyui_image_sequence",
    ".comfyui_mask_sequence_ops",
    ".comfyui_default",
    ".comfyui_sound",
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
            ".comfyui_jw",
            ".comfyui_info_hash",
            ".comfyui_debug",
            ".comfyui_string_list",
            ".comfyui_uncrop",
            ".comfyui_rc",
        ]
    )


def load_nodes(module_name: str):
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


def write_nodes_list(module_names: list[str]):
    this_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(this_dir, "nodes.log")

    lines = []

    for module_name in module_names:
        module = importlib.import_module(module_name, package=__name__)

        lines.append(module_name.strip("."))

        for identifier, display_name in module.NODE_DISPLAY_NAME_MAPPINGS.items():
            lines.append(f"  {identifier}: {display_name}")

        lines.append("")

    lines = "\n".join(lines)

    with open(path, "w", encoding="utf8") as f:
        f.write(lines)


for module_name in NODE_MODULES:
    load_nodes(module_name)

# write_nodes_list(NODE_MODULES)
