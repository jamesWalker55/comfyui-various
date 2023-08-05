import importlib

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

NODE_MODULES = [
    ".comfyui_batch_io",
    ".comfyui_group_io",
    ".comfyui_image_ops",
    ".comfyui_primitive_ops",
]


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
