import inspect
import math
import typing
from typing import Literal

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def register_node(identifier: str, display_name: str):
    def decorator(cls):
        NODE_CLASS_MAPPINGS[identifier] = cls
        NODE_DISPLAY_NAME_MAPPINGS[identifier] = display_name

        return cls

    return decorator


def generate_functional_node(
    category: str,
    identifier: str,
    display_name: str,
    *,
    multiline_string: bool = False,
    output_node: bool = False,
):
    def decorator(func):
        signature = inspect.signature(func)

        # generate INPUT_TYPES
        required_inputs = {}

        for name, param in signature.parameters.items():
            has_default = param.default is not param.empty
            param_type = param.annotation
            if param_type is int:
                if not has_default:
                    raise TypeError("INT input must have a default value")

                required_inputs[name] = (
                    "INT",
                    {
                        "default": param.default,
                        "min": -0xFFFFFFFFFFFFFFFF,
                        "max": 0xFFFFFFFFFFFFFFFF,
                    },
                )
            elif param_type is float:
                if not has_default:
                    raise TypeError("FLOAT input must have a default value")

                required_inputs[name] = (
                    "FLOAT",
                    {
                        "default": param.default,
                        "min": -99999999999999999.0,
                        "max": 99999999999999999.0,
                    },
                )
            elif param_type is str:
                if not has_default:
                    raise TypeError("STRING input must have a default value")

                required_inputs[name] = (
                    "STRING",
                    {
                        "default": param.default,
                        "multiline": multiline_string,
                        "dynamicPrompts": False,
                    },
                )
            elif isinstance(param_type, str):
                if has_default:
                    raise TypeError("Custom input types cannot have default values")

                required_inputs[name] = (param_type,)
            elif typing.get_origin(param_type) is Literal:
                choices = typing.get_args(param_type)
                if param.default is not None:
                    raise TypeError(
                        "Choice input types must have default value set to None"
                    )

                required_inputs[name] = (choices,)
            else:
                raise NotImplementedError(
                    f"Unsupported functional node type: {param_type}"
                )

        # generate RETURN_TYPES
        if signature.return_annotation is signature.empty:
            raise TypeError("Functional node must have annotation for return type")
        elif typing.get_origin(signature.return_annotation) is not tuple:
            raise TypeError("Functional node must return a tuple")

        return_types = []
        for annot in typing.get_args(signature.return_annotation):
            if isinstance(annot, str):
                return_types.append(annot)
            elif annot is int:
                return_types.append("INT")
            elif annot is float:
                return_types.append("FLOAT")
            elif annot is str:
                return_types.append("STRING")
            else:
                raise NotImplementedError(f"Unsupported return type: {annot}")

        @register_node(identifier, display_name)
        class _:
            CATEGORY = category
            INPUT_TYPES = lambda: {"required": required_inputs}
            RETURN_TYPES = tuple(return_types)
            OUTPUT_NODE = output_node
            FUNCTION = "execute"

            def execute(self, *args, **kwargs):
                return func(*args, **kwargs)

        return func

    return decorator


@generate_functional_node("jamesWalker55", "JWInteger", "Integer")
def _(value: int = 0) -> tuple[int]:
    return (value,)


@generate_functional_node("jamesWalker55", "JWIntegerToFloat", "Integer to Float")
def _(value: int = 0) -> tuple[float]:
    return (float(value),)


@generate_functional_node("jamesWalker55", "JWIntegerToString", "Integer to String")
def _(value: int = 0, format_string: str = "{:04d}") -> tuple[str]:
    return (format_string.format(value),)


@generate_functional_node("jamesWalker55", "JWIntegerAdd", "Integer Add")
def _(a: int = 0, b: int = 0) -> tuple[int]:
    return (a + b,)


@generate_functional_node("jamesWalker55", "JWIntegerSub", "Integer Subtract")
def _(a: int = 0, b: int = 0) -> tuple[int]:
    return (a - b,)


@generate_functional_node("jamesWalker55", "JWIntegerMul", "Integer Multiply")
def _(a: int = 0, b: int = 0) -> tuple[int]:
    return (a * b,)


@generate_functional_node("jamesWalker55", "JWIntegerDiv", "Integer Divide")
def _(a: int = 0, b: int = 0) -> tuple[float]:
    return (a / b,)


@generate_functional_node(
    "jamesWalker55", "JWIntegerAbsolute", "Integer Absolute Value"
)
def _(value: int = 0) -> tuple[int]:
    return (abs(value),)


@generate_functional_node("jamesWalker55", "JWIntegerMin", "Integer Minimum")
def _(a: int = 0, b: int = 0) -> tuple[int]:
    return (min(a, b),)


@generate_functional_node("jamesWalker55", "JWIntegerMax", "Integer Maximum")
def _(a: int = 0, b: int = 0) -> tuple[int]:
    return (max(a, b),)


@generate_functional_node("jamesWalker55", "JWFloat", "Float")
def _(value: float = 0) -> tuple[float]:
    return (value,)


@generate_functional_node("jamesWalker55", "JWFloatToInteger", "Float to Integer")
def _(
    value: float = 0, mode: Literal["round", "floor", "ceiling"] = None
) -> tuple[int]:
    if mode == "round":
        return (round(value),)
    elif mode == "floor":
        return (math.floor(value),)
    elif mode == "ceiling":
        return (math.ceil(value),)
    else:
        raise NotImplementedError(f"Unsupported mode: {mode}")


@generate_functional_node("jamesWalker55", "JWFloatToString", "Float to String")
def _(value: float = 0, format_string: str = "{:.6g}") -> tuple[str]:
    return (format_string.format(value),)


@generate_functional_node("jamesWalker55", "JWFloatAdd", "Float Add")
def _(a: float = 0, b: float = 0) -> tuple[float]:
    return (a + b,)


@generate_functional_node("jamesWalker55", "JWFloatSub", "Float Subtract")
def _(a: float = 0, b: float = 0) -> tuple[float]:
    return (a - b,)


@generate_functional_node("jamesWalker55", "JWFloatMul", "Float Multiply")
def _(a: float = 0, b: float = 0) -> tuple[float]:
    return (a * b,)


@generate_functional_node("jamesWalker55", "JWFloatDiv", "Float Divide")
def _(a: float = 0, b: float = 0) -> tuple[float]:
    return (a / b,)


@generate_functional_node("jamesWalker55", "JWFloatAbsolute", "Float Absolute Value")
def _(value: float = 0) -> tuple[float]:
    return (abs(value),)


@generate_functional_node("jamesWalker55", "JWFloatMin", "Float Minimum")
def _(a: float = 0, b: float = 0) -> tuple[float]:
    return (min(a, b),)


@generate_functional_node("jamesWalker55", "JWFloatMax", "Float Maximum")
def _(a: float = 0, b: float = 0) -> tuple[float]:
    return (max(a, b),)


@generate_functional_node("jamesWalker55", "JWString", "String")
def _(text: str = "") -> tuple[str]:
    return (text,)


@generate_functional_node("jamesWalker55", "JWStringToInteger", "String to Integer")
def _(text: str = "0") -> tuple[int]:
    return (int(text),)


@generate_functional_node("jamesWalker55", "JWStringToFloat", "String to Float")
def _(text: str = "0.0") -> tuple[float]:
    return (float(text),)


@generate_functional_node(
    "jamesWalker55", "JWStringMultiline", "String (Multiline)", multiline_string=True
)
def _(text: str = "") -> tuple[str]:
    return (text,)


@generate_functional_node("jamesWalker55", "JWStringConcat", "String Concatenate")
def _(a: str = "", b: str = "") -> tuple[str]:
    return (a + b,)


@generate_functional_node("jamesWalker55", "JWStringReplace", "String Replace")
def _(source: str = "", to_replace: str = "", replace_with: str = "") -> tuple[str]:
    return (source.replace(to_replace, replace_with),)


@generate_functional_node("jamesWalker55", "JWStringSplit", "String Split")
def _(
    source: str = "a,b",
    split_by: str = ",",
    from_right: Literal["false", "true"] = None,
) -> tuple[str, str]:
    from_right = from_right == "true"
    if from_right:
        splits = source.rsplit(split_by, 1)
    else:
        splits = source.split(split_by, 1)
    match splits:
        case a, b:
            return (a, b)
        case a:
            return (a, "")


@generate_functional_node("jamesWalker55", "JWStringGetLine", "String Get Line")
def _(source: str = "", line_index: int = 0) -> tuple[str]:
    return (source.splitlines()[line_index],)


@generate_functional_node("jamesWalker55", "JWStringUnescape", "String Unescape")
def _(text: str = "") -> tuple[str]:
    """parses '\\n' literals in a string to actual '\n' characters"""
    # convert to bytes, while converting unicode to escaped literals
    text_bytes = text.encode("ascii", "backslashreplace")
    # convert back to string, parsing backslash escapes
    text = text_bytes.decode("unicode-escape")
    return (text,)
