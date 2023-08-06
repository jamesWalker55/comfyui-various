# Various ComfyUI Nodes by Type

This repository provides various nodes for use in ComfyUI.

The nodes are grouped into separate files, you can just download the specific file you want to avoid filling your nodes list with nodes you don't need.

**Note:** This repo also contains nodes for my own personal use, **these nodes are very likely useless to anyone else so I recommend skipping those files**. See [Available Nodes] for more details.

## Installation

### Method 1 (Recommended): Download each file individually

Go though each file and see which nodes you want to use. Download the corresponding file and put it in:

```
ComfyUI/custom_nodes
```

### Method 2: Clone the repo

This method is **not recommended**, since it populates your node list with ALL nodes in this repository, including certain nodes that are likely useless to anyone except me (see [Available Nodes] for files to avoid).

The loaded nodes are controlled by the `__init__.py` file, I will change this file arbitrarily so an update to this repository may hide nodes that are still present in the code. **Use this installation method at your own risk.**

```
cd ComfyUI/custom_nodes
git clone https://github.com/jamesWalker55/comfyui-various
```

## Available Nodes

Each `comfyui_*.py` file contains a group of nodes of similar purpose. This repo is still in early stages so I can't write documentation for each file yet - have a look at the code for each file to see what they are for.

**`comfyui_primitive_ops.py`**:

```
JWInteger: Integer
JWIntegerToFloat: Integer to Float
JWIntegerToString: Integer to String
JWIntegerAdd: Integer Add
JWIntegerSub: Integer Subtract
JWIntegerMul: Integer Multiply
JWIntegerDiv: Integer Divide
JWFloat: Float
JWFloatToInteger: Float to Integer
JWFloatToString: Float to String
JWFloatAdd: Float Add
JWFloatSub: Float Subtract
JWFloatMul: Float Multiply
JWFloatDiv: Float Divide
JWString: String
JWStringToInteger: String to Integer
JWStringToFloat: String to Float
JWStringMultiline: String (Multiline)
JWStringConcat: String Concatenate
JWStringReplace: String Replace
```

**`comfyui_image_ops.py`**:

```
JWImageLoadRGB: Image Load RGB
JWImageResize: Image Resize
```

**`comfyui_image_channel_ops.py`**:

```
JWImageStackChannels: Image Stack Channels
```

**`comfyui_raft.py`**:

```
RAFTPreprocess: RAFT Preprocess
RAFTEstimate: RAFT Estimate
RAFTFlowToImage: RAFT Flow to Image
```

Some files contain nodes for my own personal use, and are likely completely useless to anyone else. The following files should be skipped:

- `comfyui_batch_io.py`
- `comfyui_group_io.py`
