# Various ComfyUI Nodes by Type

This repository provides various nodes for use in ComfyUI.

The nodes are grouped into separate files, you can just download the specific file you want to avoid filling your nodes list with nodes you don't need.

## Installation

### Method 1 (Recommended): Download each file individually

Go though each file and see which nodes you want to use. Download the corresponding file and put it in:

```
ComfyUI/custom_nodes
```

If you want to use `RAFTLoadFlowFromEXRChannels` from `comfyui_raft.py`, you must install `OpenEXR` in your ComfyUI Python environment.

```sh
# Activate your Python environment first.
pip install OpenEXR
```

### Method 2: Clone the repo

This method populates your node list with most nodes in this repository, which may be annoying to some (e.g. me) so I recommend using **Method 1** to keep your nodes list organised.

If you're happy with installing most nodes in this repository, clone the repository to your `custom_nodes` folder:

```
cd ComfyUI/custom_nodes
git clone https://github.com/jamesWalker55/comfyui-various
```

## Available Nodes

Each `comfyui_*.py` file contains a group of nodes of similar purpose. This repo is still in early stages so I can't write documentation for each file yet - have a look at the code for each file to see what they are for.

```
comfyui_image_ops
  JWImageLoadRGB: Image Load RGB
  JWImageLoadRGBA: Image Load RGBA
  JWLoadImagesFromString: Load Images From String
  JWImageSaveToPath: Image Save To Path
  JWImageExtractFromBatch: Image Extract From Batch
  JWImageBatchCount: Get Image Batch Count
  JWImageResize: Image Resize
  JWImageFlip: Image Flip
  JWMaskResize: Mask Resize
  JWMaskLikeImageSize: Mask Like Image Size
  JWImageResizeToSquare: Image Resize to Square
  JWImageResizeByFactor: Image Resize by Factor
  JWImageResizeByShorterSide: Image Resize by Shorter Side
  JWImageResizeByLongerSide: Image Resize by Longer Side
  JWImageResizeToClosestSDXLResolution: Image Resize to Closest SDXL Resolution
  JWImageLoadRGBFromClipboard: Image Load RGB From Clipboard
  JWImageLoadRGBA From Clipboard: Image Load RGBA From Clipboard

comfyui_primitive_ops
  JWInteger: Integer
  JWIntegerToFloat: Integer to Float
  JWIntegerToString: Integer to String
  JWIntegerAdd: Integer Add
  JWIntegerSub: Integer Subtract
  JWIntegerMul: Integer Multiply
  JWIntegerDiv: Integer Divide
  JWIntegerAbsolute: Integer Absolute Value
  JWIntegerMin: Integer Minimum
  JWIntegerMax: Integer Maximum
  JWFloat: Float
  JWFloatToInteger: Float to Integer
  JWFloatToString: Float to String
  JWFloatAdd: Float Add
  JWFloatSub: Float Subtract
  JWFloatMul: Float Multiply
  JWFloatDiv: Float Divide
  JWFloatAbsolute: Float Absolute Value
  JWFloatMin: Float Minimum
  JWFloatMax: Float Maximum
  JWString: String
  JWStringToInteger: String to Integer
  JWStringToFloat: String to Float
  JWStringMultiline: String (Multiline)
  JWStringConcat: String Concatenate
  JWStringReplace: String Replace
  JWStringSplit: String Split
  JWStringGetLine: String Get Line
  JWStringUnescape: String Unescape

comfyui_raft
  RAFTEstimate: RAFT Estimate
  RAFTFlowToImage: RAFT Flow to Image
  RAFTLoadFlowFromEXRChannels: RAFT Load Flow from EXR Channels

comfyui_image_channel_ops
  JWImageStackChannels: Image Stack Channels

comfyui_color_ops
  JWImageMix: Image Mix
  JWImageContrast: Image Contrast
  JWImageSaturation: Image Saturation
  JWImageLevels: Image Levels

comfyui_datetime
  JWDatetimeString: Datetime String

comfyui_image_sequence
  JWLoadImageSequence: Batch Load Image Sequence
  JWLoadImageSequenceWithStopIndex: Batch Load Image Sequence With Stop Index
  JWImageSequenceExtractFromBatch: Extract Image Sequence From Batch
  JWSaveImageSequence: Batch Save Image Sequence
  JWLoopImageSequence: Loop Image Sequence

comfyui_mask_sequence_ops
  JWMaskSequenceFromMask: Mask Sequence From Mask
  JWMaskSequenceJoin: Join Mask Sequence
  JWMaskSequenceApplyToLatent: Apply Mask Sequence to Latent
```

### Other nodes

Some files contain nodes for my own personal use, and are likely completely useless to anyone else. These nodes are hidden by default but can be enabled by setting the environment variable `COMFYUI_JW_ENABLE_EXTRA_NODES` to `true`. These files are hidden by default:

- `comfyui_batch_io.py`
- `comfyui_group_io.py`
- `comfyui_cn_preprocessors.py` _(Use [Fannovel16's preprocessor nodes](https://github.com/Fannovel16/comfy_controlnet_preprocessors) instead, they're way better)_
