import json
import math
import os
from pathlib import Path
from typing import Literal, TypedDict

import soundfile as sf
import torch
import torchaudio

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def register_node(identifier: str, display_name: str):
    def decorator(cls):
        NODE_CLASS_MAPPINGS[identifier] = cls
        NODE_DISPLAY_NAME_MAPPINGS[identifier] = display_name

        return cls

    return decorator


FLOAT_MAX = 99999999999999999.0


class Audio(TypedDict):
    sample_rate: int
    # .shape => torch.Size([1, 2, 1403182])
    waveform: torch.Tensor


def scalar_to_db(scalar: float):
    return 20 * math.log10(scalar)


def db_to_scalar(db: float):
    return 10 ** (db / 20)


def load_audio(
    path: str | Path,
    sr: None | int | float = None,
    offset: float = 0.0,
    duration: float | None = None,
    make_stereo: bool = True,
) -> Audio:
    import librosa

    mix, sr = librosa.load(path, sr=sr, mono=False, offset=offset, duration=duration)
    mix = torch.from_numpy(mix)

    # If stereo, shape will be:
    #   torch.Size([2, 1403182])
    # If mono, shape will be:
    #   torch.Size([1403182])
    #
    # Ensure shape is [channels, data]
    if len(mix.shape) == 1:
        mix = torch.stack([mix], dim=0)
    assert len(mix.shape) == 2

    # Convert mono to stereo if needed
    if make_stereo:
        if mix.shape[0] == 1:
            mix = torch.cat([mix, mix], dim=0)
        elif mix.shape[0] == 2:
            pass
        else:
            raise ValueError(
                f"Input audio has {mix.shape[0]} channels, cannot convert to stereo (2 channels)"
            )

    # Add extra dimension for batch size

    # shape => torch.Size([2, 1403182])
    mix = torch.unsqueeze(mix, 0)
    # shape => torch.Size([1, 2, 1403182])

    return {
        "sample_rate": round(sr),
        "waveform": mix,
    }


def save_audio(path: str | Path, mix: torch.Tensor, sr):
    path = str(path)

    # make sure tensor has shape [channels, data]
    if len(mix.shape) == 3:
        if mix.shape[0] > 1:
            raise ValueError("Audio batch size is more than 1")
        mix = mix[0]
    elif len(mix.shape) == 2:
        pass
    elif len(mix.shape) == 1:
        mix = torch.unsqueeze(mix, 0)
    else:
        raise ValueError(f"Invalid tensor shape: {mix.shape}")

    subtype = "FLOAT" if path.lower().endswith("wav") else None
    sf.write(path, mix.T, sr, subtype=subtype)


def write_audio_comment(path: str | Path, comment: str):
    try:
        from mediafile import MediaFile
    except ImportError as e:
        print(
            "[WARN] Failed to import `mediafile`, saved audio files will not have metadata"
        )
        return

    f = MediaFile(path)
    f.comments = comment
    f.save()


@register_node("JWLoadAudio", "Audio Load")
class _:
    CATEGORY = "jamesWalker55"
    INPUT_TYPES = lambda: {
        "required": {
            "path": ("STRING", {"default": "./audio.mp3"}),
            "gain_db": ("FLOAT", {"default": 0, "min": -100, "max": 100}),
            "offset_seconds": ("FLOAT", {"default": 0, "min": 0, "max": FLOAT_MAX}),
            "duration_seconds": ("FLOAT", {"default": 0, "min": 0, "max": FLOAT_MAX}),
            "resample_to_hz": ("FLOAT", {"default": 0, "min": 0, "max": FLOAT_MAX}),
            "make_stereo": ("BOOLEAN", {"default": True}),
        }
    }
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "execute"

    def execute(
        self,
        path: str,
        gain_db: float,
        offset_seconds: float,
        duration_seconds: float,
        resample_to_hz: float,
        make_stereo: bool,
    ) -> tuple[Audio]:
        rv = load_audio(
            path,
            sr=resample_to_hz if resample_to_hz > 0 else None,
            offset=offset_seconds,
            duration=duration_seconds if duration_seconds > 0 else None,
            make_stereo=make_stereo,
        )
        if gain_db != 0.0:
            gain_scalar = db_to_scalar(gain_db)
            rv["waveform"] = gain_scalar * rv["waveform"]

        return (rv,)

    @classmethod
    def IS_CHANGED(
        cls,
        path: str,
        *args,
    ):
        if os.path.exists(path):
            mtime = os.path.getmtime(path)
        else:
            mtime = None

        return (mtime, path, *args)


@register_node("JWAudioBlend", "Audio Blend")
class _:
    CATEGORY = "jamesWalker55"
    INPUT_TYPES = lambda: {
        "required": {
            "a": ("AUDIO",),
            "b": ("AUDIO",),
            "ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
            "if_durations_differ": (
                ("use_longest", "use_shortest"),
                {"default": "use_longest"},
            ),
            "if_samplerates_differ": (
                ("use_highest", "use_lowest"),
                {"default": "use_highest"},
            ),
        }
    }
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "execute"

    def execute(
        self,
        a: Audio,
        b: Audio,
        ratio: float,
        if_durations_differ: Literal["use_longest", "use_shortest"],
        if_samplerates_differ: Literal["use_highest", "use_lowest"],
    ) -> tuple[Audio]:
        import librosa

        # shallow clone audios
        a = {**a}
        b = {**b}

        # # if they have different batch size, attempt to resolve them
        # if a["waveform"].shape[0] != b["waveform"].shape[0]:
        #     pass

        # if they have different channels, attempt to resolve them
        if a["waveform"].shape[1] != b["waveform"].shape[1]:
            # if one of them is mono, distribute it
            if a["waveform"].shape[1] == 1:
                a["waveform"] = a["waveform"].expand(-1, b["waveform"].shape[1])
            elif b["waveform"].shape[1] == 1:
                b["waveform"] = b["waveform"].expand(-1, a["waveform"].shape[1])

        # ensure audio has same sample rate
        if a["sample_rate"] != b["sample_rate"]:
            # determine which rate to use
            if if_samplerates_differ == "use_highest":
                sr = max(a["sample_rate"], b["sample_rate"])
            elif if_samplerates_differ == "use_lowest":
                sr = min(a["sample_rate"], b["sample_rate"])
            else:
                raise NotImplementedError(if_samplerates_differ)

            # do the resampling
            if a["sample_rate"] != sr:
                a["waveform"] = torchaudio.functional.resample(
                    a["waveform"], a["sample_rate"], sr
                )
            if b["sample_rate"] != sr:
                b["waveform"] = torchaudio.functional.resample(
                    b["waveform"], b["sample_rate"], sr
                )

        # ensure input has same length
        if a["waveform"].shape[-1] != b["waveform"].shape[-1]:
            # determine which duration to use
            if if_durations_differ == "use_longest":
                duration = max(a["waveform"].shape[-1], b["waveform"].shape[-1])
            elif if_durations_differ == "use_shortest":
                duration = min(a["waveform"].shape[-1], b["waveform"].shape[-1])
            else:
                raise NotImplementedError(if_samplerates_differ)

            def waveform_with_duration(wave: torch.Tensor, new_duration: int):
                batch, channels, original_duration = wave.shape
                if original_duration >= new_duration:
                    return wave[:, :, new_duration]
                else:
                    rv = torch.zeros(batch, channels, new_duration)
                    rv[:, :, original_duration] = wave[:, :, original_duration]
                    return rv

            # do the chopping
            if a["waveform"].shape[-1] != duration:
                a["waveform"] = waveform_with_duration(a["waveform"], duration)
            if b["waveform"].shape[-1] != duration:
                b["waveform"] = waveform_with_duration(b["waveform"], duration)

        rv: Audio = {
            "sample_rate": sr,
            "waveform": a["waveform"] * (1.0 - ratio) + a["waveform"] * ratio,
        }

        return (rv,)


class ResultItem(TypedDict):
    filename: str
    subfolder: str
    type: Literal["output"]


@register_node("JWAudioSaveToPath", "Audio Save to Path")
class _:
    CATEGORY = "jamesWalker55"
    INPUT_TYPES = lambda: {
        "required": {
            "audio": ("AUDIO",),
            "path": ("STRING", {"default": "./audio.mp3"}),
            "overwrite": ("BOOLEAN", {"default": True}),
        },
        "hidden": {
            "prompt": "PROMPT",
            "extra_pnginfo": "EXTRA_PNGINFO",
        },
    }
    RETURN_TYPES = ()
    FUNCTION = "execute"
    OUTPUT_NODE = True

    def execute(
        self,
        path: str | Path,
        audio: Audio,
        overwrite: bool,
        prompt=None,
        extra_pnginfo=None,
    ):
        path = Path(path)

        path.parent.mkdir(exist_ok=True)

        metadata = {**(extra_pnginfo or {})}
        if prompt is not None:
            metadata["prompt"] = prompt
        metadata_str = json.dumps(metadata)

        results: list[ResultItem] = []

        if audio["waveform"].shape[0] == 1:
            # batch has 1 audio only
            if overwrite or not path.exists():
                save_audio(
                    path,
                    audio["waveform"][0],
                    audio["sample_rate"],
                )
                write_audio_comment(path, metadata_str)
                results.append(
                    {
                        "filename": path.name,
                        "subfolder": str(path.parent),
                        "type": "output",
                    }
                )
        else:
            # batch has multiple images
            for i, subwaveform in enumerate(audio["waveform"]):
                subpath = path.with_stem(f"{path.stem}-{i}")
                if overwrite or not path.exists():
                    save_audio(
                        subpath,
                        subwaveform,
                        audio["sample_rate"],
                    )
                    write_audio_comment(subpath, metadata_str)
                    results.append(
                        {
                            "filename": subpath.name,
                            "subfolder": str(subpath.parent),
                            "type": "output",
                        }
                    )

        return {"ui": {"audio": results}}
