import argparse
import logging
from pathlib import Path
from typing import BinaryIO

_SUPPORTED_FORMATS = ("png", "gif", "webp")
_MEASURE_TABLE = {
    "element": ("element",),
    "frame": ("frame",),
    "both": (
        "element",
        "frame",
    ),
}


def _guess_format(format: str | None, output: Path, default: str) -> str:
    if format is not None:
        return format

    suffix = output.suffix.lower()
    if suffix and suffix[1:] in _SUPPORTED_FORMATS:
        return suffix[1:]
    return default


def inspect(args: list[str] | None = None) -> None:
    from kanimtool.inspect import print_anim, print_build
    from kanimtool.parser import Parser

    p = argparse.ArgumentParser()
    p.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="Enable verbose inspect",
    )
    p.add_argument(
        "file",
        type=argparse.FileType("rb"),
        help="anim or build file",
    )

    ns = p.parse_args(args)
    verbose = ns.verbose
    file: BinaryIO = ns.file

    with file:
        data: bytes = file.read()

    parser = Parser()
    if data[0:4] == b"BILD":
        build = parser.parse_build(data)
        print_build(
            build,
            print_frames=verbose >= 1,
        )
    elif data[0:4] == b"ANIM":
        anim = parser.parse_anim(data)
        print_anim(
            anim,
            print_symbols=verbose >= 1,
            print_frames=verbose >= 2,
        )
    else:
        p.exit(-1, f"Bad file format: {data[0:4]!r}")


def _parse_mappings(lines: list[str]) -> list[tuple[str, str | None]]:
    result: list[tuple[str, str | None]] = []

    for line in lines:
        for item in line.split(","):
            symbol, eq, repl = item.partition("=")

            if repl:
                result.append((symbol, repl))
            else:
                result.append((symbol, None))

    return result


def _parse_affix(line: str) -> tuple[str | None, str, str]:
    if line.count("=") >= 2 or line.count(":") >= 2:
        raise ValueError(f"Bad affix: {line!r}")

    batch_name: str | None
    batch_name, eq, affix = line.partition("=")
    if not eq:
        affix = batch_name
        batch_name = None

    prefix, colon, suffix = affix.partition(":")
    return batch_name, prefix, suffix


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="Enable verbose logging",
    )

    p.add_argument(
        "--build",
        type=Path,
        required=True,
        action="append",
        help="build file(s)",
    )
    p.add_argument(
        "--texture",
        type=Path,
        required=True,
        action="append",
        help="texture file(s)",
    )

    p.add_argument(
        "--margin",
        type=float,
        default=0,
        help="margin around image",
    )
    p.add_argument(
        "--scale",
        type=float,
        default=1,
        help="scale factor",
    )
    p.add_argument(
        "--resampling",
        choices=("nearest", "bilinear", "bicubic"),
        default="nearest",
    )
    p.add_argument(
        "--measure",
        choices=tuple(_MEASURE_TABLE.keys()),
        default="frame",
        help="measure canvas size by frame, element, or both",
    )

    p.add_argument(
        "--outline",
        action="store_true",
        default=False,
        help="show canvas and symbol frame outlines for debugging",
    )
    p.add_argument(
        "--format",
        choices=_SUPPORTED_FORMATS,
        default=None,
        help="output file format",
    )

    p.add_argument(
        "--affix",
        default=None,
        help="Apply affixies",
    )
    p.add_argument(
        "--map",
        default=[],
        action="append",
        help="remapping symbol name",
    )

    p.add_argument("anim", type=Path, help="anim file")
    p.add_argument("target", type=str, help="animation name")
    p.add_argument("output", type=Path, help="output file name")
    return p


def make(args: list[str] | None = None) -> None:
    p = _make_parser()

    ns = p.parse_args(args)
    verbose = ns.verbose

    if verbose >= 1:
        logging_level = logging.DEBUG
    else:
        logging_level = logging.WARNING

    logging.basicConfig(level=logging_level, force=True)

    build_paths: list[Path] = ns.build
    texture_paths: list[Path] = ns.texture
    anim_path: Path = ns.anim

    margin = ns.margin
    scale = ns.scale
    resampling: str = ns.resampling
    measure_option = ns.measure

    debug_outline: bool = ns.outline
    format: str | None = ns.format

    anim_name: str = ns.target
    output_path: Path = ns.output
    affix: str | None = ns.affix
    mappings: list[str] = ns.map

    from PIL import Image

    from kanimtool.builder import BuildRegistry, save_animation
    from kanimtool.parser import Parser

    #
    format = _guess_format(format, output_path, _SUPPORTED_FORMATS[0])
    if format not in _SUPPORTED_FORMATS:
        p.exit(1, f"Bad format: {format!r}")

    if len(build_paths) != len(texture_paths):
        p.exit(message="the number of --build and --texture must be equal")

    parser = Parser()
    anim_group = parser.parse_anim(anim_path.read_bytes())

    registry = BuildRegistry.from_files(
        list(zip(build_paths, texture_paths)),
        parser=parser,
    )

    if affix:
        build_name, prefix, suffix = _parse_affix(affix)
        if build_name is None:
            build_name = registry.get_build_names()[-1]
        registry.apply_affixes(build_name, prefix, suffix)

    if mappings:
        registry.apply_mapping(_parse_mappings(mappings))

    measure_methods = _MEASURE_TABLE[measure_option]

    frames = []
    frame_rate = 30.0
    for anim_title in anim_name.split(","):
        animation = anim_group.get_animation(anim_title)
        frame_rate = animation.frame_rate
        frames.extend(animation.frames)

    if not frames:
        p.exit(1, "No frames were added")

    from kanimtool.builder import AnimationBuilder

    builder = AnimationBuilder(
        registry,
        frames,
        margins=margin,
        scale=scale,
        measure_methods=measure_methods,
        resampling=getattr(Image.Resampling, resampling.upper()),
        debug_outline=debug_outline,
    )

    images = []
    for frame_idx in range(len(frames)):
        im = builder.draw_frame(frame_idx)
        images.append(im)

    data = save_animation(
        images,
        format=format,
        frame_rate=frame_rate,
    )
    output_path.write_bytes(data)
