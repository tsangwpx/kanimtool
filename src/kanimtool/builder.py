import dataclasses
import logging
import math
from io import BytesIO
from pathlib import Path
from typing import Any, Self, Sequence, TypeAlias, cast

import numpy as np
from PIL import Image
from PIL.ImageDraw import ImageDraw

from kanimtool.parser import AnimFrame, Build, FrameElement, Parser, Symbol, SymbolFrame

ImageTuple: TypeAlias = tuple[Image.Image, ...]


def _matrix_translate(x: float, y: float) -> np.ndarray:
    """translation matrix"""
    return np.array(
        (
            (1, 0, x),
            (0, 1, y),
            (0, 0, 1),
        )
    )


def _matrix_scale(sx: float, sy: float) -> np.ndarray:
    """scale matrix"""
    return np.array(
        (
            (sx, 0, 0),
            (0, sy, 0),
            (0, 0, 1),
        ),
        dtype=float,
    )


class BuildRegistry:
    """
    BuildRegistry store builds for accessing symbols and related images

    A build consist of a build file and several images.
    By default, symbols from lately added build have priority over previous symbols.
    Remapping mechanism allow redirecting or hiding particular symbols in symbol resolving.
    Remapping by affix is bultin on top of basic remapping.
    """

    @classmethod
    def from_files(
        cls,
        spec: Sequence[tuple[Path, ...]],
        *,
        parser: Parser | None = None,
    ) -> Self:
        if parser is None:
            parser = Parser()

        registry = cls(
            strings=parser.strings,
        )

        for build_path, *image_paths in spec:
            build = parser.parse_build(build_path.read_bytes())
            images = []
            for im_path in image_paths:
                im = Image.open(im_path, "r")
                im.load()
                images.append(im)

            registry.add_build(build, images)

        return registry

    def __init__(
        self,
        *,
        logger: logging.Logger | None = None,
        strings: dict[int, str] | None = None,
    ):
        if logger is None:
            logger = logging.getLogger("%s.%s" % (__name__, type(self).__qualname__))

        self.logger = logger

        if strings is None:
            strings = {}

        self._strings = strings
        self._builds: dict[str, tuple[Build, ImageTuple]] = {}
        self._symbols: dict[int | str, tuple[Symbol, ImageTuple]] = {}

        # remapping table is an overlay sit on top of symbols table
        # the difference is that
        # 1. key must be int
        # 2. value can be None
        self._remapping: dict[int, tuple[Symbol, ImageTuple] | tuple[None, None]] = {}

    def apply_affixes(self, build_name: str, prefix: str, suffix: str) -> None:
        """
        Replace symbols with ones with paricular prefix and particular suffix

        :param build_name: the build which provide symbol replacement
        :param prefix: the prefix
        :param suffix: the suffix
        :return:
        """

        strings = self._strings
        build, images = self._builds[build_name]

        if not prefix and not suffix:
            raise ValueError("both prefix and suffix are empty")

        for replacement in build.symbols:
            repl_name = strings.get(replacement.hash)
            if repl_name is None:
                continue

            target_name = repl_name
            target_name = target_name.removeprefix(prefix)
            target_name = target_name.removesuffix(suffix)
            if target_name == repl_name:
                continue

            pair = self._symbols.get(target_name)
            if pair is None:
                continue

            target, _ = pair
            self._remapping[target.hash] = (replacement, images)

    def apply_mapping(self, mapping: Sequence[tuple[str, str | None]]) -> None:
        for symbol_name, repl in mapping:
            try:
                symbol, _ = self._symbols[symbol_name]
            except KeyError:
                self.logger.warning(f"Missing symbol {symbol_name!r}")
                continue

            if repl is None:
                self._remapping[symbol.hash] = (None, None)
            else:
                try:
                    pair = self._symbols[repl]
                except KeyError:
                    self.logger.warning(f"Missing symbol {repl!r}")
                    continue
                else:
                    self._remapping[symbol.hash] = pair

    def add_build(
        self,
        build: Build,
        images: Sequence[Image.Image],
    ) -> None:
        name = build.name
        images = tuple(images)

        if name in self._builds:
            raise ValueError(f"Build name {name!r} existed")

        entry = (build, images)
        self._builds[name] = entry
        self._strings.update(build.strings)

        for symbol in build.symbols:
            symbol_hash = symbol.hash
            pair = (symbol, images)

            if symbol_hash in self._symbols:
                self.logger.debug(
                    "Overwrite %r %d from %r to %r",
                    self._strings.get(symbol_hash),
                    symbol_hash,
                    None,
                    name,
                )

            self._symbols[symbol_hash] = pair

            try:
                symbol_name = self._strings[symbol_hash]
            except KeyError:
                self.logger.warning("Missing symbol name for %d", symbol_hash)
            else:
                self._symbols[symbol_name] = pair

    def get_symbols(self) -> list[Symbol]:
        return [s for (k, (s, _)) in self._symbols.items() if isinstance(k, int)]

    def resolve_symbol(
        self,
        key: int,
    ) -> tuple[Symbol, ImageTuple] | tuple[None, None]:
        """
        Resolve the symbol after remapping

        :param key: symbol hash
        :return: symbol and images; or both None if intentionally unavailable
        :raise LookupError if symbol cannot be found
        """

        # hacky default value rather than None
        pair = self._remapping.get(key)
        if pair is not None:
            return pair

        try:
            return self._symbols[key]
        except KeyError:
            raise LookupError(key) from None

    def get_build(self, build_name: str) -> tuple[Build, ImageTuple]:
        try:
            return self._builds[build_name]
        except KeyError:
            raise LookupError(build_name) from None

    def get_build_names(self) -> list[str]:
        return list(self._builds.keys())


@dataclasses.dataclass
class FrameElementSketch:
    element: FrameElement
    """Associated FrameElement"""

    symbol_hash: int
    """symbol hash"""

    image: Image.Image
    """the texture image index"""

    image_box: tuple[int, int, int, int]
    """the cropping box (left, top, right, bottom) in the texture image"""

    size: tuple[float, float]
    """size after scaling image_box"""

    pivot: tuple[float, float]
    """relative to top-left corner in `size` box"""

    alpha: float
    """transparency"""

    forward: np.ndarray
    """affine transformation of image_box"""


@dataclasses.dataclass
class FrameSketch:
    frame: AnimFrame
    elements: list[FrameElementSketch]


class ImageSketcher:
    """
    ImageSketcher processes Frame / FrameElement and returns useful information for drawing them.
    """

    def __init__(
        self,
        registry: BuildRegistry,
        *,
        logger: logging.Logger | None = None,
    ):
        if logger is None:
            logger = logging.getLogger("%s.%s" % (__name__, type(self).__qualname__))

        self.logger = logger
        self.registry = registry

    def sketch_frame(
        self,
        frame: AnimFrame,
    ) -> FrameSketch:
        elem_sketches = []

        for elem_idx in range(len(frame.elements))[::-1]:
            elem_sketch = self._sketch_frame_element(frame, elem_idx)
            if elem_sketch is None:
                continue

            elem_sketches.append(elem_sketch)

        return FrameSketch(
            frame=frame,
            elements=elem_sketches,
        )

    def _sketch_frame_element(
        self,
        frame: AnimFrame,
        elem_idx: int,
    ) -> FrameElementSketch | None:
        elem = frame.elements[elem_idx]

        try:
            symbol, images = self.registry.resolve_symbol(elem.symbol)
        except LookupError:
            self.logger.warning(
                f"Symbol {elem.symbol} cannot be resolved"
                f" when processing frame element {elem_idx}",
                exc_info=True,
            )
            return None

        if symbol is None or images is None:
            return None

        symbol_frame = symbol.get_sequential_frame(elem.frame)

        im_idx = symbol_frame.build_image_idx
        if not 0 <= im_idx < len(images):
            raise ValueError(f"Image index {im_idx} out of range {len(images)}")

        sketch = self.sketch_element(elem, symbol, symbol_frame, images[im_idx])

        return sketch

    def sketch_element(
        self,
        elem: FrameElement,
        symbol: Symbol,
        symbol_frame: SymbolFrame,
        image: Image.Image,
    ) -> FrameElementSketch:
        """Compute useful information when placing a FrameElement"""

        symbol_hash = symbol.hash

        image_box = symbol_frame.compute_box(image.size)

        im_left, im_upper, im_right, im_lower = image_box
        im_width = im_right - im_left
        im_height = im_lower - im_upper

        bx, by, bw, bh = symbol_frame.bounds

        # first, scale the image to correct size
        scale_x = bw / im_width
        scale_y = bh / im_height
        scale = _matrix_scale(scale_x, scale_y)

        # second, move the origin to the pivot
        px = bw / 2 - bx
        py = bh / 2 - by
        translate = _matrix_translate(-px, -py)

        # third, apply the transform given by the FrameElement
        affine = np.array(elem.transform + (0, 0, 1)).reshape(3, 3)

        transform = affine @ translate @ scale

        palette_values = sorted(elem.mult_alpha)
        if len(set(palette_values)) >= 3 or 1.0 not in palette_values:
            # Currently, the product carry the meaning of transparency.
            # The data is stored in either alpha or RGB channel but not both.
            # so either of them must be one.
            # If the data is stored in RGB channels, all RGB values are the same.
            raise ValueError(f"Bad alpha {elem.mult_alpha!r}")

        alpha = palette_values[0] * palette_values[1]

        return FrameElementSketch(
            element=elem,
            symbol_hash=symbol_hash,
            image=image,
            image_box=image_box,
            alpha=alpha,
            pivot=(px, py),
            size=(bw, bh),
            forward=transform,
        )


@dataclasses.dataclass
class CanvasInfo:
    size: tuple[int, int]
    pivot: tuple[float, float]

    transform: np.ndarray
    """
    transform from wold coordinate to canvas coordinate
    """


class LayoutManager:
    """
    LayoutManager collects sketches and produces canvas info.


    Three points
    1. the origin (top left concer)
    2. the pivot, P
    3. the bottom right concer, Q relative to P

    the canvas size is given by Q relative to origin

    This class collects the maximum of P and Q to compute the canvas size

    After computing above cavnas size, the folowings are taken in order:

    1. margins are added to canvas size
    2. canvas size is scaled accordinly

    the canvas pivot is placed properly in the canvas  of the latest size
    """

    margins: tuple[int, int, int, int]
    canvas_pivot: tuple[float, float]
    canvas_size: tuple[int, int]
    canvas_info: CanvasInfo

    def __init__(
        self,
        scale: float | tuple[float, float] | None = None,
        margins: tuple[int, int, int, int] | int | None = None,
    ):
        if scale is None:
            scale = 1
        if isinstance(scale, (int, float)):
            scale = (scale, scale)

        if not scale or any(s <= 0 for s in scale):
            raise ValueError(f"Bad scale {scale!r}")

        if margins is None:
            margins = 0
        if isinstance(margins, (int, float)):
            margins = (margins, margins, margins, margins)

        self.scale: tuple[float, float] = scale
        self.margins = margins

        # Pivot
        self._px_max = 0.0
        self._py_max = 0.0

        # bottom right, relative to pivot
        self._qx_max = 0.0
        self._qy_max = 0.0

        # max frame size suggested by AnimFrame
        self._width_max = 0.0
        self._height_max = 0.0

    def update(self) -> CanvasInfo:
        px = self._px_max
        py = self._py_max
        qx = self._qx_max
        qy = self._qy_max
        margin_left, margin_top, margin_right, margin_bottom = self.margins
        scale_x, scale_y = self.scale

        full_width = max(px + qx, self._width_max) + margin_left + margin_right
        full_height = max(py + qy, self._height_max) + margin_top + margin_bottom

        px_ratio = (px + margin_left) / full_width
        py_ratio = (py + margin_top) / full_height

        canvas_width = math.ceil(full_width * scale_x)
        canvas_height = math.ceil(full_height * scale_y)

        cpx = canvas_width * px_ratio
        cpy = canvas_height * py_ratio

        info = CanvasInfo(
            size=(canvas_width, canvas_height),
            pivot=(cpx, cpy),
            transform=_matrix_translate(cpx, cpy) @ _matrix_scale(scale_x, scale_y),
        )

        self.canvas_size = info.size
        self.canvas_pivot = info.pivot
        self.canvas_info = info

        return info

    def feed_frames(self, frames: Sequence[AnimFrame]) -> None:
        px = py = 0.0
        qx = qy = 0.0

        width_max = self._width_max
        height_max = self._height_max

        for frame in frames:
            # iterate all frames, keep the most extreme bounding box
            dx, dy, fw, fh = frame.bounds

            width_max = max(width_max, fw)
            height_max = max(height_max, fh)

            # In the pivot reference frame,
            # (0, 0) is the pivot
            # (dx, dy) is the center of bounding box
            # the left and right side of bounding box are
            # (dx - fw / 2) and (dx + fw / 2)
            # the top and bottom side of bounding box are
            # (dy - fh / 2) and (dy + fh / 2)

            # In the reference frame in the top right corner of the canvas
            # (fw / 2 - dx, fh / 2 - dy) is the pivot
            px = max(px, fw / 2 - dx)
            py = max(py, fh / 2 - dy)
            qx = max(qx, dx + fw / 2)
            qy = max(qy, dy + fh / 2)

        self._px_max = max(self._px_max, px)
        self._py_max = max(self._py_max, py)
        self._qx_max = max(self._qx_max, qx)
        self._qy_max = max(self._qy_max, qy)
        self._width_max = max(self._width_max, width_max)
        self._height_max = max(self._height_max, height_max)

    def feed_element_sketches(self, sketches: Sequence[FrameElementSketch]) -> None:
        px = py = 0.0
        qx = qy = 0.0

        for sketch in sketches:
            left, top, right, bottom = sketch.image_box
            w = right - left
            h = bottom - top

            vertices = np.array(
                (
                    (0, 0, 1),
                    (0, h, 1),
                    (w, h, 1),
                    (w, 0, 1),
                )
            ).reshape(4, 3, 1)

            result = sketch.forward @ vertices
            xmin = np.min(result[..., 0, :])
            xmax = np.max(result[..., 0, :])
            ymin = np.min(result[..., 1, :])
            ymax = np.max(result[..., 1, :])

            px = max(px, -xmin)
            py = max(py, -ymin)
            qx = max(qx, xmax)
            qy = max(qy, ymax)

        self._px_max = max(self._px_max, px)
        self._py_max = max(self._py_max, py)
        self._qx_max = max(self._qx_max, qx)
        self._qy_max = max(self._qy_max, qy)


class ImageCompositor:
    def __init__(
        self,
        canvas_info: CanvasInfo,
        debug_outline: bool = False,
        resampling: Image.Resampling | None = None,
    ):
        if resampling is None:
            resampling = Image.Resampling.NEAREST

        # note that only a few resampling method is supported in Pillow's transform()
        # I assumed Pillow will raise errors in that case

        self.canvas_info = canvas_info
        self.debug_outline = debug_outline
        self.resampling = resampling

    def _new_canvas(self) -> Image.Image:
        return Image.new(
            "RGBA",
            self.canvas_info.size,
        )

    def _draw_frame_outline(
        self,
        im: Image.Image,
        frame: AnimFrame,
    ) -> None:
        """Draw the frame outline in the output canvas image"""
        draw = ImageDraw(im)

        # frame pivot and canvas pivot must be coincident
        draw.circle(self.canvas_info.pivot, 2, outline="green")

        fx, fy, fw, fh = frame.bounds
        left = fx - fw / 2
        right = fx + fw / 2
        upper = fy - fh / 2
        lower = fy + fh / 2

        # first, define the vertices
        vertices = np.array(
            (
                (left, upper, 1),
                (left, lower, 1),
                (right, lower, 1),
                (right, upper, 1),
            )
        ).reshape((4, 3, 1))

        vertices = self.canvas_info.transform @ vertices
        vertices = [tuple(row) for row in vertices[0:4, 0:2, 0].tolist()]

        draw.polygon(vertices, outline="green")

    def _draw_element_outline(
        self,
        im: Image.Image,
        sketch: FrameElementSketch,
    ) -> None:
        """the element layout in the element image"""

        ow, oh = im.size  # input
        iw, ih = sketch.size  # output

        # pivot sketch in image_box coordinate
        px, py = sketch.pivot
        px *= ow / iw
        py *= oh / ih

        draw = ImageDraw(im)
        draw.circle((px, py), 2, outline="red", width=2)
        draw.polygon(
            (
                (0, 0),
                (0, oh - 1),
                (ow - 1, oh - 1),
                (ow - 1, 0),
            ),
            outline="red",
        )

    def draw_frame(self, frame_sketch: FrameSketch) -> Image.Image:
        out = self._new_canvas()

        if self.debug_outline:
            self._draw_frame_outline(out, frame_sketch.frame)

        for sketch in frame_sketch.elements:
            forward = self.canvas_info.transform @ sketch.forward
            backward = np.linalg.inv(forward)
            transform = cast(
                tuple[float, float, float, float, float, float],
                backward.flat[0:6],
            )

            elem_im = sketch.image.crop(sketch.image_box)

            if self.debug_outline:
                self._draw_element_outline(elem_im, sketch)

            if abs(sketch.alpha - 1.0) > 1.0e-5:
                alpha = elem_im.getchannel("A").point(lambda s: s * sketch.alpha)
                elem_im.putalpha(alpha)

            elem_im = elem_im.transform(
                out.size,
                Image.Transform.AFFINE,
                transform,
                resample=self.resampling,
            )

            out.alpha_composite(elem_im)

        return out


class AnimationBuilder:
    """Facade of underlying objects"""

    def __init__(
        self,
        registry: BuildRegistry,
        frames: list[AnimFrame],
        *,
        scale: float | None = None,
        margins: int | None = None,
        resampling: Image.Resampling | None = None,
        measure_methods: Sequence[str] | None = None,
        debug_outline: bool = False,
    ):
        if measure_methods is None:
            measurements = {"frame"}
        else:
            measurements = set(measure_methods)

        if not measurements:
            raise ValueError("Empty measurements")

        if any(s not in {"frame", "element"} for s in measurements):
            raise ValueError(f"Bad measurements: {sorted(measurements)!r}")

        sketcher = ImageSketcher(registry)
        lm = LayoutManager(
            scale=scale,
            margins=margins,
        )

        frame_sketches = []
        element_sketches = []

        for frame in frames:
            frame_sketch = sketcher.sketch_frame(frame)

            frame_sketches.append(frame_sketch)
            element_sketches += frame_sketch.elements

        if "frame" in measurements:
            lm.feed_frames(frames)

        if "elements" in measurements:
            lm.feed_element_sketches(element_sketches)

        canvas_info = lm.update()
        compositor = ImageCompositor(
            canvas_info=canvas_info,
            debug_outline=debug_outline,
            resampling=resampling,
        )

        self._frame_sketches = frame_sketches
        self._compositor = compositor

    def draw_frame(self, index: int) -> Image.Image:
        if not 0 <= index < len(self._frame_sketches):
            raise ValueError(index)

        sketch = self._frame_sketches[index]
        return self._compositor.draw_frame(sketch)


def _pillow_save_params(format: str) -> dict[str, Any]:
    params = {}

    if format == "png":
        params["disposal"] = 1
    elif format == "gif":
        params["disposal"] = 2
    elif format == "webp":
        pass
    else:
        raise ValueError(format)
    return params


def save_animation(
    seq: list[Image.Image],
    *,
    format: str,
    frame_rate: float,
) -> bytes:
    format = format.lower()
    save_params = _pillow_save_params(format)

    first, rest = seq[0], seq[1:]

    with BytesIO() as bio:
        first.save(
            fp=bio,
            save_all=len(seq) >= 2,
            format=format,
            append_images=rest,
            duration=1000 / frame_rate,  # milliseconds
            **save_params,
        )

        return bio.getvalue()
