import dataclasses
from dataclasses import dataclass
from typing import cast

from kanimtool.io import DataIO


def _read_kstring(dio: DataIO) -> str | None:
    """
    kstring is a string encoded in 32bit length followed by its bytes in utf-8
    """
    length = dio.read_i32()
    if length < 0:
        return None

    return dio.readexactly(length).decode("utf-8")


def _read_hash_strings(dio: DataIO) -> list[tuple[int, str]]:
    count = dio.read_i32()

    items = []
    for _ in range(count):
        h = dio.read_i32()
        v = _read_kstring(dio)
        if v is None:
            raise ValueError("Null string")
        items.append((h, v))

    return items


def _sdbm_lower(s: str) -> int:
    """
    Compute lowercased SDBM hash
    """
    h = 0
    mask = 0xFFFF_FFFF
    for ch in s.lower():
        h = (ord(ch) + (h << 6) + (h << 16) - h) & mask

    # https://stackoverflow.com/a/37095855/1692260
    sign = 0x8000_0000
    return (h ^ sign) - sign


@dataclass
class SymbolFrame:
    seq_idx: int
    """starting offset"""

    duration: int

    build_image_idx: int

    bounds: tuple[float, float, float, float]
    """
    (x, y, width, height)
    
    (x, y) is the center of the bounding box relative to the pivot
    (width, height) is the size of the bounding box
    """

    uvbox: tuple[float, float, float, float]
    """
    (u1, v1, u2, v2)
    
    UV-coordinate
    
    (u1, v1) top left corner in the texture  
    (u2, v2) bottom right corner in in the texture
    """

    def compute_box(self, size: tuple[int, int]) -> tuple[int, int, int, int]:
        """Get the bounding box in (left, upper, right, lower) convention"""

        w, h = size
        if w < 0 or h < 0:
            raise ValueError(f"Bad dimension: {(w, h)!r}")

        u1, v1, u2, v2 = self.uvbox

        width = int(w * abs(u2 - u1))
        height = int(h * abs(v2 - v1))

        left = int(w * u1)
        upper = int(h * v1)

        if left < 0 or upper < 0 or left + width > w or upper + height > h:
            raise ValueError(f"Bad diemension: {self.uvbox!r} {size!r}")

        return left, upper, left + width, upper + height


@dataclasses.dataclass
class Symbol:
    hash: int
    path_id: int
    flags: int
    frames: list[SymbolFrame]

    sequence: list[SymbolFrame] = dataclasses.field(
        default_factory=list,
        repr=False,
        init=False,
    )

    def __post_init__(self):
        # symbol frame has seq_idx and duration to form a sequence
        seq_len = 0

        for frame in self.frames:
            seq_len = max(seq_len, frame.seq_idx + frame.duration)

        seq = [None] * seq_len

        for frame in self.frames:
            assert frame.duration >= 1, frame
            start = frame.seq_idx
            stop = start + frame.duration
            seq[start:stop] = (frame,) * frame.duration

        # According to stone hatch,
        # missing frame is replaced by next available frame
        # seq_idx beyond the list imply the last valid frame

        # filling missing frame from the back
        # the last must be non-None
        for index in range(seq_len)[::-1]:
            if seq[index] is None:
                seq[index] = seq[index + 1]

        self.sequence = cast(list[SymbolFrame], seq)

    def get_sequential_frame(self, index: int) -> SymbolFrame:
        assert index >= 0, index

        if index >= len(self.sequence):
            # return the last frame
            return self.sequence[-1]
        else:
            return self.sequence[index]


@dataclasses.dataclass
class Build:
    name: str
    symbols: list[Symbol]
    strings: dict[int, str]

    _symbol_index: dict[str | int, Symbol] = dataclasses.field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def __post_init__(self):
        self.rebuild_index()

    def rebuild_index(self) -> None:
        self._symbol_index = {}

        for sym in self.symbols:
            self._symbol_index[sym.hash] = sym

            try:
                name = self.strings[sym.hash]
            except KeyError:
                print(f"Missing name for symbol {sym.hash!r}")
            else:
                self._symbol_index[name] = sym

    def get_symbol(self, key: str | int) -> Symbol:
        try:
            return self._symbol_index[key]
        except KeyError:
            raise LookupError(key) from None

    def get_symbol_name(self, key: str | int) -> str:
        if isinstance(key, str) and key in self._symbol_index:
            return key
        elif key in self.strings:
            return self.strings[key]  # type: ignore[index]
        else:
            raise LookupError(key) from None


@dataclasses.dataclass
class FrameElement:
    symbol: int
    frame: int
    mult_alpha: tuple[float, float, float, float]
    transform: tuple[float, float, float, float, float, float]


@dataclasses.dataclass
class AnimFrame:
    bbox: tuple[float, float, float, float]
    elements: list[FrameElement]

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return self.bbox


@dataclasses.dataclass
class Animation:
    name: str
    frame_rate: float
    frames: list[AnimFrame]


@dataclasses.dataclass
class AnimGroup:
    animations: list[Animation]
    strings: dict[int, str]

    _anim_index: dict[str, Animation] = dataclasses.field(
        default_factory=dict,
        repr=False,
        init=False,
    )

    def __post_init__(self):
        self.rebuild_index()

    def rebuild_index(self) -> None:
        self._anim_index = {}

        for anim in self.animations:
            self._anim_index[anim.name] = anim

    def get_animation(self, name: str) -> Animation:
        try:
            return self._anim_index[name]
        except KeyError:
            raise LookupError(name) from None


class Parser:
    def __init__(
        self,
        strings: dict[int, str] | None = None,
    ):
        if strings is None:
            strings = {}

        self.strings: dict[int, str] = strings

    def parse_build(self, buf: bytes | bytearray | memoryview) -> Build:
        dio = DataIO(buf)

        if dio.readexactly(4) != b"BILD":
            raise ValueError("Bad signature")

        version = dio.read_i32()
        if version not in (9, 10):
            raise ValueError(f"Bad version {version}")

        symbol_count = dio.read_i32()
        _unused = dio.read_i32()

        name = _read_kstring(dio)
        if name is None:
            raise ValueError("Build has no name")

        symbols = []

        for symbol_idx in range(symbol_count):
            symbol_hash = dio.read_i32()

            path_hash = 0
            if version >= 10:
                path_hash = dio.read_i32()
            _symbol_channels = dio.read_i32()
            symbol_flags = dio.read_i32()
            symbol_frame_count = dio.read_i32()

            frames = []

            for frame_idx in range(symbol_frame_count):
                frame_seq_id = dio.read_i32()
                frame_duration = dio.read_i32()
                frame_build_image_idx = dio.read_i32()

                bounds = dio.read_struct("ffff")
                uvbox = dio.read_struct("ffff")

                frame = SymbolFrame(
                    seq_idx=frame_seq_id,
                    duration=frame_duration,
                    build_image_idx=frame_build_image_idx,
                    bounds=cast(tuple[float, float, float, float], bounds),
                    uvbox=cast(tuple[float, float, float, float], uvbox),
                )
                frames.append(frame)

            symbol = Symbol(
                hash=symbol_hash,
                path_id=path_hash,
                flags=symbol_flags,
                frames=frames,
            )
            symbols.append(symbol)

        hash_strings = _read_hash_strings(dio)
        self.strings.update(hash_strings)

        return Build(
            name=name,
            symbols=symbols,
            strings=self.strings,
        )

    def parse_anim(self, buf: bytes | bytearray | memoryview) -> AnimGroup:
        dio = DataIO(buf)

        if dio.readexactly(4) != b"ANIM":
            raise ValueError("Bad signature")

        version = dio.read_i32()
        if version not in (5,):
            raise ValueError(f"Bad version: {version}")

        _unused1 = dio.read_i32()
        _unused2 = dio.read_i32()

        anim_count = dio.read_i32()

        animations = []

        for anim_idx in range(anim_count):
            anim_name = _read_kstring(dio)

            if anim_name is None:
                raise ValueError(f"Animation {anim_idx} has no name")

            _unused3 = dio.read_i32()  # root symbol hash
            anim_frame_rate = dio.read_f32()
            anim_frame_count = dio.read_i32()

            frames = []

            for frame_idx in range(anim_frame_count):
                bbox = dio.read_struct("ffff")

                element_count = dio.read_i32()

                elements = []

                for elem_idx in range(element_count):
                    elem_symbol = dio.read_i32()
                    elem_frame = dio.read_i32()
                    _unused4 = dio.read_i32()
                    _unused5 = dio.read_i32()

                    c0, c1, c2, c3 = dio.read_struct("ffff")
                    mult_alpha = (c0, c1, c2, c3)

                    # the transform matrix is in column major style
                    # here the variable naming are in row major style
                    m00, m10, m01, m11, m02, m12 = dio.read_struct("ffffff")
                    transform = (m00, m01, m02, m10, m11, m12)

                    _unused6 = dio.read_f32()

                    elem = FrameElement(
                        symbol=elem_symbol,
                        mult_alpha=mult_alpha,
                        frame=elem_frame,
                        transform=transform,
                    )
                    elements.append(elem)

                frame = AnimFrame(
                    bbox=cast(tuple[float, float, float, float], bbox),
                    elements=elements,
                )
                frames.append(frame)

            anim = Animation(
                name=anim_name,
                frame_rate=anim_frame_rate,
                frames=frames,
            )
            animations.append(anim)

        _unused7 = dio.read_i32()

        hash_strings = _read_hash_strings(dio)
        self.strings.update(hash_strings)

        return AnimGroup(
            animations=animations,
            strings=self.strings,
        )
