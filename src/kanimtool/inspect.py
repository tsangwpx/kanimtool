import builtins
from collections import Counter, defaultdict

from kanimtool.parser import AnimGroup, Build


def _indentation(n: int) -> str:
    return " " * n


_IND1 = _indentation(1)
_IND3 = _indentation(3)
_IND5 = _indentation(5)
_IND7 = _indentation(7)


def print_build(
    build: Build,
    print_frames: bool = False,
    *,
    print=builtins.print,
) -> None:
    print("build", build.name, "symbols", len(build.symbols))

    for symbol in build.symbols:
        symbol_name = build.get_symbol_name(symbol.hash)

        print(
            _IND1,
            "symbol",
            symbol_name,
            symbol.hash,
            "frames",
            len(symbol.frames),
            "sequences",
            len(symbol.sequence),
        )

        if print_frames:
            for frame_idx, frame in enumerate(symbol.frames):
                print(_IND3, frame)


def _format_ids(seq: list[int]) -> str:
    parts = []

    idx = 0
    size = len(seq)

    while idx < size:
        start = idx
        num = seq[idx]
        length = 1

        while start + length < size and seq[start + length] == num + length:
            length += 1

        idx = start + length

        if length == 1:
            parts.append(f"{start}")
        else:
            parts.append(f"{start}:{start + length}")

    return ",".join(parts)


def print_anim(
    anim_group: AnimGroup,
    print_frames: bool = False,
    print_symbols: bool = False,
    *,
    print=builtins.print,
) -> None:
    table = anim_group.strings

    print("animations", len(anim_group.animations))
    for animation in anim_group.animations:
        print(
            _IND1,
            "anim",
            animation.name,
            "frames",
            len(animation.frames),
            "rate",
            animation.frame_rate,
        )

        if print_symbols:
            symbol_frames: dict[int, Counter[int]] = defaultdict(Counter)

            for frame_idx, frame in enumerate(animation.frames):
                for elem in frame.elements:
                    symbol_id = elem.symbol
                    symbol_frames[symbol_id][frame_idx] += 1

            for symbol_id, frame_counter in symbol_frames.items():
                symbol_name = table.get(symbol_id)
                print(
                    _IND3,
                    "symbol",
                    symbol_name,
                    symbol_id,
                    "frame",
                    _format_ids(sorted(frame_counter.keys())),
                )

        if print_frames:
            for frame_idx, frame in enumerate(animation.frames):
                print(
                    _IND3,
                    "frame",
                    frame_idx,
                    "bounds",
                    frame.bounds,
                    "elements",
                    len(frame.elements),
                )

                for elem in frame.elements:
                    symbol_id = elem.symbol
                    symbol_name = table.get(symbol_id)
                    alpha = elem.mult_alpha[0] * elem.mult_alpha[1]
                    print(
                        _IND5,
                        "elem",
                        symbol_name,
                        symbol_id,
                        "frame",
                        elem.frame,
                        "alpha",
                        alpha,
                        "transform",
                        elem.transform,
                    )
