import abc
import contextlib
import io
import struct
from io import BytesIO
from struct import pack as _pack
from struct import unpack as _unpack
from typing import Any, BinaryIO, Iterator, TypeVar, Union

Stream = TypeVar("Stream", bound=io.IOBase)


@contextlib.contextmanager
def revert_position(
    stream: Stream,
    *,
    offset: int | None = None,
    whence: int = io.SEEK_SET,
) -> Iterator[Stream]:
    pos = stream.tell()
    if offset is not None:
        stream.seek(offset, whence)

    try:
        yield stream
    finally:
        stream.seek(pos, io.SEEK_SET)


class DataReaderMixin(io.IOBase, metaclass=abc.ABCMeta):
    """Mixin class for reading structured data"""

    def readexactly(self, n: int) -> bytes:
        """Read exactly `n` bytes"""

        data = self.read(n)
        if 0 <= n != len(data):
            raise EOFError(
                f"{n} bytes were expected but {len(data)} bytes were returned"
            )
        return data

    def read_u8(self) -> int:
        return _unpack("B", self.readexactly(1))[0]

    def read_u16(self, byteorder: str = "") -> int:
        return _unpack("H", self.readexactly(2))[0]

    def read_u32(self, byteorder: str = "") -> int:
        return _unpack("I", self.readexactly(4))[0]

    def read_u64(self, byteorder: str = "") -> int:
        return _unpack("Q", self.readexactly(8))[0]

    def read_i8(self, byteorder: str = "") -> int:
        return _unpack("b", self.readexactly(1))[0]

    def read_i16(self, byteorder: str = "") -> int:
        return _unpack("h", self.readexactly(2))[0]

    def read_i32(self, byteorder: str = "") -> int:
        return _unpack("i", self.readexactly(4))[0]

    def read_i64(self, byteorder: str = "") -> int:
        return _unpack("q", self.readexactly(8))[0]

    def read_f32(self, byteorder: str = "") -> float:
        return _unpack("f", self.readexactly(4))[0]

    def read_f64(self, byteorder: str = "") -> float:
        return _unpack("d", self.readexactly(8))[0]

    def peek(self, n: int) -> bytes:
        """A default implementation of peek"""

        with revert_position(self):
            return self.read(n)

    def peekexactly(self, n: int) -> bytes:
        """Peek exactly `n` bytes"""
        data = self.peek(n)

        if len(data) != n:
            raise EOFError(
                f"{n} bytes were expected but {len(data)} bytes were returned"
            )

        return data

    def read_cstr(self, n: int = -1, raise_eof: bool = False) -> bytes:
        """Read bytes until `n` is read or reach `NUL` character or EOF"""
        buf = bytearray()

        while n < 0 or len(buf) < n:
            byte = self.read(1)
            if not byte:
                if raise_eof:
                    raise EOFError

                break

            buf += byte

            if byte[0] == 0:
                break

        return buf

    def read_struct(self, fmt: str | bytes | struct.Struct) -> tuple[Any, ...]:
        if isinstance(fmt, (str, bytes)):
            size = struct.calcsize(fmt)
            return struct.unpack(fmt, self.readexactly(size))
        elif isinstance(fmt, struct.Struct):
            return fmt.unpack(self.readexactly(fmt.size))
        else:
            raise TypeError(
                f"'fmt' expects a str, a bytes or a Struct object but {type(fmt)!r} was given"
            )

    def skip(self, n: int) -> None:
        self.seek(n, io.SEEK_CUR)


class DataWriterMixin(io.IOBase, metaclass=abc.ABCMeta):
    """Mixin class for writing structured data"""

    def writeexactly(self, buf: bytes) -> int:
        n = self.write(buf)

        if n != len(buf):
            raise IOError(f"{n} bytes were written but {len(buf)} bytes were expected")

        return n

    def write_u8(self, n: int, *, byteorder: str = "") -> int:
        return self.writeexactly(_pack("%sB", n))

    def write_u16(self, n: int, *, byteorder: str = "") -> int:
        return self.writeexactly(_pack("H", n))

    def write_u32(self, n: int, *, byteorder: str = "") -> int:
        return self.writeexactly(_pack("I", n))

    def write_u64(self, n: int, *, byteorder: str = "") -> int:
        return self.writeexactly(_pack("Q", n))

    def write_i8(self, n: int, *, byteorder: str = "") -> int:
        return self.writeexactly(_pack("b", n))

    def write_i16(self, n: int, *, byteorder: str = "") -> int:
        return self.writeexactly(_pack("h", n))

    def write_i32(self, n: int, *, byteorder: str = "") -> int:
        return self.writeexactly(_pack("%si", n))

    def write_i64(self, n: int, *, byteorder: str = "") -> int:
        return self.writeexactly(_pack("q", n))

    def write_struct(self, fmt: Union[bytes, str, struct.Struct], *args: Any) -> int:
        if isinstance(fmt, (str, bytes)):
            buf = struct.pack(fmt, *args)
        elif isinstance(fmt, struct.Struct):
            buf = fmt.pack(*args)
        else:
            raise TypeError(
                f"'fmt' expects str, bytes, or Struct object but {type(fmt)!r} given"
            )

        return self.writeexactly(buf)


class _StreamWrapper(io.BufferedIOBase):
    raw: BinaryIO | None

    def __init__(self, raw: BinaryIO):
        self.raw = raw

    def seek(self, pos: int, whence: int = io.SEEK_SET) -> int:
        return self.raw.seek(pos, whence)

    def tell(self) -> int:
        return self.raw.tell()

    def detach(self) -> BinaryIO:
        if self.raw is None:
            raise ValueError("underlying stream already detached")

        self.flush()
        raw = self.raw
        self.raw = None
        return raw

    def write(self, buf: bytes | bytearray | memoryview) -> int:
        return self.raw.write(buf)

    def read(self, n: int | None = -1) -> bytes:
        return self.raw.read(n)

    def read1(self, n: int | None = -1) -> bytes:
        return self.raw.read(n)

    def close(self) -> None:
        self.raw.close()

    def fileno(self) -> int:
        return self.raw.fileno()

    def flush(self) -> None:
        self.raw.flush()

    def isatty(self) -> bool:
        return self.raw.isatty()

    def readable(self) -> bool:
        return self.raw.readable()

    def seekable(self) -> bool:
        return self.raw.seekable()

    def truncate(self, pos: int | None = None) -> int:
        return self.raw.truncate(pos)

    def writable(self) -> bool:
        return self.raw.writable()


class DataIO(_StreamWrapper, io.BufferedIOBase, DataReaderMixin, DataWriterMixin):
    def __init__(
        self,
        source: Union[BinaryIO, BytesIO, bytes, bytearray, memoryview],
    ):
        if isinstance(source, (bytes, bytearray, memoryview)):
            source = BytesIO(source)

        super().__init__(source)
