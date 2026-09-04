"""POSIX shared-memory round-trip from Python.

This is the mechanism the frame ring is built on: the C++ side creates and
writes a segment, Python maps the same name read-only and unpacks it. The
original version of this file was a demo script that ran at import time and
required a segment left behind by test_shm_writer -- as a test it failed on a
clean machine, so it now creates whatever it reads.

The C++ writer is still exercised, but only when its binary has been built.
"""
import mmap
import os
import struct
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SHM_DIR = Path("/dev/shm")


@pytest.fixture
def segment():
    """A private 4-byte segment, unlinked afterwards whatever happens."""
    name = f"pytest_shm_{os.getpid()}"
    path = SHM_DIR / name
    fd = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        os.ftruncate(fd, 4)
        yield name, fd
    finally:
        os.close(fd)
        path.unlink(missing_ok=True)


def test_write_then_read_roundtrip(segment):
    name, fd = segment
    with mmap.mmap(fd, 4) as mm:
        mm[:4] = struct.pack("<i", 513)

    rfd = os.open(str(SHM_DIR / name), os.O_RDONLY)
    try:
        with mmap.mmap(rfd, 4, prot=mmap.PROT_READ) as mm:
            assert struct.unpack("<i", mm[:4])[0] == 513
    finally:
        os.close(rfd)


def test_writes_are_visible_without_reopening(segment):
    """Shared mappings are the same physical pages -- a reader sees a writer's
    store with no flush, which is what lets the ring work at frame rate."""
    name, fd = segment
    rfd = os.open(str(SHM_DIR / name), os.O_RDONLY)
    try:
        with mmap.mmap(fd, 4) as w, mmap.mmap(rfd, 4, prot=mmap.PROT_READ) as r:
            for value in (1, 2, 99, -7):
                w[:4] = struct.pack("<i", value)
                assert struct.unpack("<i", r[:4])[0] == value
    finally:
        os.close(rfd)


def test_segment_is_owner_only(segment):
    """The pipeline's segment is created 0600, which is why every service has to
    run as the same user. Pin the assumption the systemd units depend on."""
    name, _ = segment
    mode = (SHM_DIR / name).stat().st_mode & 0o777
    assert mode == 0o600


def test_reading_a_missing_segment_raises():
    with pytest.raises(FileNotFoundError):
        os.open(str(SHM_DIR / "pytest_shm_definitely_absent"), os.O_RDONLY)


def test_cpp_writer_output_is_readable_from_python():
    """Cross-language check: run the C++ test_shm_writer binary and read what it
    wrote. Skipped when the binary hasn't been built."""
    binary = REPO_ROOT / "build" / "test_shm_writer"
    if not binary.exists():
        pytest.skip("build/test_shm_writer not built")

    path = SHM_DIR / "demo_int"
    path.unlink(missing_ok=True)
    subprocess.run([str(binary)], check=True, capture_output=True, timeout=30)
    try:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            with mmap.mmap(fd, 4, prot=mmap.PROT_READ) as mm:
                assert struct.unpack("<i", mm[:4])[0] == 513
        finally:
            os.close(fd)
    finally:
        path.unlink(missing_ok=True)
