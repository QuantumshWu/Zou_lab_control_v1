"""Print a sha256 over the given files' contents -- the build-cache key for
build_and_program.bat.  If the hash matches the one stored next to the bitstream, the
sources that affect the bitstream are unchanged, so the (slow) synth+impl can be skipped
and the existing .bit programmed directly.

Usage: python src_hash.py FILE [FILE ...]   (missing files are skipped)
"""
import hashlib
import os
import sys

h = hashlib.sha256()
for p in sys.argv[1:]:
    if p and os.path.exists(p):
        # include the (normalized) path so a file appearing / disappearing / being renamed
        # changes the key even if some other file's bytes happen to compensate.
        h.update(p.replace("\\", "/").encode("utf-8"))
        with open(p, "rb") as f:
            h.update(f.read())
print(h.hexdigest())
