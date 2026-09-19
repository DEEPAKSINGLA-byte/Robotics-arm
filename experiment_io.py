"""Small, model-independent file helpers shared by experiments."""
import hashlib
import json
from pathlib import Path


def file_hash(path):
    """SHA-256 of file bytes; accepts either a string or a Path."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value, *, trailing_newline=False):
    """Replace a completed JSON file, never leave a partially written result.

    Reject non-finite numbers before touching the destination. The caller owns
    directory creation; concurrent writers to the same path are not supported.
    """
    path = Path(path)
    text = json.dumps(value, indent=2, allow_nan=False)
    if trailing_newline:
        text += '\n'
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(text)
    temporary.replace(path)
