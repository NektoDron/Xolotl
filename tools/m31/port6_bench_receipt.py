"""Link a frozen port6 ELF to an explicit build/source receipt."""
import json
from pathlib import Path

from port5_bench_prepare import fingerprint


def verify_receipt(path: Path, elf_sha256: str, binary: str):
    receipt = json.loads(path.read_text())
    if receipt.get('elf', {}).get(binary) != elf_sha256:
        raise ValueError(f'build receipt does not identify selected {binary} ELF SHA256')
    source = receipt.get('source')
    if not isinstance(source, dict) or not source or any(not isinstance(value, str) or len(value) != 64 for value in source.values()):
        raise ValueError('build receipt must include source-file SHA256 mapping')
    return {**fingerprint(path), 'verified_binary': binary, 'verified_elf_sha256': elf_sha256}
