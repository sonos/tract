import re
from pathlib import Path

import tract
from tract.bindings import lib


def test_every_called_symbol_is_exported():
    package = Path(tract.__file__).parent
    called = set()
    for source in package.glob("*.py"):
        called.update(re.findall(r"\blib\.(tract_\w+)", source.read_text()))
    assert called
    missing = sorted(name for name in called if not hasattr(lib, name))
    assert missing == []
