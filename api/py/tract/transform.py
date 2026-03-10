import json
from typing import Dict, Optional, Union


class TransformSpec:
    """Base class for typed transform specifications.

    Subclasses represent specific transforms with typed parameters.
    Can be passed directly to :meth:`Model.transform`.
    """

    def to_json(self) -> str:
        """Serialize this transform spec to the JSON string the FFI layer expects."""
        raise NotImplementedError


class ConcretizeSymbols(TransformSpec):
    """Replace symbolic dimensions with concrete integer values.

    Example::

        model.transform(ConcretizeSymbols({"B": 1}))
        # or with builder pattern:
        model.transform(ConcretizeSymbols().value("B", 1))
    """

    def __init__(self, values: Optional[Dict[str, int]] = None):
        self._values: Dict[str, int] = dict(values) if values else {}

    def value(self, symbol: str, val: int) -> "ConcretizeSymbols":
        """Set a symbol to a concrete value. Returns self for chaining."""
        self._values[symbol] = val
        return self

    def to_json(self) -> str:
        return json.dumps({"name": "concretize_symbols", "values": self._values})


class Pulse(TransformSpec):
    """Convert a model to a pulsed (streaming) model.

    Example::

        model.transform(Pulse("5", symbol="B"))
        # or with builder pattern:
        model.transform(Pulse("5").symbol("B"))
    """

    def __init__(self, pulse: Union[str, int], *, symbol: Optional[str] = None):
        self._pulse = str(pulse)
        self._symbol = symbol

    def symbol(self, symbol: str) -> "Pulse":
        """Set the symbol to pulse over. Returns self for chaining."""
        self._symbol = symbol
        return self

    def to_json(self) -> str:
        d = {"name": "pulse", "pulse": self._pulse}
        if self._symbol is not None:
            d["symbol"] = self._symbol
        return json.dumps(d)
