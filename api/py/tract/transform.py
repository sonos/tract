import json
from typing import Dict, Optional, Union

from .value import DatumType


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


class FloatPrecision(TransformSpec):
    """Change the float precision of a model (e.g. F32 to F16).

    Example::

        model.transform(FloatPrecision(DatumType.F32, DatumType.F16))
        # or with filter:
        model.transform(FloatPrecision(DatumType.F32, DatumType.F16, filter="!=layer.1"))
    """

    _DT_NAMES = {
        DatumType.F16: "f16",
        DatumType.F32: "f32",
        DatumType.F64: "f64",
    }

    def __init__(self, from_dt: DatumType, to_dt: DatumType, *, filter: Optional[str] = None):
        if from_dt not in self._DT_NAMES:
            raise ValueError(f"from_dt must be a float DatumType, got {from_dt}")
        if to_dt not in self._DT_NAMES:
            raise ValueError(f"to_dt must be a float DatumType, got {to_dt}")
        self._from = from_dt
        self._to = to_dt
        self._filter = filter

    def filter(self, pattern: str) -> "FloatPrecision":
        """Set a node-name filter pattern. Returns self for chaining."""
        self._filter = pattern
        return self

    def to_json(self) -> str:
        d = {
            "name": "float_precision",
            "from": self._DT_NAMES[self._from],
            "to": self._DT_NAMES[self._to],
        }
        if self._filter is not None:
            d["filter"] = self._filter
        return json.dumps(d)
