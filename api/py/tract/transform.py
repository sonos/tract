import json
from abc import ABC, abstractmethod
from typing import Dict, Optional, Union

from .tensor import DatumType


class TransformSpec(ABC):
    """Base class for typed transform specifications.

    Subclasses represent specific transforms with typed parameters.
    Can be passed directly to :meth:`Model.transform`.
    """

    @abstractmethod
    def to_json(self) -> str:
        """Serialize this transform spec to the JSON string the FFI layer expects."""
        ...


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
        # with include/exclude:
        model.transform(FloatPrecision(DatumType.F32, DatumType.F16, exclude=["layer.1"]))
    """

    _DT_NAMES = {
        DatumType.F16: "f16",
        DatumType.F32: "f32",
        DatumType.F64: "f64",
    }

    def __init__(
        self,
        from_dt: DatumType,
        to_dt: DatumType,
        *,
        include: Optional[list] = None,
        exclude: Optional[list] = None,
    ):
        if from_dt not in self._DT_NAMES:
            raise ValueError(f"from_dt must be a float DatumType, got {from_dt}")
        if to_dt not in self._DT_NAMES:
            raise ValueError(f"to_dt must be a float DatumType, got {to_dt}")
        self._from = from_dt
        self._to = to_dt
        self._include = list(include) if include else None
        self._exclude = list(exclude) if exclude else None

    def include(self, patterns: list) -> "FloatPrecision":
        """Set include patterns — only matching nodes are translated. Returns self for chaining."""
        self._include = list(patterns)
        return self

    def exclude(self, patterns: list) -> "FloatPrecision":
        """Set exclude patterns — matching nodes are excluded. Returns self for chaining."""
        self._exclude = list(patterns)
        return self

    def to_json(self) -> str:
        d = {
            "name": "float_precision",
            "from": self._DT_NAMES[self._from],
            "to": self._DT_NAMES[self._to],
        }
        if self._include is not None:
            d["include"] = self._include
        if self._exclude is not None:
            d["exclude"] = self._exclude
        return json.dumps(d)
