from typing import TYPE_CHECKING

from classiq.interface.model.handle_binding import HandleBinding
from classiq.interface.model.quantum_statement import QuantumOperation

from classiq.model_expansions.quantum_operations.emitter import Emitter

if TYPE_CHECKING:
    from classiq.model_expansions.interpreters.base_interpreter import BaseInterpreter


class HandleEvaluator(Emitter[QuantumOperation]):
    def __init__(self, interpreter: "BaseInterpreter", handle_name: str) -> None:
        super().__init__(interpreter)
        self._handle_name = handle_name

    def emit(self, op: QuantumOperation, /) -> bool:
        handle = getattr(op, self._handle_name)
        if not isinstance(handle, HandleBinding):
            return False
        evaluated_handle = self._interpreter.evaluate(handle).value.handle.collapse()
        setattr(op, self._handle_name, evaluated_handle)
        return False
