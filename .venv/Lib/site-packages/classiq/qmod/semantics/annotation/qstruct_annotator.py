from classiq.interface.exceptions import ClassiqExpansionError
from classiq.interface.generator.functions.type_name import TypeName
from classiq.interface.generator.types.struct_declaration import StructDeclaration
from classiq.interface.model.model_visitor import ModelVisitor

from classiq.qmod.builtins.enums import BUILTIN_ENUM_DECLARATIONS
from classiq.qmod.builtins.structs import BUILTIN_STRUCT_DECLARATIONS
from classiq.qmod.model_state_container import QMODULE

IGNORED_MISSING_TYPES = {"MoleculeResult"}


class QStructAnnotator(ModelVisitor):
    def __init__(self) -> None:
        self._visited: set[int] = set()
        self._defined_type_names = (
            set(QMODULE.enum_decls)
            | set(QMODULE.type_decls)
            | set(QMODULE.qstruct_decls)
            | set(BUILTIN_STRUCT_DECLARATIONS)
            | set(BUILTIN_ENUM_DECLARATIONS)
            | IGNORED_MISSING_TYPES
        )

    def visit_TypeName(self, type_name: TypeName) -> None:
        if type_name.name not in self._defined_type_names:
            raise ClassiqExpansionError(f"Undefined type {type_name.name}")
        self._annotate_quantum_struct(type_name)
        self._annotate_classical_struct(type_name)

    def _annotate_quantum_struct(self, type_name: TypeName) -> None:
        if (
            type_name.has_classical_struct_decl
            or type_name.has_fields
            or id(type_name) in self._visited
        ):
            return
        decl = QMODULE.qstruct_decls.get(type_name.name)
        if decl is None:
            return
        self._visited.add(id(type_name))
        new_fields = {
            field_name: field_type.model_copy()
            for field_name, field_type in decl.fields.items()
        }
        # We first visit the new fields and then set to deal with recursive
        # qstructs
        self.visit(new_fields)
        type_name.set_fields(new_fields)

    def _annotate_classical_struct(self, type_name: TypeName) -> None:
        if (
            type_name.has_classical_struct_decl
            or type_name.has_fields
            or id(type_name) in self._visited
        ):
            return
        decl = QMODULE.type_decls.get(type_name.name)
        if decl is None:
            return
        self._visited.add(id(type_name))
        new_fields = {
            field_name: field_type.model_copy(deep=True)
            for field_name, field_type in decl.variables.items()
        }
        self.visit(new_fields)
        type_name.set_classical_struct_decl(
            StructDeclaration(name=decl.name, variables=new_fields)
        )
