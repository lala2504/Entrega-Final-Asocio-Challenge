# We'll create a small module `heur_instance_io.py` with functions to load and validate
# the JSON instances and to build some convenient reverse indices and summaries.
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Dict, List, Set, Tuple, Union, IO, Optional
import json
import pandas as pd  # type: ignore

# ------------------------------
# Data model
# ------------------------------

@dataclass
class ProblemInstance:
    
    # ------------------------------
    
    # ATTRIBUTES
    employees: List[str]
    desks: List[str]
    days: List[str]
    groups: List[str]
    zones: List[str]
    desks_by_zone: Dict[str, List[str]]
    desks_by_employee: Dict[str, List[str]]
    employees_by_group: Dict[str, List[str]]
    days_by_employee: Dict[str, List[str]]

    # Convenience reverse indices (populated by build_reverse_indices)
    zone_of_desk: Dict[str, str] = field(default_factory=dict)
    group_of_employee: Dict[str, str] = field(default_factory=dict)

    # ------------------------------
    
    # METHODS

    # Lite summary cache
    _summary: Optional[Dict[str, int]] = field(default=None, init=False, repr=False)

    def summary(self) -> Dict[str, int]:
        """Return basic counts for a quick glance."""
        if self._summary is None:
            self._summary = {
                "n_employees": len(self.employees),
                "n_desks": len(self.desks),
                "n_days": len(self.days),
                "n_groups": len(self.groups),
                "n_zones": len(self.zones),
            }
        return self._summary
    
    # Print Attributes
    def attribute_names(self, include_private: bool = False, with_types: bool = False) -> list[str]:
        """
        Retorna una lista con los nombres (y opcionalmente tipos) de los atributos
        definidos como campos del dataclass.
        - include_private: incluye atributos que empiezan por '_' (p.ej. _summary)
        - with_types: devuelve 'nombre: tipo' en lugar de solo el nombre
        """
        names = []
        for f in fields(self):
            if not include_private and f.name.startswith('_'):
                continue
            if with_types:
                names.append(f"{f.name}: {f.type}")
            else:
                names.append(f.name)
        return names

    def print_attributes(self, include_private: bool = False, with_types: bool = False) -> None:
        """Imprime la lista de atributos (opcionalmente con tipos)."""
        for i, name in enumerate(self.attribute_names(include_private, with_types), 1):
            print(f"{i}. {name}")


# ------------------------------
# Load & validate
# ------------------------------

REQUIRED_KEYS = [
    "Employees",
    "Desks",
    "Days",
    "Groups",
    "Zones",
    "Desks_Z",
    "Desks_E",
    "Employees_G",
    "Days_E",
]

def _read_json(source: Union[str, Path, IO[str], dict]) -> dict:
    if isinstance(source, dict):
        return source
    if hasattr(source, "read"):
        return json.load(source)  # type: ignore[arg-type]
    p = Path(source)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _unique(seq: List[str]) -> bool:
    return len(seq) == len(set(seq))


def validate_instance_dict(data: dict) -> List[str]:
    """Return a list of validation error messages (empty if valid)."""
    errors: List[str] = []

    # Keys
    for k in REQUIRED_KEYS:
        if k not in data:
            errors.append(f"Missing key: {k}")

    if errors:
        return errors

    # Basic types
    def _expect_list_of_str(key: str):
        if not isinstance(data[key], list) or not all(isinstance(x, str) for x in data[key]):
            errors.append(f"Key '{key}' must be a list[str].")

    for key in ["Employees", "Desks", "Days", "Groups", "Zones"]:
        _expect_list_of_str(key)

    # Uniqueness
    for key in ["Employees", "Desks", "Days", "Groups", "Zones"]:
        if not _unique(data[key]):
            errors.append(f"Values under '{key}' must be unique.")

    # Dicts
    def _expect_dict_of_list_str(key: str):
        if not isinstance(data[key], dict):
            errors.append(f"Key '{key}' must be a dict.")
            return
        for k, v in data[key].items():
            if not isinstance(k, str) or not isinstance(v, list) or not all(isinstance(x, str) for x in v):
                errors.append(f"Key '{key}' must map str -> list[str].")
                break

    for key in ["Desks_Z", "Desks_E", "Employees_G", "Days_E"]:
        _expect_dict_of_list_str(key)

    if errors:
        return errors

    employees: Set[str] = set(data["Employees"])
    desks: Set[str] = set(data["Desks"])
    days: Set[str] = set(data["Days"])
    groups: Set[str] = set(data["Groups"])
    zones: Set[str] = set(data["Zones"])

    # Desks_Z: zones exist; desks exist; each desk appears in at most one zone
    seen_desks: Set[str] = set()
    for z, ds in data["Desks_Z"].items():
        if z not in zones:
            errors.append(f"Zone '{z}' in Desks_Z not declared in Zones.")
        for d in ds:
            if d not in desks:
                errors.append(f"Desk '{d}' in Desks_Z[{z}] not declared in Desks.")
            if d in seen_desks:
                errors.append(f"Desk '{d}' appears in multiple zones.")
            seen_desks.add(d)

    # Desks_E: employees exist; desks exist
    for e, ds in data["Desks_E"].items():
        if e not in employees:
            errors.append(f"Employee '{e}' in Desks_E not declared in Employees.")
        for d in ds:
            if d not in desks:
                errors.append(f"Desk '{d}' in Desks_E[{e}] not declared in Desks.")

    # Employees_G: groups exist; employees exist; employees unique across groups
    seen_emp_in_groups: Set[str] = set()
    for g, es in data["Employees_G"].items():
        if g not in groups:
            errors.append(f"Group '{g}' in Employees_G not declared in Groups.")
        for e in es:
            if e not in employees:
                errors.append(f"Employee '{e}' in Employees_G[{g}] not declared in Employees.")
            if e in seen_emp_in_groups:
                errors.append(f"Employee '{e}' appears in multiple groups.")
            seen_emp_in_groups.add(e)

    # Days_E: employees exist; days exist
    for e, ds in data["Days_E"].items():
        if e not in employees:
            errors.append(f"Employee '{e}' in Days_E not declared in Employees.")
        for d in ds:
            if d not in days:
                errors.append(f"Day '{d}' in Days_E[{e}] not declared in Days.")

    return errors


def build_reverse_indices(inst: ProblemInstance) -> None:
    """Populate zone_of_desk and group_of_employee in place."""
    # zone_of_desk
    zone_of_desk: Dict[str, str] = {}
    for z, ds in inst.desks_by_zone.items():
        for d in ds:
            zone_of_desk[d] = z
    inst.zone_of_desk = zone_of_desk

    # group_of_employee (assume each employee in exactly one group)
    group_of_employee: Dict[str, str] = {}
    for g, es in inst.employees_by_group.items():
        for e in es:
            group_of_employee[e] = g
    inst.group_of_employee = group_of_employee


def load_instance(source: Union[str, Path, IO[str], dict], *, strict: bool = True) -> ProblemInstance:
    """
    Load a problem instance from a path, file-like, or dict.
    If strict=True, raise ValueError on validation errors; else return best-effort.
    """
    data = _read_json(source)
    errors = validate_instance_dict(data)
    if errors and strict:
        raise ValueError("Invalid instance:\n- " + "\n- ".join(errors))

    inst = ProblemInstance(
        employees=list(data.get("Employees", [])),
        desks=list(data.get("Desks", [])),
        days=list(data.get("Days", [])),
        groups=list(data.get("Groups", [])),
        zones=list(data.get("Zones", [])),
        desks_by_zone=dict(data.get("Desks_Z", {})),
        desks_by_employee=dict(data.get("Desks_E", {})),
        employees_by_group=dict(data.get("Employees_G", {})),
        days_by_employee=dict(data.get("Days_E", {})),
    )
    build_reverse_indices(inst)
    return inst


# ------------------------------
# Optional helpers for analysis
# ------------------------------

def to_dataframes(inst: ProblemInstance) -> Dict[str, pd.DataFrame]:
    """Return tidy DataFrames for quick inspection and joins."""
    df_desks_zone = (
        pd.DataFrame([(z, d) for z, ds in inst.desks_by_zone.items() for d in ds], columns=["zone", "desk"])
        .sort_values(["zone", "desk"], ignore_index=True)
    )
    df_groups_emp = (
        pd.DataFrame([(g, e) for g, es in inst.employees_by_group.items() for e in es], columns=["group", "employee"])
        .sort_values(["group", "employee"], ignore_index=True)
    )
    df_days_emp = (
        pd.DataFrame([(e, d) for e, ds in inst.days_by_employee.items() for d in ds], columns=["employee", "day"])
        .sort_values(["employee", "day"], ignore_index=True)
    )
    df_compat = (
        pd.DataFrame([(e, d) for e, ds in inst.desks_by_employee.items() for d in ds], columns=["employee", "desk"])
        .sort_values(["employee", "desk"], ignore_index=True)
    )
    return {
        "desks_zone": df_desks_zone,
        "groups_employees": df_groups_emp,
        "days_employees": df_days_emp,
        "compat_employee_desk": df_compat,
    }


def load_all_instances(directory: Union[str, Path]) -> Dict[str, ProblemInstance]:
    """
    Load all *.json instances in a directory. Returns a dict mapping filename stem -> instance.
    """
    directory = Path(directory)
    result: Dict[str, ProblemInstance] = {}
    for p in sorted(directory.glob("*.json")):
        try:
            result[p.stem] = load_instance(p, strict=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load {p.name}: {e}")
    return result