# precalculos.py
from dataclasses import dataclass, fields
from typing import Dict, Set, List
from read_instances import ProblemInstance

@dataclass
class Precalc: # Tipos de datos
    # Básicos
    cap_zone: Dict[str, int]                     # Z -> #desks
    compat: Dict[str, Set[str]]                  # e -> {desks compatibles}
    avail: Dict[str, Set[str]]                   # e -> {días disponibles}
    group_of_emp: Dict[str, str]                 # e -> g
    employees_of_group: Dict[str, List[str]]     # g -> [e,...]
    zone_of_desk: Dict[str, str]                 # d -> z
    # Derivados útiles
    group_size: Dict[str, int]                   # g -> #empleados
    avail_gd: Dict[str, Dict[str, int]]          # g -> {día -> #miembros disponibles}
    common_days_group: Dict[str, List[str]]      # g -> {días en los que todo el grupo está disponible}
    compat_in_zone: Dict[str, Dict[str, int]]    # e -> {zona -> #desks compatibles en esa zona}
    compat_union_gz: Dict[str, Dict[str, int]]   # g -> {zona -> #desks distintos compatibles para el grupo}
    load_day: Dict[str, int]                     # día -> carga inicial (0)
    
    # Print Attributes
    def attribute_names(self, with_types: bool = False) -> list[str]:
        """
        Retorna una lista con los nombres (y opcionalmente tipos) de los atributos
        definidos como campos del dataclass.
        - with_types: devuelve 'nombre: tipo' en lugar de solo el nombre
        """
        names = []
        for f in fields(self):
            if with_types:
                names.append(f"{f.name}: {f.type}")
            else:
                names.append(f.name)
        return names

    def print_attributes(self, with_types: bool = False) -> None:
        """Imprime la lista de atributos (opcionalmente con tipos)."""
        for i, name in enumerate(self.attribute_names(with_types), 1):
            print(f"{i}. {name}")

def compute_precalcs(inst: ProblemInstance) -> Precalc: # Calculo de los datos
    # ---- Básicos
    cap_zone = {z: len(ds) for z, ds in inst.desks_by_zone.items()}
    compat   = {e: set(ds) for e, ds in inst.desks_by_employee.items()}
    avail    = {e: set(ds) for e, ds in inst.days_by_employee.items()}
    group_of_emp = dict(inst.group_of_employee)          # ya lo construye el loader
    employees_of_group = {g: list(es) for g, es in inst.employees_by_group.items()}
    zone_of_desk = dict(inst.zone_of_desk)               # ya lo construye el loader

    # Conjuntos por zona para intersecciones rápidas
    desks_in_zone = {z: set(ds) for z, ds in inst.desks_by_zone.items()}
    
    # tamaño del grupo
    group_size = {g: len(es) for g, es in employees_of_group.items()}

    # ---- g × día: cuántos del grupo pueden asistir ese día
    avail_gd: Dict[str, Dict[str, int]] = {g: {d: 0 for d in inst.days} for g in inst.groups}
    for g, es in employees_of_group.items():
        for e in es:
            for d in avail.get(e, set()):
                avail_gd[g][d] += 1
                
    # ---- g × día: días en los que todo el grupo está disponible
    common_days_group: Dict[str, List[str]] = {g: [] for g in inst.groups}
    for g in employees_of_group:
        for d in inst.days:
            if avail_gd.get(g, {}).get(d, 0) == group_size[g]:
                common_days_group[g].append(d)
                    

    # ---- e × zona: cuántos escritorios compatibles tiene el empleado en cada zona
    compat_in_zone: Dict[str, Dict[str, int]] = {e: {} for e in inst.employees} 
    for e in inst.employees: 
        for z in inst.zones: 
            compat_in_zone[e][z] = len(compat.get(e, set()) & desks_in_zone.get(z, set()))

    # ---- g × zona: escritorios DISTINTOS que cubre el grupo (unión de compatibilidades del grupo)
    compat_union_gz: Dict[str, Dict[str, int]] = {g: {} for g in inst.groups}
    for g, es in employees_of_group.items():
        union_compat = set().union(*(compat.get(e, set()) for e in es))
        for z in inst.zones:
            compat_union_gz[g][z] = len(union_compat & desks_in_zone[z])

    # ---- carga inicial por día (se irá actualizando en Fase 2)
    load_day = {d: 0 for d in inst.days}

    return Precalc(
        cap_zone=cap_zone,
        compat=compat,
        avail=avail,
        group_of_emp=group_of_emp,
        employees_of_group=employees_of_group,
        zone_of_desk=zone_of_desk,
        group_size=group_size,
        avail_gd=avail_gd,
        common_days_group=common_days_group,
        compat_in_zone=compat_in_zone,
        compat_union_gz=compat_union_gz,
        load_day=load_day,
    )