from collections import defaultdict, Counter
from dataclasses import dataclass, fields
import itertools
import math
import pandas as pd
from read_instances import ProblemInstance
from precalculos import Precalc
from typing import Callable, Dict, Iterator, NamedTuple, Set, Tuple, List, Optional
import random
import numpy as np

@dataclass
class Solution:
    group_meeting_day: Dict[str, str]
    schedule_by_employee: Dict[str, Set[str]]
    assignments: List[Tuple[str, str, str]]
    df_assign: pd.DataFrame
    forced_stats: Dict
    valid_assingments: int
    employee_preferences: int
    isolated_employees: int
    score: float
    
    # Print Attributes
    def attribute_names(self, with_types: bool = False) -> list[str]:
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


# ============================ MÉTODOS A IMPORTAR EN EL NOTEBOOK ============================


def build_initial_sol(inst: ProblemInstance, pre: Precalc, weights: List[float]) -> Solution:
    """Constructivo original (greedy determinista)."""
    gmd, _warns = phase1(inst, pre)
    sched, _ = phase2(inst, pre, gmd, target_days_default=2)
    assigns, df_assign, persist, fstats = phase3(inst, pre, sched, prefer_persistent=True)
    VA = count_valid_assignments(inst, assigns)
    EP = count_employee_preferences(inst, sched)
    IE = count_isolated_employees(df_assign)
    
    sol = Solution(
        group_meeting_day = gmd,
        schedule_by_employee = sched,
        assignments = assigns,
        df_assign = df_assign,
        forced_stats = fstats,
        valid_assingments = VA,
        employee_preferences = EP,
        isolated_employees = IE,
        score = 0.0)
    
    calculated_score = evaluate_sol(inst, sol, weights)
    sol.score = calculated_score
    
    return(sol)

def evaluate_sol(inst: ProblemInstance, sol: Solution, weights: List[float]):
    
    VA = sol.valid_assingments
    EP = sol.employee_preferences
    IE = sol.isolated_employees
    
    score = (VA * weights[0]) + (EP * weights[1]) + (IE * weights[2])
    return score


# =============================== MÉTRICAS =============================================

def count_valid_assignments(inst: ProblemInstance, assignments: List[Tuple[str, str, str]]):
    desks_by_employee = {e: set(v) for e, v in inst.desks_by_employee.items()}
    valid_assignments = 0
    for e, d, desk in assignments:
        if desk != "none" and desk in desks_by_employee.get(e, set()):
            valid_assignments += 1
    return valid_assignments

def count_employee_preferences(inst: ProblemInstance, schedule_by_employee: Dict[str, Set[str]]):
    days_by_employee = {e: set(v) for e, v in inst.days_by_employee.items()}
    prefs_count = 0
    for e, days in schedule_by_employee.items():
        prefs_count += sum(1 for d in days if d in days_by_employee.get(e, set()))
    employee_preferences = int(prefs_count)
    return employee_preferences

def count_isolated_employees(df_assign: pd.DataFrame) -> int:  
    # Usamos solo filas con zona conocida
    df = df_assign[df_assign['Zone'].notna()].copy()
    # Conteos por (Group, Day, Zone)
    ctz = (
        df.groupby(['Group','Day','Zone'], dropna=False)
          .size()
          .reset_index(name='n')
    )
    # (Group, Day) candidatos: usan más de 2 zonas
    zones_per_gd = (
        ctz.groupby(['Group','Day'])['Zone']
           .nunique()
           .reset_index(name='num_zones')
    )
    cand = zones_per_gd[zones_per_gd['num_zones'] > 1][['Group','Day']]
    if cand.empty:
        return 0
    ctz = ctz.merge(cand, on=['Group','Day'], how='inner')
    # Aislados por (Group, Day):
    # - si todas las zonas tienen n==1 -> aislados = total del grupo ese día
    # - si hay zonas con n>1 -> aislados = # de personas en zonas con n==1
    def _isolated_grupo_dia(g):
        singles = (g['n'] == 1)
        if singles.all():
            return int(g['n'].sum())
        return int(g.loc[singles, 'n'].sum())
    isolated_total = int(
        ctz.groupby(['Group','Day']).apply(_isolated_grupo_dia, include_groups=False).sum()
        )
    return isolated_total


# =============================== FACTIBILIDAD =============================================

def es_factible(sol: Solution, inst: ProblemInstance) -> bool:
    """
    Verifica factibilidad de una Solution respecto a las restricciones duras usadas en Fase 1–3.
    Restricciones chequeadas:
      (R1) Dominio: grupos, empleados, días y escritorios válidos.
      (R2) Meeting day: todos los miembros de cada grupo asisten al meeting_day de su grupo.
      (R3) Capacidad diaria: #empleados asignados a un día <= #escritorios totales.
      (R4) Cobertura de asignaciones: todo (empleado, día) del schedule tiene exactamente un escritorio asignado.
      (R5) Unicidad de escritorios por día: un desk no puede repetirse el mismo día.
      (R6) Preferencias de días (DURA): para cada empleado, los días asignados
           EXCEPTO el meeting day deben estar dentro de sus preferencias (inst.days_by_employee[e]).
    Nota:
      - La compatibilidad empleado↔escritorio (desk ∈ inst.desks_by_employee[e]) NO se verifica como dura aquí.
    Devuelve:
      True si TODAS las restricciones se cumplen; False en caso contrario.
    """
    # ---------- R1: Dominio básico ----------
    days_set = set(inst.days)
    desks_set = set(inst.desks)
    emps_set  = set(inst.employees)
    groups_set = set(inst.groups)

    # a) group_meeting_day bien formado
    if set(sol.group_meeting_day.keys()) != groups_set:
        return False
    if any((d not in days_set) for d in sol.group_meeting_day.values()):
        return False

    # b) schedule_by_employee bien formado
    if set(sol.schedule_by_employee.keys()) != emps_set:
        return False
    for e, days in sol.schedule_by_employee.items():
        if not set(days).issubset(days_set):
            return False

    # c) assignments bien formadas: (e, d, desk)
    for (e, d, desk) in sol.assignments:
        if e not in emps_set or d not in days_set:
            return False
        if desk not in desks_set:
            return False  # 'none' no permitido (R8)

    # ---------- R2: Meeting day obligatorio ----------
    for g, members in inst.employees_by_group.items():
        gmd = sol.group_meeting_day[g]
        for e in members:
            if gmd not in sol.schedule_by_employee.get(e, set()):
                return False

    # ---------- R3: Capacidad diaria ----------
    load_day = defaultdict(int)
    for e, days in sol.schedule_by_employee.items():
        for d in days:
            load_day[d] += 1
    n_desks = len(desks_set)
    if any(load_day[d] > n_desks for d in days_set):
        return False

    # ---------- R4: Cobertura exacta de asignaciones ----------
    asg_by_ed = defaultdict(list)
    for (e, d, desk) in sol.assignments:
        asg_by_ed[(e, d)].append(desk)

    # (i) cada (e,d) del schedule debe tener exactamente 1 desk
    for e, days in sol.schedule_by_employee.items():
        for d in days:
            desks_here = asg_by_ed.get((e, d), [])
            if len(desks_here) != 1:
                return False

    # (ii) no deben existir assignments para (e,d) que no esté en schedule
    for (e, d) in asg_by_ed.keys():
        if d not in sol.schedule_by_employee.get(e, set()):
            return False

    # ---------- R5: Unicidad de escritorios por día ----------
    used_by_day = defaultdict(list)
    for (e, d, desk) in sol.assignments:
        used_by_day[d].append(desk)
    for d, desks_d in used_by_day.items():
        if len(desks_d) != len(set(desks_d)):
            return False

    # ---------- R6: Preferencias de días (DURA, excluye meeting) ----------
    # Mapa empleado->grupo y meeting day de su grupo
    emp_group: Dict[str, str] = {}
    for g, mems in inst.employees_by_group.items():
        for e in mems:
            emp_group[e] = g

    # Preferencias: se espera inst.days_by_employee[e] (lista de días preferidos).
    # Si no existe ese atributo, la restricción no se puede verificar -> consideramos NO factible.
    if not hasattr(inst, "days_by_employee"):
        return False

    for e, days in sol.schedule_by_employee.items():
        g = emp_group[e]
        md = sol.group_meeting_day[g]
        non_meeting_days = set(days) - {md}

        prefs_e = set(inst.days_by_employee.get(e, []))
        # Si el empleado no tiene preferencias registradas, no podemos aceptar días no-meeting
        if not prefs_e and non_meeting_days:
            return False

        if not non_meeting_days.issubset(prefs_e):
            return False

    return True


# ============================ FUNCIONES Y CLASES DE UTILIDAD ============================

class Move(NamedTuple):
    kind: str
    payload: tuple 

def _meeting_load_by_day(gmd: Dict[str, str], pre, days: list[str]) -> Dict[str, int]:
    """Carga de meeting por día: suma tamaños de grupo en ese día."""
    load = {d: 0 for d in days}
    for g, d in gmd.items():
        load[d] += pre.group_size[g]
    return load

def _day_counts(schedule_by_employee: Dict[str, Set[str]], days: List[str]) -> Dict[str, int]:
    cnt = {d: 0 for d in days}
    for e, ds in schedule_by_employee.items():
        for d in ds:
            cnt[d] += 1
    return cnt

def _emp_group_map(inst) -> Dict[str, str]:
    return {e: g for g, mems in inst.employees_by_group.items() for e in mems}

def _meeting_day_of(employee: str, emp_group: Dict[str,str], gmd: Dict[str,str]) -> Optional[str]:
    g = emp_group.get(employee)
    return gmd.get(g, None)

def _assignments_index(sol: Solution):
    """Índices rápidos: (e,d)->desk y day->list of (e,desk)."""
    ed_to_desk = {}
    by_day = {d: [] for d in set(d for _, d, _ in sol.assignments)}
    for e, d, desk in sol.assignments:
        ed_to_desk[(e, d)] = desk
        by_day.setdefault(d, []).append((e, desk))
    return ed_to_desk, by_day

def apply_and_eval(sol: Solution, move: Move, inst, pre, weights) -> Optional[Solution]:
    """
    Devuelve una nueva Solution si el movimiento es factible; si no, None.
    Reutiliza evaluate_sol(inst, sol, weights) para actualizar sol.score.
    """
    kind, P = move.kind, move.payload

    if kind == "SwapDesksSameDay":
        e1, e2, d = P
        # mapa (e,d)->desk
        ed_to_desk, _ = _assignments_index(sol)
        desk1 = ed_to_desk.get((e1, d), "none")
        desk2 = ed_to_desk.get((e2, d), "none")

        # construir nuevas asignaciones (swap solo en ese día)
        new_assigns = []
        for e, dd, desk in sol.assignments:
            if dd != d:
                new_assigns.append((e, dd, desk))
            else:
                if e == e1:
                    new_assigns.append((e, dd, desk2))
                elif e == e2:
                    new_assigns.append((e, dd, desk1))
                else:
                    new_assigns.append((e, dd, desk))


        # Nota: como solo movimos desks, gmd y sched no cambian.
        sol2 = rebuild_from_phase3(
            inst, pre,
            sol.group_meeting_day,
            sol.schedule_by_employee
        )
        # sobre-escribir assignments para NO perder persistencias (ya reconstruidas)
        # Creamos una nueva instancia Solution preservando los datos recalculados en rebuild,
        # pero inyectando los assignments modificados manualmente arriba.
        sol2 = Solution(
            group_meeting_day=sol2.group_meeting_day,
            schedule_by_employee=sol2.schedule_by_employee,
            assignments=new_assigns,
            df_assign=sol2.df_assign,
            forced_stats=sol2.forced_stats,
            valid_assingments=sol2.valid_assingments,
            employee_preferences=sol2.employee_preferences,
            isolated_employees=sol2.isolated_employees,
            score=sol2.score
        )
        
        new_score = evaluate_sol(inst, sol2, weights)
        if new_score is None or math.isnan(new_score):
            return None
        
        sol2.score = float(new_score)
        return sol2

    if kind == "InsertDay":
        e, d_out, d_in = P
        # construir nuevo schedule (copia profunda – solo e cambia)
        sched2 = {k: set(v) for k, v in sol.schedule_by_employee.items()}
        if d_out not in sched2.get(e, set()):
            return None
        sched2[e].remove(d_out)
        sched2[e].add(d_in)

        # reconstruir solución SOLO desde (gmd, sched2)
        sol2 = rebuild_from_phase3(
            inst, pre,
            sol.group_meeting_day,
            sched2
        )
        new_score = evaluate_sol(inst, sol2, weights)
        if new_score is None or math.isnan(new_score):
            return None
        
        sol2.score = float(new_score)
        return sol2

    if kind == "InsertMeetingDay":
        g, d_old, d_new = P

        # 1) clonar gmd y aplicar el cambio
        gmd2 = dict(sol.group_meeting_day)
        if gmd2.get(g) != d_old:
            return None  # inconsistencia
        gmd2[g] = d_new

        # 2) reconstruir solución completa desde (gmd2, Fase 2 y Fase 3)
        sol2 = rebuild_from_phase2(inst, pre, gmd2, target_days_default=2)
        new_score = evaluate_sol(inst, sol2, weights)
        if new_score is None:
            return None
            
        sol2.score = float(new_score)
        return sol2

    if kind == "SwapMeetingDay":
        g1, d1, g2, d2 = P

        # 1) clonar gmd y aplicar el swap
        gmd2 = dict(sol.group_meeting_day)
        if not (gmd2.get(g1) == d1 and gmd2.get(g2) == d2):
            return None  # inconsistencia
        gmd2[g1], gmd2[g2] = d2, d1

        # 2) reconstruir solución completa desde (gmd2, Fase 2 y Fase 3)
        sol2 = rebuild_from_phase2(inst, pre, gmd2, target_days_default=2)
        new_score = evaluate_sol(inst, sol2, weights)
        if new_score is None:
            return None
            
        sol2.score = float(new_score)
        return sol2
    
    return None


# ============================ OPERADORES DE MUTACIÓN ============================

def neigh_swap_desks_same_day(sol: Solution, inst, pre) -> Iterator[Move]:
    """
    Genera intercambios de escritorios entre dos empleados presentes el mismo día.
    Orden determinista: por día asc, luego por par (e1<e2).
    """
    _, by_day = _assignments_index(sol)
    desks_by_employee = {e: set(v) for e, v in inst.desks_by_employee.items()}

    for d in sorted(by_day.keys()):
        # listar solo quienes tienen desk asignado ese día
        todays = [(e, desk) for (e, desk) in by_day[d] if desk and desk.lower() != "none"]
        # pares ordenados
        for (e1, desk1), (e2, desk2) in itertools.combinations(sorted(todays), 2):
            # factibilidad del swap (compatibilidad cruzada)
            if desk2 in desks_by_employee.get(e1, set()) and desk1 in desks_by_employee.get(e2, set()):
                yield Move("SwapDesksSameDay", (e1, e2, d))
neigh_swap_desks_same_day.KIND = "SwapDesksSameDay"

def neigh_insert_day(sol: Solution, inst, pre) -> Iterator[Move]:
    """
    Mueve a un empleado de un día no-meeting a otro día de sus preferencias con capacidad.
    Orden determinista por empleado y días.
    """
    gmd = sol.group_meeting_day
    sched = sol.schedule_by_employee
    emp_group = _emp_group_map(inst)
    day_cap = len(inst.desks)
    day_cnt = _day_counts(sched, inst.days)

    for e in sorted(inst.employees):
        days_e = set(sched.get(e, set()))
        d_meet = _meeting_day_of(e, emp_group, gmd)
        # candidatos a quitar (no meeting)
        removable = sorted([d for d in days_e if d != d_meet])
        if not removable:
            continue
        # días preferidos del empleado
        pref = sorted(set(inst.days_by_employee.get(e, [])))
        for d_out in removable:
            for d_in in pref:
                if d_in == d_out or d_in in days_e:
                    continue
                if day_cnt[d_in] + 1 > day_cap:
                    continue
                yield Move("InsertDay", (e, d_out, d_in))
neigh_insert_day.KIND = "InsertDay"

def neigh_insert_meeting_day(sol: Solution, inst, pre) -> Iterator[Move]:
    """
    Mueve el meeting day de un grupo g desde d_old a d_new, si hay capacidad dura:
        load[d_new] + size[g] <= #desks
    Orden determinista: grupos y días ordenados.
    """
    gmd = sol.group_meeting_day
    days = list(inst.days)
    cap = len(inst.desks)
    load = _meeting_load_by_day(gmd, pre, days)

    for g in sorted(gmd.keys()):
        d_old = gmd[g]
        size_g = pre.group_size[g]
        for d_new in sorted(days):
            if d_new == d_old:
                continue
            if load[d_new] + size_g <= cap:
                # Vecino “insert”: g pasa a d_new (no toca otros grupos)
                yield Move("InsertMeetingDay", (g, d_old, d_new))                
neigh_insert_meeting_day.KIND = "InsertMeetingDay"

def neigh_swap_meeting_day(sol: Solution, inst, pre) -> Iterator[Move]:
    """
    Intercambia meeting entre dos grupos (g1,d1) y (g2,d2), si al final
    ambas cargas diarias respetan capacidad.
    Orden determinista por par de grupos.
    """
    gmd = sol.group_meeting_day
    days = list(inst.days)
    cap = len(inst.desks)
    load = _meeting_load_by_day(gmd, pre, days)

    groups_sorted = sorted(gmd.keys())
    for g1, g2 in itertools.combinations(groups_sorted, 2):
        d1, d2 = gmd[g1], gmd[g2]
        if d1 == d2:
            continue
        s1, s2 = pre.group_size[g1], pre.group_size[g2]

        # nuevas cargas tras el swap:
        # d1 pierde s1 y gana s2; d2 pierde s2 y gana s1
        ok_d1 = (load[d1] - s1 + s2) <= cap
        ok_d2 = (load[d2] - s2 + s1) <= cap
        if ok_d1 and ok_d2:
            yield Move("SwapMeetingDay", (g1, d1, g2, d2))          
neigh_swap_meeting_day.KIND = "SwapMeetingDay"


# ============================ FASES DEL MÉTODO CONSTRUCTIVO ==============================

# ------------- FASE 1 ----------------
def phase1(inst, pre, weights=(0.6, 0.2, 0.2), prefer_common_first=True):
    """
    Asigna 1 meeting day por grupo con CAPACIDAD DURA por día:
    - Nunca permite que la suma de tamaños de grupos asignados a un día supere |desks|.
    - Prefiere días 100% comunes; si no caben, usa el segundo mejor día con capacidad y así sucesivamente.
    - Si NINGÚN día tiene capacidad suficiente para un grupo, reporta inviabilidad.

    Retorna:
      group_meeting_day: dict[group, day]
      warnings: list[str]
    """
    # Helpers en sets para rapidez
    days_by_employee = {e: set(v) for e, v in inst.days_by_employee.items()}
    employees_by_group = {g: set(v) for g, v in inst.employees_by_group.items()}
    desk_count = len(inst.desks)

    # Capacidad dura por día
    day_capacity_left = {d: desk_count for d in inst.days}

    # Orden de grupos: grandes primero (más difíciles)
    groups_sorted = sorted(
        inst.groups,
        key=lambda g: ((-pre.group_size[g]),g)
    )

    # Cache de cohesión por zonas (constante en d)
    def cluster(g):
        counts = []
        for z in inst.zones:
            c = sum(1 for e in employees_by_group[g] if pre.compat_in_zone.get(e, {}).get(z, 0) > 0)
            counts.append(c)
        counts.sort(reverse=True)
        return ((counts[0] if counts else 0) + (counts[1] if len(counts) > 1 else 0)) / max(1, pre.group_size[g])
    cluster_cache = {g: cluster(g) for g in inst.groups}

    def coverage(g, d):
        return sum(1 for e in employees_by_group[g] if d in days_by_employee[e]) / max(1, pre.group_size[g])

    def day_score(g, d):
        """Score para ordenar días (sin violar capacidad)."""
        cov = coverage(g, d)
        # usamos holgura real (proporción de sillas libres)
        slack = day_capacity_left[d] / max(1, desk_count)
        coh = cluster_cache[g]
        is_common = 1 if d in set(pre.common_days_group.get(g, [])) else 0
        score = (weights[0]*cov + weights[1]*slack + weights[2]*coh + 0.05*is_common)
        # tupla con desempates deterministas
        return (is_common if prefer_common_first else 0, score, cov, slack, coh, -pre.group_size[g])

    group_meeting_day = {}
    warnings = []

    for g in groups_sorted:
        size_g = pre.group_size[g]
        commons = set(pre.common_days_group.get(g, []))

        # 1) Construir listas de días factibles (con capacidad dura >= tamaño del grupo)
        feasible_common = [d for d in inst.days if d in commons and day_capacity_left[d] >= size_g]
        feasible_noncommon = [d for d in inst.days if d not in commons and day_capacity_left[d] >= size_g]

        chosen = None

        # 2) Si hay días comunes factibles, elige el mejor por score
        if feasible_common:
            feasible_common.sort(key=lambda d: day_score(g, d), reverse=True)
            chosen = feasible_common[0]
            warnings.append(
                f"[{g}] con día 100% común con capacidad; se eligió '{chosen}' con cobertura "
                f"{coverage(g, chosen):.2f} (preferencias al 100%)."
            )
        # 3) Si no caben en días comunes, intenta no comunes (con aviso)
        elif feasible_noncommon:
            feasible_noncommon.sort(key=lambda d: day_score(g, d), reverse=True)
            chosen = feasible_noncommon[0]
            warnings.append(
                f"[{g}] sin día 100% común con capacidad; se eligió '{chosen}' con cobertura "
                f"{coverage(g, chosen):.2f} (preferencias no 100%)."
            )
        else:
            # 4) Ningún día tiene capacidad suficiente para este grupo
            warnings.append(
                f"[{g}] INVIABLE en Fase 1 con capacidad dura: tamaño del grupo={size_g} "
                f"no cabe en ningún día con capacidad restante. Considera repair o cambiar pesos/orden."
            )
            # Puedes optar por continuar sin asignar o por romper aquí:
            # raise RuntimeError(warnings[-1])
            continue

        # 5) Fijar elección y descontar capacidad
        group_meeting_day[g] = chosen
        day_capacity_left[chosen] -= size_g

    # Chequeo de seguridad: no debe haber día excedido
    for d in inst.days:
        used = desk_count - day_capacity_left[d]
        if used > desk_count:
            raise AssertionError(f"Capacidad excedida en {d}: {used}>{desk_count}")

    return group_meeting_day, warnings

# ------------- FASE 2 ----------------
def phase2(inst, pre, group_meeting_day, target_days_default):
    """
    Fase 2 (con 'meeting day' obligatorio y heurística de menor holgura):
    - Paso 1: Forzar asistencia al meeting day de cada grupo (capacidad dura por día).
    - Paso 2: Completar días extra por empleado, priorizando:
        (a) empleados con menos opciones disponibles (apretados primero),
        (b) días con más compañeros de su grupo (cohesión),
        (c) días con mayor capacidad restante (factibilidad).

    Entradas:
      - inst: objeto con .employees, .days, .desks, .days_by_employee, .employees_by_group
      - pre:  objeto con .group_size (no es estrictamente necesario aquí, pero útil para ordenar grupos)
      - group_meeting_day: Dict[group, day]
      - target_days_default: int (p. ej., 2)

    Salidas:
      - schedule_by_employee: Dict[emp, Set[day]]
      - day_capacity_left: Dict[day, int]
    """
    # --------- helpers a sets ---------
    days_by_employee = {e: set(v) for e, v in inst.days_by_employee.items()}
    employees_by_group = {g: set(v) for g, v in inst.employees_by_group.items()}

    # --------- capacidad dura por día ---------
    day_capacity_left = {d: len(inst.desks) for d in inst.days}

    # --------- inicializaciones ---------
    schedule_by_employee = {e: set() for e in inst.employees}
    group_pressure = {d: Counter() for d in inst.days}  # cuántos del grupo g ya van el día d

    # ====================================================
    # Paso 1: meeting day OBLIGATORIO (grupos grandes primero)
    # ====================================================
    for g in sorted(inst.groups, key=lambda x: -pre.group_size[x]):
        d_meet = group_meeting_day[g]
        members = employees_by_group[g]
        # Seguridad: con Fase 1 de capacidad dura, esto no debería fallar
        if day_capacity_left[d_meet] < len(members):
            raise AssertionError(
                f"Capacidad insuficiente en {d_meet} para forzar meeting day de {g}. "
                f"Capacidad restante={day_capacity_left[d_meet]}, miembros={len(members)}."
            )
        for e in members:
            schedule_by_employee[e].add(d_meet)
            day_capacity_left[d_meet] -= 1
            group_pressure[d_meet][g] += 1

    # ====================================================
    # Paso 2: completar días extra por empleado
    # Heurística 'apretados primero' (menor holgura):
    #   - calcular cuántas opciones reales tiene cada empleado (preferencias con cupo)
    #   - ordenar ascendente por #opciones, y en empate, por mayor necesidad (target que le falte)
    # ====================================================

    def need_of(e):
        return max(0, target_days_default - len(schedule_by_employee[e]))

    def options_now(e):
        """Días preferidos disponibles (capacidad > 0) que aún NO tenga asignados."""
        return [d for d in days_by_employee.get(e, set())
                if d not in schedule_by_employee[e] and day_capacity_left[d] > 0]

    # Construir lista de empleados con necesidad > 0
    employees_to_assign = [e for e in inst.employees if need_of(e) > 0]

    # Orden inicial por 'apretados primero' (menos opciones → primero), y en empate por mayor necesidad
    employees_to_assign.sort(key=lambda e: (len(options_now(e)), -need_of(e)))

    # Asignación greedy por empleado (recalculando candidatos con capacidad dinámica)
    for e in employees_to_assign:
        g_e = next((g for g, mems in employees_by_group.items() if e in mems), None)
        if g_e is None:
            continue

        # Recalcular necesidad y opciones con la capacidad actual
        need = need_of(e)
        if need <= 0:
            continue

        # Candidatos: solo preferencias (extra meeting) con cupo
        candidates = []
        for d in inst.days:
            if d in schedule_by_employee[e]:
                continue
            if d not in days_by_employee.get(e, set()):
                continue
            cap = day_capacity_left[d]
            if cap <= 0:
                continue
            mates = group_pressure[d][g_e]
            candidates.append((mates, cap, d))

        # Orden: más compañeros primero, luego más capacidad
        candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)

        # Asignar hasta cumplir 'need' o quedarnos sin candidatos
        for mates, cap, d in candidates:
            if need <= 0:
                break
            if day_capacity_left[d] <= 0:
                continue
            schedule_by_employee[e].add(d)
            day_capacity_left[d] -= 1
            group_pressure[d][g_e] += 1
            need -= 1
        # Si 'need' sigue > 0, el empleado se queda con menos días por falta de preferencias/cupo

    return schedule_by_employee, day_capacity_left

# ------------- FASE 3 ----------------
def phase3(inst, pre, schedule_by_employee: Dict[str, Set[str]], prefer_persistent: bool = True, rk_employee_order: Optional[List[str]] = None):
    
    """
    Asigna escritorios por día respetando compatibilidad y unicidad.
    Al final, garantiza que nadie quede con 'none' si está asignado a ese día,
    rellenando con cualquier desk libre del día y reportando:
      - cuántos casos se forzaron,
      - quiénes fueron (global y por día),
      - de esos, cuántos y quiénes quedaron en desks NO compatibles.
    """
    # --- índices deterministas ---
    # Si estos vienen como sets, imponemos orden
    days_order = {d:i for i,d in enumerate(list(inst.days))}
    zones_sorted = list(sorted(inst.zones))
    desks_all = set(inst.desks)  # set para operaciones; cuando el orden importe, usaremos sorted(...)
    
    # Map empleado->grupo (determinista)
    emp_group = {}
    for g in sorted(inst.employees_by_group.keys()):
        for e in sorted(inst.employees_by_group[g]):
            emp_group[e] = g

    desks_by_employee = {e: set(v) for e, v in inst.desks_by_employee.items()}
    desks_by_zone     = {z: set(v) for z, v in inst.desks_by_zone.items()}
    zone_of = pre.zone_of_desk

    # Empleados presentes por día (ordenar empleados)
    employees_by_day: Dict[str, List[str]] = {d: [] for d in inst.days}
    #for e in sorted(inst.employees):
        #for d in sorted(schedule_by_employee.get(e, set()), key=lambda x: days_order[x]):
            #employees_by_day[d].append(e)
    
    employees_source = rk_employee_order if rk_employee_order is not None else sorted(inst.employees)

    for e in employees_source: # <--- Ahora usa el orden RK
        for d in sorted(schedule_by_employee.get(e, set()), key=lambda x: days_order[x]):
            employees_by_day[d].append(e)

    persistent_desk: Dict[str, str] = {}
    assignments: List[Tuple[str, str, str]] = []
    #df_assign: pd.DataFrame = pd.DataFrame(columns=["Employee","Day","Desk"])

    # --- por día en el orden de inst.days ---
    for d in inst.days:
        todays = employees_by_day[d]
        if not todays:
            continue

        free_desks = set(desks_all)

        # Agrupar por grupo (con orden determinista de grupos y miembros)
        groups_today = defaultdict(list)
        for e in todays:  # 'todays' ya está ordenado
            g_e = emp_group.get(e)
            groups_today[g_e].append(e)

        # Grupos grandes primero, y en empate por id de grupo
        groups_sorted = sorted(groups_today.keys(), key=lambda g: (-len(groups_today[g]), g))

        for g in groups_sorted:
            members = sorted(groups_today[g])  # orden fijo por id de empleado

            # --- elegir 1–2 zonas ancla con orden fijo ---
            zone_scores = []
            for z in zones_sorted:  # orden fijo de zonas
                desks_z_free = free_desks & desks_by_zone[z]
                if not desks_z_free:
                    zone_scores.append((0, z)); continue
                count_fit = sum(1 for e in members if (desks_by_employee[e] & desks_z_free))
                zone_scores.append((count_fit, z))
            # mayor cobertura primero; en empate, por id de zona (ya va en la tupla)
            zone_scores.sort(reverse=True)
            top_zones = [z for _, z in zone_scores[:2]]

            # --- intento principal: persistencia + cohesión ---
            unassigned = []
            for e in members:
                chosen = None

                # 1) persistencia (si cae en top_zones)
                if prefer_persistent and e in persistent_desk:
                    p_d = persistent_desk[e]
                    if (p_d in free_desks) and (p_d in desks_by_employee[e]) and (zone_of.get(p_d) in top_zones):
                        chosen = p_d

                # 2) compatible en top_zones, "rara-es-mejor" con tiebreak determinista
                if chosen is None and top_zones:
                    # juntar candidatos compatibles en todas las top_zones
                    cand = set()
                    for z in top_zones:
                        if z in desks_by_zone:
                            cand |= (desks_by_employee[e] & free_desks & desks_by_zone[z])
                    if cand:
                        def demand_in_group(desk):
                            return sum(1 for ee in members if desk in desks_by_employee[ee])
                        # tiebreak determinista por id de desk
                        chosen = min(cand, key=lambda desk: (demand_in_group(desk), desk))

                if chosen is None:
                    unassigned.append(e)
                else:
                    assignments.append((e, d, chosen))
                    free_desks.remove(chosen)
                    if e not in persistent_desk:
                        persistent_desk[e] = chosen

            # --- fallback: fuera de top_zones ---
            for e in unassigned:
                cand = desks_by_employee[e] & free_desks
                chosen = None

                # persistencia fuera de top_zones (determinista)
                if prefer_persistent and e in persistent_desk and persistent_desk[e] in cand:
                    chosen = persistent_desk[e]

                # cualquier compatible libre, "rara-es-mejor" a nivel día con tiebreak por id
                if chosen is None and cand:
                    def demand_today(desk):
                        return sum(1 for ee in todays if desk in desks_by_employee[ee])
                    chosen = min(cand, key=lambda desk: (demand_today(desk), desk))

                if chosen is None:
                    # marcamos 'none' temporal; lo resolveremos determinísticamente al final del día
                    assignments.append((e, d, "none"))
                else:
                    assignments.append((e, d, chosen))
                    free_desks.remove(chosen)
                    if e not in persistent_desk:
                        persistent_desk[e] = chosen

    # -------- POST-PASO determinista (sin 'none') --------
    forced_fill_total = 0
    forced_fill_incompatible_total = 0
    forced_fill_by_day = defaultdict(int)
    forced_fill_incompatible_by_day = defaultdict(int)
    forced_fill_employees: List[str] = []
    forced_fill_employees_by_day = defaultdict(list)
    forced_incompatible_employees: List[str] = []
    forced_incompatible_employees_by_day = defaultdict(list)

    # índice: day -> indices en 'assignments'
    idxs_by_day = defaultdict(list)
    for i, (_, d, _) in enumerate(assignments):
        idxs_by_day[d].append(i)

    desks_all_sorted = sorted(inst.desks)  # orden fijo para el fill

    for d in inst.days:
        idxs = idxs_by_day.get(d, [])
        if not idxs: 
            continue
        used = set()
        none_idxs = []
        for i in idxs:
            e_i, d_i, desk_i = assignments[i]
            if desk_i == "none":
                none_idxs.append(i)
            else:
                used.add(desk_i)

        free = [k for k in desks_all_sorted if k not in used]  # determinista

        if len(free) < len(none_idxs):
            raise AssertionError(f"No hay suficientes escritorios libres en {d} para rellenar 'none'.")

        # rellenar en orden determinista de las apariciones
        for i in none_idxs:
            chosen = free.pop(0)
            e_i, d_i, _ = assignments[i]
            assignments[i] = (e_i, d_i, chosen)

            forced_fill_total += 1
            forced_fill_by_day[d_i] += 1
            forced_fill_employees.append(e_i)
            forced_fill_employees_by_day[d_i].append(e_i)

            if chosen not in desks_by_employee.get(e_i, set()):
                forced_fill_incompatible_total += 1
                forced_fill_incompatible_by_day[d_i] += 1
                forced_incompatible_employees.append(e_i)
                forced_incompatible_employees_by_day[d_i].append(e_i)

            if e_i not in persistent_desk:
                persistent_desk[e_i] = chosen

    forced_stats = {
        "forced_fill_total": forced_fill_total,
        "forced_fill_by_day": dict(forced_fill_by_day),
        "forced_fill_employees": forced_fill_employees,
        "forced_fill_employees_by_day": {d: emps for d, emps in forced_fill_employees_by_day.items()},
        "forced_fill_incompatible_total": forced_fill_incompatible_total,
        "forced_fill_incompatible_by_day": dict(forced_fill_incompatible_by_day),
        "forced_incompatible_employees": forced_incompatible_employees,
        "forced_incompatible_employees_by_day": {d: emps for d, emps in forced_incompatible_employees_by_day.items()},
    }
    
    # Dataframe de assignments (incluye zonas y grupos):
    df_assign = pd.DataFrame(assignments, columns=["Employee","Day","Desk"])
    # Para agregar el grupo
    group_col = df_assign['Employee'].map(pre.group_of_emp)
    df_assign.insert(0, 'Group', group_col)
    # Para agregar la zona
    pos = df_assign.columns.get_loc('Desk') + 1
    zone_col = df_assign['Desk'].map(pre.zone_of_desk)
    df_assign.insert(pos, 'Zone', zone_col)

    return assignments, df_assign, persistent_desk, forced_stats

# Para cuando usemos los vecindarios 1 y 2
def rebuild_from_phase3(inst, pre,
                        group_meeting_day: Dict[str, str],
                        schedule_by_employee: Dict[str, Set[str]]) -> Solution:
    
    # Fase 3 
    assignments, df_assign, persistent_desk, forced_stats = phase3(inst, pre, schedule_by_employee, prefer_persistent=True)
    
    VA = count_valid_assignments(inst, assignments)
    EP = count_employee_preferences(inst, schedule_by_employee)
    IE = count_isolated_employees(df_assign)
    
    sol = Solution(
        group_meeting_day=group_meeting_day,
        schedule_by_employee=schedule_by_employee,
        assignments=assignments,
        df_assign=df_assign,
        forced_stats=forced_stats,
        valid_assingments=VA,
        employee_preferences=EP,
        isolated_employees=IE,
        score = 0.0
    )
    sol.score = evaluate_sol(inst, sol, weights=[0.2, 0.4, -0.4])
    return sol
    
# Para cuando usemos los vecindarios 3 y 4
def rebuild_from_phase2(inst, pre,
                        group_meeting_day: Dict[str, str],
                        target_days_default: int = 2,
                        prefer_persistent: bool = True) -> Solution:
    
    # Fase 2
    schedule_by_employee, _ = phase2(inst, pre, group_meeting_day, target_days_default=2)
    # Fase 3 
    assignments, df_assign, persistent_desk, forced_stats = phase3(inst, pre, schedule_by_employee, prefer_persistent=True)
    
    VA = count_valid_assignments(inst, assignments)
    EP = count_employee_preferences(inst, schedule_by_employee)
    IE = count_isolated_employees(df_assign)
    
    sol = Solution(
        group_meeting_day=group_meeting_day,
        schedule_by_employee=schedule_by_employee,
        assignments=assignments,
        df_assign=df_assign,
        forced_stats=forced_stats,
        valid_assingments=VA,
        employee_preferences=EP,
        isolated_employees=IE,
        score = 0.0
    )
    sol.score = evaluate_sol(inst, sol, weights=[0.2, 0.4, -0.4])
    return sol


# ======================= FASES DEL MÉTODO CONSTRUCTIVO ALEATORIZADO ==============================

# ------------- FASE 1 ----------------
def phase1_randomized(inst, pre, weights: Tuple, alpha: float, group_jitter: float, seed: int):
    rng = random.Random(seed)

    days_by_employee = {e: set(v) for e, v in inst.days_by_employee.items()}
    employees_by_group = {g: set(v) for g, v in inst.employees_by_group.items()}
    desk_count = len(inst.desks)
    day_capacity_left = {d: desk_count for d in inst.days}

    # Orden: más grandes primero, desempate con el jitter
    groups_sorted = sorted(
        inst.groups,
        key=lambda g: ((-pre.group_size[g]),
                       (-pre.group_size[g]) + rng.uniform(-group_jitter, group_jitter))
    )

    # Precalcular "cluster" por grupo (capacidad de compactarse en 1–2 zonas)
    def cluster(g):
        members = employees_by_group[g]
        counts = []
        for z in inst.zones:
            c = sum(1 for e in members if pre.compat_in_zone.get(e, {}).get(z, 0) > 0)
            counts.append(c)
        counts.sort(reverse=True)
        top1 = counts[0] if counts else 0
        top2 = counts[1] if len(counts) > 1 else 0
        return (top1 + top2) / max(1, pre.group_size[g])
    cluster_cache = {g: cluster(g) for g in inst.groups}

    def coverage(g, d):
        return sum(1 for e in employees_by_group[g] if d in days_by_employee[e]) / max(1, pre.group_size[g])

    def day_score(g, d):
        cov = coverage(g, d)
        slack = day_capacity_left[d] / max(1, desk_count)
        coh = cluster_cache[g]
        is_common = 1 if d in set(pre.common_days_group.get(g, [])) else 0
        return weights[0]*cov + weights[1]*slack + weights[2]*coh + 0.05*is_common

    def choose_from_rcl(cands, score_fn):
        if not cands:
            return None
        scored = [(score_fn(d), d) for d in cands]
        scored.sort(reverse=True)
        s_max = scored[0][0]; s_min = scored[-1][0]
        thr = s_max - alpha*(s_max - s_min)
        rcl = [d for s,d in scored if s >= thr]
        return rng.choice(rcl)

    group_meeting_day = {}
    warnings = []

    for g in groups_sorted:
        size_g = pre.group_size[g]
        commons = set(pre.common_days_group.get(g, []))

        feasible_common = [d for d in inst.days if d in commons and day_capacity_left[d] >= size_g]
        feasible_noncommon = [d for d in inst.days if d not in commons and day_capacity_left[d] >= size_g]

        chosen = None
        if feasible_common:
            chosen = choose_from_rcl(feasible_common, lambda d: day_score(g, d))
        elif feasible_noncommon:
            chosen = choose_from_rcl(feasible_noncommon, lambda d: day_score(g, d))
            warnings.append(
                f"[{g}] sin día 100% común con capacidad; elegido '{chosen}' cobertura "
                f"{coverage(g, chosen):.2f} (preferencias no 100%)."
                )
        else:
            warnings.append(f"[{g}] INVIABLE: no cabe en ningún día (capacidad dura).")
            continue

        group_meeting_day[g] = chosen
        day_capacity_left[chosen] -= size_g

    return group_meeting_day, warnings

# ------------- FASE 2 ----------------
def phase2_randomized(
    inst, pre, group_meeting_day: Dict[str, str],
    target_days_default: int,
    alpha_days: float,   # RCL por umbral sobre CAPACIDAD, después de fijar 'mates'
    tie_jitter: float,    # jitter SOLO para desempatar orden entre empleados “apretados”
    seed: int | None = None
):
    rng = random.Random(seed)

    # --- sets para rapidez ---
    days_by_employee = {e: set(v) for e, v in inst.days_by_employee.items()}
    employees_by_group = {g: set(v) for g, v in inst.employees_by_group.items()}
    desk_count = len(inst.desks)

    # Mapa empleado -> grupo (para no buscarlo todo el tiempo)
    emp_group = {e: g for g, mems in inst.employees_by_group.items() for e in mems}

    # Capacidad por día y resultado
    day_capacity_left: Dict[str, int] = {d: desk_count for d in inst.days}
    schedule_by_employee: Dict[str, Set[str]] = {e: set() for e in inst.employees}
    # Presión de grupo por día
    group_pressure: Dict[str, Counter] = {d: Counter() for d in inst.days}

    # ---- Paso 0: MEETING DAY obligatorio (determinista) ----
    for g in sorted(inst.groups, key=lambda x: -pre.group_size[x]):
        d_meet = group_meeting_day[g]
        members = employees_by_group[g]
        if day_capacity_left[d_meet] < len(members):
            raise AssertionError(f"Capacidad insuficiente en {d_meet} para grupo {g} (Fase 2).")
        for e in sorted(members):  # orden fijo dentro del grupo
            schedule_by_employee[e].add(d_meet)
            day_capacity_left[d_meet] -= 1
            group_pressure[d_meet][g] += 1

    # Helpers
    day_index = {d: i for i, d in enumerate(list(inst.days))}  # para desempates deterministas
    def need_of(e: str) -> int:
        return max(0, target_days_default - len(schedule_by_employee[e]))

    def options_now(e: str) -> List[str]:
        # Días en preferencias, aún no asignados y con cupo
        return [d for d in inst.days
                if (d in days_by_employee.get(e, set()))
                and (d not in schedule_by_employee[e])
                and (day_capacity_left[d] > 0)]

    # ---- Orden de empleados "apretados" (con posible jitter en empates) ----
    employees_to_assign = [e for e in inst.employees if need_of(e) > 0]
    employees_to_assign.sort(
        key=lambda e: (len(options_now(e)),
                       -need_of(e),
                       (rng.random()*tie_jitter if tie_jitter > 0 else 0.0))
    )

    # ---- Completar días por empleado ----
    for e in employees_to_assign:
        g_e = emp_group.get(e, None)
        if g_e is None:
            # sin grupo (raro), igual completamos según preferencias/capacidad
            pass

        need = need_of(e)
        # Iterar hasta cubrir necesidad o sin candidatos
        while need > 0:
            cands = options_now(e)
            if not cands:
                break

            # 1) PRIORIDAD por "mates": quedarse con los días con MÁXIMO group_pressure[d][g_e]
            #    (si g_e es None, tratamos pressure = 0)
            mates_vals = [(group_pressure[d][g_e] if g_e is not None else 0, d) for d in cands]
            max_mates = max(m for m, _ in mates_vals)
            top_by_mates = [d for (m, d) in mates_vals if m == max_mates]

            # 2) RCL por umbral 'alpha_days' usando CAPACIDAD NORMALIZADA
            caps = [(day_capacity_left[d] / desk_count, d) for d in top_by_mates]
            # Si alpha_days == 0.0, elegimos determinista el de mayor capacidad (tie por orden de día)
            if alpha_days <= 0.0:
                d_pick = max(top_by_mates, key=lambda d: (day_capacity_left[d], -day_index[d]))
            else:
                s_max = max(s for s, _ in caps)
                s_min = min(s for s, _ in caps)
                thr = s_max - alpha_days * (s_max - s_min)
                rcl = [d for (s, d) in caps if s >= thr]
                d_pick = rng.choice(rcl)

            # Asignar si aún hay cupo (debería haber, pero chequeamos por seguridad)
            if day_capacity_left[d_pick] <= 0:
                # quitar este día y continuar (raro por la comprobación previa)
                # recomputará cands en el siguiente loop
                # Para evitar loop infinito, lo removemos de preferencias temporales:
                # (pero como options_now usa inst.days_by_employee, no mutamos datos de entrada)
                # Así que simplemente continuamos y se recalcularán candidatos
                # (si vuelve a salir, mates/caps podrían cambiar por otras asignaciones previas)
                # En la práctica no debería ocurrir.
                continue

            schedule_by_employee[e].add(d_pick)
            day_capacity_left[d_pick] -= 1
            if g_e is not None:
                group_pressure[d_pick][g_e] += 1
            need -= 1

    return schedule_by_employee, day_capacity_left

# ------------- FASE 3 ----------------
def phase3_randomized(
    inst, pre, 
    schedule_by_employee: Dict[str, Set[str]],
    zones_topH: int,      # elegir 1–2 zonas aleatorias dentro de las H mejores
    desks_topK: int,      # elegir escritorio al azar entre los k más “raros”
    group_jitter: float,    # jitter para orden de grupos del día
    seed: int,
    prefer_persistent: bool):
    
    rng = random.Random(seed)

    desks_by_employee = {e: set(v) for e, v in inst.desks_by_employee.items()}
    desks_by_zone     = {z: set(v) for z, v in inst.desks_by_zone.items()}
    employees_by_group = {g: set(v) for g, v in inst.employees_by_group.items()}
    zone_of = pre.zone_of_desk
    desks_all = set(inst.desks)

    # índice por día (desde Fase 2)
    employees_by_day: Dict[str, List[str]] = {d: [] for d in inst.days}
    for e, days in schedule_by_employee.items():
        for d in days:
            employees_by_day[d].append(e)

    persistent_desk: Dict[str, str] = {}
    assignments: List[Tuple[str, str, str]] = []

    for d in inst.days:
        todays = employees_by_day[d]
        if not todays:
            continue
        free_desks = set(desks_all)

        groups_today = defaultdict(list)
        # determinismo base: ordenar presentes por id
        for e in sorted(todays):
            g_e = next((g for g, mems in employees_by_group.items() if e in mems), None)
            groups_today[g_e].append(e)

        # grupos grandes primero + jitter
        base = [(-len(mems), rng.uniform(-group_jitter, group_jitter), g) for g, mems in groups_today.items()]
        groups_sorted = [g for *_, g in sorted(base)]

        for g in groups_sorted:
            members = sorted(groups_today[g])

            # puntuar zonas; tomar un pool top-H y elegir 1–2 al azar
            zone_scores = []
            for z in inst.zones:
                desks_z_free = free_desks & desks_by_zone[z]
                if not desks_z_free:
                    zone_scores.append((0, z)); continue
                count_fit = sum(1 for e in members if (desks_by_employee[e] & desks_z_free))
                zone_scores.append((count_fit, z))
            zone_scores.sort(reverse=True)
            H = min(zones_topH, len(zone_scores))
            top_pool = [z for _, z in zone_scores[:H]]
            if len(top_pool) >= 2:
                top_zones = rng.sample(top_pool, 2)
            elif len(top_pool) == 1:
                top_zones = top_pool
            else:
                top_zones = []

            # intento principal: persistencia + cohesión
            unassigned = []
            for e in members:
                chosen = None

                if prefer_persistent and e in persistent_desk:
                    p_d = persistent_desk[e]
                    if (p_d in free_desks) and (p_d in desks_by_employee[e]) and (zone_of.get(p_d) in top_zones):
                        chosen = p_d

                if chosen is None and top_zones:
                    cand = list(desks_by_employee[e] & free_desks & set().union(*(desks_by_zone[z] for z in top_zones)))
                    if cand:
                        def demand_in_group(desk):
                            return sum(1 for ee in members if desk in desks_by_employee[ee])
                        cand.sort(key=demand_in_group)        # de más raro a más popular
                        pick_pool = cand[:min(desks_topK, len(cand))]
                        chosen = rng.choice(pick_pool)

                if chosen is None:
                    unassigned.append(e)
                else:
                    assignments.append((e, d, chosen))
                    free_desks.remove(chosen)
                    if e not in persistent_desk:
                        persistent_desk[e] = chosen

            # fallback: fuera de top_zones
            for e in unassigned:
                cand = list(desks_by_employee[e] & free_desks)
                chosen = None

                if prefer_persistent and e in persistent_desk and persistent_desk[e] in cand:
                    chosen = persistent_desk[e]

                if chosen is None and cand:
                    def demand_today(desk):
                        return sum(1 for ee in todays if desk in desks_by_employee[ee])
                    cand.sort(key=demand_today)
                    pick_pool = cand[:min(desks_topK, len(cand))]
                    chosen = rng.choice(pick_pool)

                if chosen is None:
                    assignments.append((e, d, "none"))
                else:
                    assignments.append((e, d, chosen))
                    free_desks.remove(chosen)
                    if e not in persistent_desk:
                        persistent_desk[e] = chosen

    # --- post-paso: rellenar 'none' y stats (determinista dentro del día) ---
    forced_fill_total = 0
    forced_fill_incompatible_total = 0
    forced_fill_by_day = defaultdict(int)
    forced_fill_incompatible_by_day = defaultdict(int)
    forced_fill_employees: List[str] = []
    forced_fill_employees_by_day = defaultdict(list)
    forced_incompatible_employees: List[str] = []
    forced_incompatible_employees_by_day = defaultdict(list)

    idxs_by_day = defaultdict(list)
    for i, (_, d, _) in enumerate(assignments):
        idxs_by_day[d].append(i)

    desks_all_sorted = sorted(inst.desks)

    for d in inst.days:
        idxs = idxs_by_day.get(d, [])
        if not idxs:
            continue
        used = set()
        none_idxs = []
        for i in idxs:
            e_i, d_i, desk_i = assignments[i]
            if desk_i == "none":
                none_idxs.append(i)
            else:
                used.add(desk_i)

        free = [k for k in desks_all_sorted if k not in used]
        if len(free) < len(none_idxs):
            raise AssertionError(f"No hay suficientes escritorios libres en {d} para rellenar 'none'.")

        for i in none_idxs:
            chosen = free.pop(0)  # determinista
            e_i, d_i, _ = assignments[i]
            assignments[i] = (e_i, d_i, chosen)
            forced_fill_total += 1
            forced_fill_by_day[d_i] += 1
            forced_fill_employees.append(e_i)
            forced_fill_employees_by_day[d_i].append(e_i)
            if chosen not in desks_by_employee.get(e_i, set()):
                forced_fill_incompatible_total += 1
                forced_fill_incompatible_by_day[d_i] += 1
                forced_incompatible_employees.append(e_i)
                forced_incompatible_employees_by_day[d_i].append(e_i)
            if e_i not in persistent_desk:
                persistent_desk[e_i] = chosen

    forced_stats = {
        "forced_fill_total": forced_fill_total,
        "forced_fill_by_day": dict(forced_fill_by_day),
        "forced_fill_employees": forced_fill_employees,
        "forced_fill_employees_by_day": {d: emps for d, emps in forced_fill_employees_by_day.items()},
        "forced_fill_incompatible_total": forced_fill_incompatible_total,
        "forced_fill_incompatible_by_day": dict(forced_fill_incompatible_by_day),
        "forced_incompatible_employees": forced_incompatible_employees,
        "forced_incompatible_employees_by_day": {d: emps for d, emps in forced_incompatible_employees_by_day.items()},
    }
    
    # Dataframe de assignments (incluye zonas y grupos):
    df_assign = pd.DataFrame(assignments, columns=["Employee","Day","Desk"])
    # Para agregar el grupo
    group_col = df_assign['Employee'].map(pre.group_of_emp)
    df_assign.insert(0, 'Group', group_col)
    # Para agregar la zona
    pos = df_assign.columns.get_loc('Desk') + 1
    zone_col = df_assign['Desk'].map(pre.zone_of_desk)
    df_assign.insert(pos, 'Zone', zone_col)

    return assignments, df_assign, persistent_desk, forced_stats

# ======================= RANDOM MOVE PARA RECOCIDO SIMULADO ==============================

def get_random_move(sol: Solution, inst: ProblemInstance, pre: Precalc, weights: List[float]) -> Optional[Solution]:
    """
    Selecciona aleatoriamente un operador de vecindario, genera sus movimientos posibles,
    elige uno al azar y lo aplica.
    """
    # Lista de operadores disponibles en tools2.py
    neighborhoods = [
        neigh_swap_desks_same_day, # Intercambio de escritorios (Fase 3)
        neigh_insert_day,          # Cambio de día (Fase 2)
        neigh_insert_meeting_day,  # Cambio de GMD (Fase 1)
        neigh_swap_meeting_day     # Swap de GMD (Fase 1)
    ]
    
    # Intentamos hasta 3 veces encontrar un vecino válido (por si un vecindario está vacío)
    for _ in range(3):
        # 1. Elegir un vecindario al azar
        neigh_func = random.choice(neighborhoods)
        
        # 2. Generar TODOS los movimientos posibles de ese vecindario
        # Pasamos sol directamente
        moves = list(neigh_func(sol, inst, pre))
        
        if not moves:
            continue # Si este vecindario no tiene movimientos, probar otro
            
        # 3. Elegir un movimiento al azar
        move = random.choice(moves)
        
        # 4. Aplicar el movimiento (recibe sol, retorna sol)
        new_sol = apply_and_eval(sol, move, inst, pre, weights)
        
        if new_sol is not None:
            return new_sol

    return None # No se encontró un movimiento válido


# ================================= SHAKE PARA VNS ===================================

def shake_once(sol: Solution,
               inst, pre, weights,
               neigh_gen: Callable,
               rng: random.Random) -> Solution:
    """
    Aplica UN movimiento aleatorio factible del vecindario 'neigh_gen'.
    Si no encuentra ninguno factible, devuelve 'sol' sin cambios.
    """
    # Materializamos un pequeño pool de candidatos en orden determinista y luego aleatorizamos
    MAX_CANDS = 1000
    pool = []
    # neigh_gen ahora recibe sol directamente
    for i, mv in enumerate(neigh_gen(sol, inst, pre)):
        pool.append(mv)
        if i+1 >= MAX_CANDS:
            break
    if not pool:
        return sol

    rng.shuffle(pool)
    for mv in pool:
        cand_sol = apply_and_eval(sol, mv, inst, pre, weights)
        if cand_sol is not None:
            return cand_sol  # aplicamos el primero factible al azar
    return sol  # si ninguno fue factible, sin cambios

def shake(sol: Solution,
          inst, pre, weights,
          neigh_gen: Callable,
          strength: int,
          rng: random.Random) -> Solution:
    """
    Aplica 'strength' veces 'shake_once' usando SIEMPRE el mismo vecindario (k-ésimo).
    Acepta cambios aunque empeoren (propósito: salir del óptimo local).
    """
    s = sol
    for _ in range(max(1, strength)):
        s = shake_once(s, inst, pre, weights, neigh_gen, rng)
    return s


# ================================= METODOS PARA BRKGA ===================================

# 1) RANDOM-KEY LAYOUT (cómo mapeamos las keys al decoder)
@dataclass
class RKLayout:
    # Rangos (slices) dentro del vector x ∈ [0,1]^D
    # meetings[g] = (i0, i1) -> bloque de |days| keys para elegir orden de días del grupo g
    # emp_days[e] = (i0, i1) -> bloque de |days| keys para priorizar días del empleado e (Fase 2)
    # emp_desks[e] = (i0, i1) -> bloque de |desks| keys para priorizar escritorios del empleado e (tie-break en Fase 3)
    meetings: Dict[str, Tuple[int, int]]
    emp_days: Dict[str, Tuple[int, int]]
    emp_desks: Dict[str, Tuple[int, int]]
    D: int
    days: List[str]
    desks: List[str]
    employees: List[str]
    groups: List[str]

def build_rk_layout(inst) -> RKLayout:
    days = list(inst.days)
    desks = list(inst.desks)
    employees = list(inst.employees)
    groups = list(inst.groups)

    idx = 0
    meetings = {}
    for g in groups:
        meetings[g] = (idx, idx + len(days)); idx += len(days)

    emp_days = {}
    for e in employees:
        emp_days[e] = (idx, idx + len(days)); idx += len(days)

    emp_desks = {}
    for e in employees:
        emp_desks[e] = (idx, idx + len(desks)); idx += len(desks)

    return RKLayout(meetings, emp_days, emp_desks, idx, days, desks, employees, groups)

# 2) Función Decodificadora BRKGA 
def decode_rk_to_solution(
    rk: np.ndarray,
    inst: ProblemInstance,
    pre: Precalc,
    layout: RKLayout,
    target_days_default: int,
    weights: List[float]
    ) -> Solution:
    
    # -----------------------------------------------------
    # 1. Mapeo de Random Keys a Prioridades de Ordenamiento
    # -----------------------------------------------------
    
    num_groups = len(inst.groups)
    
    # a) Prioridad de Grupos (para Phase 1)
    # Se usa el primer bloque de RKs para ordenar los grupos.
    rk_groups = rk[:num_groups]
    group_priority_list = sorted(
        zip(rk_groups, inst.groups), 
        key=lambda x: x[0] # Ordenar por el valor del RK (menor RK = mayor prioridad)
    )
    # Lista de grupos en orden de prioridad del cromosoma:
    ordered_groups = [g for rk_val, g in group_priority_list]
    
    # b) Prioridad de Empleados (para Phase 2)
    rk_employees = rk[num_groups:]
    employee_priority_list = sorted(
        zip(rk_employees, inst.employees),
        key=lambda x: x[0]
    )
    # Lista de empleados en orden de prioridad del cromosoma:
    ordered_employees = [e for rk_val, e in employee_priority_list]
    
    
    # ====================================================
    # 2. FASE 1 MODIFICADA: Asignación de Día de Reunión (GMD)
    #    (Mantiene tu lógica greedy, pero con el orden RK)
    # ====================================================
    
    days_by_employee = {e: set(v) for e, v in inst.days_by_employee.items()}
    employees_by_group = {g: set(v) for g, v in inst.employees_by_group.items()}
    desk_count = len(inst.desks)
    day_capacity_left = {d: desk_count for d in inst.days}
    
    # Cache de cohesión (igual que en tu phase1)
    def cluster(g):
        # ... (código de tu cluster aquí, o asume que cluster_cache está disponible)
        counts = []
        for z in inst.zones:
            c = sum(1 for e in employees_by_group[g] if pre.compat_in_zone.get(e, {}).get(z, 0) > 0)
            counts.append(c)
        counts.sort(reverse=True)
        return ((counts[0] if counts else 0) + (counts[1] if len(counts) > 1 else 0)) / max(1, pre.group_size[g])
    cluster_cache = {g: cluster(g) for g in inst.groups}

    def coverage(g, d):
        return sum(1 for e in employees_by_group[g] if d in days_by_employee[e]) / max(1, pre.group_size[g])
    
    def day_score(g, d):
        """Score para ordenar días (sin violar capacidad)."""
        # ... (código de tu day_score aquí)
        cov = coverage(g, d)
        slack = day_capacity_left[d] / max(1, desk_count)
        coh = cluster_cache[g]
        is_common = 1 if d in set(pre.common_days_group.get(g, [])) else 0
        score = (weights[0]*cov + weights[1]*slack + weights[2]*coh + 0.05*is_common)
        # tupla con desempates deterministas
        return (is_common, score, cov, slack, coh, -pre.group_size[g])

    group_meeting_day = {}
    warnings = [] # Advertencias se ignoran en el decodificador, pero se mantienen

    # Bucle ahora usa el orden RK:
    for g in ordered_groups: 
        size_g = pre.group_size[g]
        commons = set(pre.common_days_group.get(g, []))

        # El resto de la lógica greedy es la misma:
        feasible_common = [d for d in inst.days if d in commons and day_capacity_left[d] >= size_g]
        feasible_noncommon = [d for d in inst.days if d not in commons and day_capacity_left[d] >= size_g]

        chosen = None
        # (Lógica de selección del día con sort y day_score, igual a tu phase1)
        if feasible_common:
            feasible_common.sort(key=lambda d: day_score(g, d), reverse=True)
            chosen = feasible_common[0]
        elif feasible_noncommon:
            feasible_noncommon.sort(key=lambda d: day_score(g, d), reverse=True)
            chosen = feasible_noncommon[0]
        else:
            warnings.append(f"[{g}] INVIABLE en Fase 1 por falta de capacidad.")
            continue

        group_meeting_day[g] = chosen
        day_capacity_left[chosen] -= size_g # Descuenta capacidad


    # ====================================================
    # 3. FASE 2 MODIFICADA: Completar Días Extra 
    # ====================================================

    # --------- inicializaciones ---------
    schedule_by_employee = {e: set() for e in inst.employees}
    group_pressure = {d: Counter() for d in inst.days}
    
    # Paso 1 MODIFICADO: meeting day OBLIGATORIO
    # Ya usamos group_meeting_day para inicializar day_capacity_left en Phase 1, 
    # pero necesitamos restaurar day_capacity_left y actualizar group_pressure.
    
    # Restauramos capacidad con el resultado de Phase 1 para que Phase 2 lo pueda usar
    day_capacity_left = {d: desk_count for d in inst.days}
    
    # Forzamos la asistencia al GMD y actualizamos contadores
    for g in sorted(inst.groups, key=lambda x: -pre.group_size[x]): # Orden por tamaño (o usa el RK order aquí también)
        d_meet = group_meeting_day.get(g)
        if d_meet is None: continue # Grupo inviable en Phase 1
            
        members = employees_by_group[g]
        for e in members:
            # Esta asignación SÍ pasa, porque la capacidad ya fue garantizada en Phase 1
            schedule_by_employee[e].add(d_meet)
            day_capacity_left[d_meet] -= 1
            group_pressure[d_meet][g] += 1
            
    
    # 🚨 Paso 2 MODIFICADO: completar días extra por empleado 🚨

    def need_of(e):
        return max(0, target_days_default - len(schedule_by_employee[e]))

    # Solo empleados que necesitan días extra
    employees_to_assign = [e for e in inst.employees if need_of(e) > 0]

    # CLAVE: Iteramos sobre los empleados en el ORDEN dado por el RK,
    # ignorando la lógica de 'apretados primero' de tu phase2 original.
    for e in ordered_employees:
        if e not in employees_to_assign:
            continue
            
        g_e = pre.group_of_emp.get(e)
        if g_e is None:
            continue

        # Recalcular necesidad
        need = need_of(e)
        if need <= 0:
            continue

        # Candidatos: (código de tu phase2 original)
        candidates = []
        for d in inst.days:
            if d in schedule_by_employee[e]:
                continue
            # Solo consideramos preferencias del empleado (d in days_by_employee)
            if d not in days_by_employee.get(e, set()): 
                continue
            if day_capacity_left[d] <= 0:
                continue
            
            mates = group_pressure[d][g_e]
            candidates.append((mates, day_capacity_left[d], d))

        # Orden de asignación greedy (más compañeros, luego más capacidad),
        # que es la lógica original de tu phase2.
        candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)

        # Asignar hasta cumplir 'need'
        for mates, cap, d in candidates:
            if need <= 0:
                break
            # Verificación de seguridad
            if day_capacity_left[d] <= 0:
                continue 
            schedule_by_employee[e].add(d)
            day_capacity_left[d] -= 1
            group_pressure[d][g_e] += 1
            need -= 1
    
    
    # ====================================================
    # 4. FASE 3: Asignación de Escritorios (Desk)
    # ====================================================

    # La Phase 3 (de tools.py) utiliza el `schedule_by_employee` resultante,
    # que ya fue influenciado por los RKs en Phase 1 y Phase 2.
    # No es necesario modificar phase3, ya que la entrada es diferente.
    #assignments, df_assign, persistent_desk, forced_stats = tools.phase3(
        #inst, pre, schedule_by_employee, prefer_persistent=True
    #)

    assignments, df_assign, persistent_desk, forced_stats = phase3(
        inst, pre, schedule_by_employee, 
        prefer_persistent=True, 
        rk_employee_order=ordered_employees # <-- NUEVO ARGUMENTO
    )

    # -----------------------------------------------------
    # 5. Creación de la Solución Final
    # -----------------------------------------------------
    
    VA = count_valid_assignments(inst, assignments)
    EP = count_employee_preferences(inst, schedule_by_employee)
    IE = count_isolated_employees(df_assign)
    
    sol = Solution(
        group_meeting_day = group_meeting_day,
        schedule_by_employee = schedule_by_employee,
        assignments = assignments,
        df_assign = df_assign,
        forced_stats = forced_stats,
        valid_assingments = VA,
        employee_preferences = EP,
        isolated_employees = IE,
        score = 0.0)
    
    sol.score = evaluate_sol(inst, sol, weights)
    
    return sol

# 3) Clase con parametros del BRKGA
@dataclass
class BRKGAParams:
    pop_size: int              # P: Tamaño de la población
    gens: int                  # G: Número de generaciones
    p_elite_frac: float        # Pe: Fracción de la población que es "élite"
    p_crossover_bias: float    # Pc: Prob. de heredar el gen del padre de élite
    p_mutant_frac: float       # Pm: Fracción de la población que son "mutantes" (nuevos aleatorios)
    target_days_default: int   # Parámetro para tu decodificador
    #seed: int = 2026
    seed: Optional[int]
    weights: List[float]