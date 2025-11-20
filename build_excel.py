from dataclasses import dataclass
from typing import Dict, Set, List, Tuple
import pandas as pd
from precalculos import Precalc
from read_instances import ProblemInstance
from collections import defaultdict, Counter
import numpy as np
from tools import count_valid_assignments, count_employee_preferences, count_isolated_employees


@dataclass
class export_results:
    # Outputs
    df_assign: pd.DataFrame  
    df_groups: pd.DataFrame
    df_summary: pd.DataFrame 


def build_outputs(inst: ProblemInstance, pre: Precalc, group_meeting_day: Dict[str, str], schedule_by_employee: Dict[str, Set[str]], 
                  assignments: List[Tuple[str, str, str]], df_assign: pd.DataFrame) -> export_results:    
    """
    Construye los tres DataFrames exigidos por la plantilla:
      1) EmployeeAssignment (ANCHO): columnas = ['Employee'] + list(inst.days)
         Valor = desk si (employee, day) asignado; 'None' si no asiste ese día.
      2) Groups Meeting day: (Group, Day)
      3) Summary:
         - Valid assignments = # (e,d) con Desk != 'none' y compatible (desk ∈ desks_by_employee[e])
         - Employee preferences = # (e,d) de Fase 2 donde d ∈ days_by_employee[e]
         - Isolated employees = suma empleado–día donde e es el único de su grupo asistiendo ese día
    """


    # ================= EmployeeAssignment (ANCHO) =================
    # Mapa (Employee, Day) -> Desk
    # Si hubiera duplicados (no debería), nos quedamos con el último
    ed_to_desk = {}
    for e, d, desk in assignments:
        ed_to_desk[(e, d)] = str(desk)

    # Construimos filas: una por empleado, columnas por día
    rows = []
    day_list = list(inst.days)  # respeta el orden definido en la instancia
    for e in sorted(inst.employees):
        row = {"Employee": e}
        for d in day_list:
            row[d] = ed_to_desk.get((e, d), "None")
        rows.append(row)

    df_assign_wide = pd.DataFrame(rows, columns=["Employee"] + day_list)

    # ================= Groups Meeting day =================
    df_groups = pd.DataFrame(
        [(g, d) for g, d in group_meeting_day.items()],
        columns=["Group", "Day"]
    ).sort_values("Group")

    # ================= Summary =================
    # -- 1) Valid assignments (desde 'assignments' largos)
    valid_assignments = count_valid_assignments(inst, assignments)

    # -- 2) Employee preferences (desde schedule_by_employee)
    employee_preferences = count_employee_preferences(inst, schedule_by_employee)

    # -- 3) Isolated employees (suma empleado–día): fuera del núcleo (1–2 zonas sucesivas) del grupo ese día
    isolated_employees = count_isolated_employees(df_assign)

    df_summary = pd.DataFrame([{
        "Valid assignments": int(valid_assignments),
        "Employee preferences": int(employee_preferences),
        "Isolated employees": int(isolated_employees),
    }])

    return export_results(
        df_assign=df_assign_wide,
        df_groups=df_groups,
        df_summary=df_summary
    )


def export_solution_excel(path, df_assign, df_groups, df_summary):
    """
    Exporta a Excel con las TRES hojas con nombres EXACTOS:
      - 'EmployeeAssignment'  (ANCHO)
      - 'Groups Meeting day'
      - 'Summary'
    """
    # Fallback de motor de Excel
    try:
        writer = pd.ExcelWriter(path, engine="xlsxwriter")
    except ModuleNotFoundError:
        writer = pd.ExcelWriter(path, engine="openpyxl")

    with writer as w:
        df_assign.to_excel(w, index=False, sheet_name="EmployeeAssignment")
        df_groups.to_excel(w, index=False, sheet_name="Groups Meeting day")
        df_summary.to_excel(w, index=False, sheet_name="Summary")

        # Autoajuste simple
        for sheet_name, df in {
            "EmployeeAssignment": df_assign,
            "Groups Meeting day": df_groups,
            "Summary": df_summary
        }.items():
            ws = w.sheets[sheet_name]
            for i, col in enumerate(df.columns):
                width = max(12, min(40, int(df[col].astype(str).map(len).max() if not df.empty else 12) + 2))
                ws.set_column(i, i, width)

