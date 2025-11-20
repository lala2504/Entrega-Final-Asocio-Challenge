import random
import time
import tools
import math
from typing import Callable, List, Optional, Tuple
from read_instances import ProblemInstance
from precalculos import Precalc
import numpy as np

# ========================= CONSTRUCTIVO =========================

def constructivo(inst: ProblemInstance, pre: Precalc, weights: List[float]) -> tools.Solution:
    """Constructivo original (greedy determinista)."""
    gmd, _warns = tools.phase1(inst, pre)
    sched, _ = tools.phase2(inst, pre, gmd, target_days_default=2)
    assigns, df_assign, persist, fstats = tools.phase3(inst, pre, sched, prefer_persistent=True)
    
    VA = tools.count_valid_assignments(inst, assigns)
    EP = tools.count_employee_preferences(inst, sched)
    IE = tools.count_isolated_employees(df_assign)
    
    sol = tools.Solution(
        group_meeting_day = gmd,
        schedule_by_employee = sched,
        assignments = assigns,
        df_assign = df_assign,
        forced_stats = fstats,
        valid_assingments = VA,
        employee_preferences = EP,
        isolated_employees = IE,
        score = 0.0)
    sol.score = tools.evaluate_sol(inst,sol,weights)
    return sol


# ===================== CONSTRUCTIVO ALEATORIZADO =====================

def constructivo_aleatorizado(inst: ProblemInstance, pre: Precalc, weights: List[float]) -> tools.Solution:
    """Constructivo aleatorizado (greedy randomized construction, estilo GRASP)."""
    gmd, _warns = tools.phase1_randomized(inst, pre,
                                           weights=(0.6, 0.2, 0.2),
                                           alpha=1, group_jitter=1, seed=42)
    sched, _ = tools.phase2_randomized(inst, pre, gmd, target_days_default=2,
                                        alpha_days=0.5, tie_jitter=0.5, seed=42)
    assigns, df_assign, persist, fstats = tools.phase3_randomized(inst, pre, sched,
                                                                   zones_topH=4, desks_topK=2, group_jitter=1, seed=42,
                                                                   prefer_persistent=True)
    
    VA = tools.count_valid_assignments(inst, assigns)
    EP = tools.count_employee_preferences(inst, sched)
    IE = tools.count_isolated_employees(df_assign)
    
    sol = tools.Solution(
        group_meeting_day = gmd,
        schedule_by_employee = sched,
        assignments = assigns,
        df_assign = df_assign,
        forced_stats = fstats,
        valid_assingments = VA,
        employee_preferences = EP,
        isolated_employees = IE,
        score = 0.0)
    sol.score = tools.evaluate_sol(inst,sol,weights)
    return sol


# ========================= RECOCIDO SIMULADO =============================

def simulated_annealing(
    inst, # Tipo: ProblemInstance
    pre,  # Tipo: Precalc
    initial_sol: tools.Solution,
    weights: List[float],
    t_init: float = 100.0,
    alpha: float = 0.99,
    max_iter: int = 5000,
    max_time_seconds: int = 60
) -> Tuple[tools.Solution, List[float]]: # El tipo de retorno refleja (Solution, history)
    """
    Algoritmo de Recocido Simulado para maximizar el Score.
    """
    
    # 1. Inicialización
    # Ya no se necesita tools.State. current_sol y best_sol son objetos Solution.
    current_sol = initial_sol
    best_sol = initial_sol
    
    T = t_init
    start_time = time.time()
    
    #print(f"--- Inicio SA ---")
    #print(f"Score Inicial: {current_sol.score:.2f}") # Acceso directo a .score
    
    history = []

    # 2. Bucle Principal
    for i in range(max_iter):
        # Chequeo de tiempo
        if time.time() - start_time > max_time_seconds:
            print(f"Tiempo límite alcanzado en iteración {i}")
            break
            
        # 3. Obtener vecino aleatorio
        # tools.get_random_move ahora recibe y retorna Solution
        neighbor_sol = tools.get_random_move(current_sol, inst, pre, weights)
        
        if neighbor_sol is None:
            continue # Movimiento inválido, siguiente iteración
            
        # 4. Calcular Delta (Maximización: Nuevo - Actual)
        delta = neighbor_sol.score - current_sol.score # Acceso directo a .score
        
        # 5. Criterio de Aceptación
        accept = False
        if delta > 0:
            accept = True # Siempre aceptar mejoras
        else:
            # Aceptar peores soluciones con probabilidad e^(delta/T)
            # (delta es negativo aquí)
            prob = math.exp(delta / T)
            if random.random() < prob:
                accept = True
        
        if accept:
            current_sol = neighbor_sol # Actualizar la solución actual
            # Actualizar mejor global
            if current_sol.score > best_sol.score:
                best_sol = current_sol # Actualizar la mejor solución global
                #print(f"Iter {i}: Nuevo Mejor Score = {best_sol.score:.2f} (T={T:.2f})")
        
        # 6. Enfriamiento
        T *= alpha
        
        # Guardar historia para gráficas
        history.append(current_sol.score)
        
        # Reiniciar temperatura si se congela (Reheating simple)
        if T < 0.01:
            T = t_init * 0.5

    #print(f"--- Fin SA ---")
    #print(f"Mejor Score Final: {best_sol.score:.2f}")
    
    # Retorna la mejor Solution y la historia.
    return best_sol, history


# ========================= BUSQUEDA LOCAL =============================

def local_search_FI(sol: tools.Solution, inst, pre, weights,
                    neighborhoods: List) -> tools.Solution:
    """
    Recorre vecindarios en orden; acepta el primer vecino que mejore; reinicia.
    """
    improved = True
    while improved:
        improved = False
        for gen in neighborhoods:
            for move in gen(sol, inst, pre):
                cand: Optional[tools.Solution] = tools.apply_and_eval(sol, move, inst, pre, weights)
                if cand and cand.score > sol.score:
                    sol = cand
                    improved = True
                    break
            if improved:
                break
    return sol

def local_search_BI(sol: tools.Solution, inst, pre, weights,
                    neighborhoods: List) -> tools.Solution:
    """
    Enumera cada vecindario completo y toma el mejor; si mejora, reinicia desde el 1º.
    """
    while True:
        any_improvement = False
        for gen in neighborhoods:
            best_neighbor: Optional[tools.Solution] = None
            
            for move in gen(sol, inst, pre):
                cand: Optional[tools.Solution] = tools.apply_and_eval(sol, move, inst, pre, weights)
                # Compara con el mejor vecino (best_neighbor)
                if cand and (best_neighbor is None or cand.score > best_neighbor.score):
                    best_neighbor = cand        
            # Compara el mejor vecino con la solución actual (sol)
            if best_neighbor and best_neighbor.score > sol.score:
                sol = best_neighbor
                any_improvement = True
                break  # reinicia vecindarios
        if not any_improvement:
            return sol

# ================================ VNS ===================================

def vns_search(inst, pre,
               initial_sol: tools.Solution, # Cambiado de initial_state a initial_sol
               weights,
               neighborhoods: List[Callable],
               improve_method: str = "FI",
               k_max: Optional[int] = None,
               shake_strengths: Optional[List[int]] = None,
               max_outer_iters: int = 100,
               seed: int = 0) -> tools.Solution:
    """
    VNS (Variable Neighborhood Search)
    """
    rng = random.Random(seed)
    if k_max is None:
        k_max = len(neighborhoods)
    assert 1 <= k_max <= len(neighborhoods)

    if shake_strengths is None:
        shake_strengths = list(range(1, k_max + 1))  # [1,2,3,...]

    # Selección de política de mejora
    if improve_method.upper() == "FI":
        improve = local_search_FI
    elif improve_method.upper() == "BI":
        improve = local_search_BI
    else:
        raise ValueError("improve_method debe ser 'FI' o 'BI'.")

    best: tools.Solution = initial_sol
    outer = 0
    while outer < max_outer_iters:
        outer += 1
        k = 1
        improved_any = False

        while k <= k_max:
            neigh_k = neighborhoods[k - 1]
            strength_k = shake_strengths[min(k - 1, len(shake_strengths) - 1)]

            # 1) SHAKE en el vecindario k
            s_shaken: tools.Solution = tools.shake(best, inst, pre, weights, neigh_k, strength_k, rng)

            # 2) BÚSQUEDA LOCAL desde s_shaken
            s_loc: tools.Solution = improve(s_shaken, inst, pre, weights, neighborhoods)

            # 3) Movida de aceptación VNS: si mejoró el mejor global, adoptamos y reiniciamos k
            if s_loc.score > best.score:
                best = s_loc
                improved_any = True
                k = 1  # reinicia desde el vecindario más suave
            else:
                k += 1  # probar un vecindario más fuerte

        # Si tras pasar por todos los vecindarios no hubo mejora, podemos terminar
        if not improved_any:
            break
        
    return best

# ================================ BRKGA ===================================

def brkga_optimize(inst, pre, params: tools.BRKGAParams) -> Tuple[tools.Solution, np.ndarray]:
    """
    Ejecuta el algoritmo BRKGA.
    Retorna: (best_sol: tools.Solution, best_chromosome: np.ndarray)
    """
    
    rng = np.random.default_rng(params.seed)
    random.seed(params.seed)

    # 1. Obtenemos el layout y la dimensión del cromosoma (D)
    layout = tools.build_rk_layout(inst) # (Función de tu Celda 13)
    D = layout.D
    
    # 2. Definimos los tamaños de los grupos
    P = params.pop_size
    n_elite = int(P * params.p_elite_frac)
    n_crossover = P - n_elite - int(P * params.p_mutant_frac)
    n_mutants = P - n_elite - n_crossover 

    if n_elite <= 0 or n_crossover <= 0:
        raise ValueError("El tamaño de la población (pop_size) es muy pequeño o p_elite_frac es muy alto.")

    # 3. Inicialización: P cromosomas aleatorios en [0,1]^D
    pop_X = rng.random((P, D))
    pop_scores = np.zeros(P)
    best_sol: Optional[tools.Solution] = None
    best_chromosome: Optional[np.ndarray] = None

    # Evaluación de la población inicial (Población 0)
    for i in range(P):
        sol = tools.decode_rk_to_solution(pop_X[i], inst, pre, layout, 
                                           target_days_default=params.target_days_default,
                                           weights=params.weights)
        pop_scores[i] = float(sol.score) # Guardamos el score para la clasificación
        if best_sol is None or pop_scores[i] > best_sol.score:
            best_sol = sol
            best_chromosome = pop_X[i]

    # 4. Bucle Generacional
    for g in range(params.gens):
        
        # 5. Clasificación:
        # Obtenemos los índices de la población ordenados por score (descendente)
        sorted_indices = np.argsort(pop_scores)[::-1]
        
        elite_indices = sorted_indices[:n_elite]
        non_elite_indices = sorted_indices[n_elite:]

        # La nueva población
        next_pop_X = np.zeros((P, D))
        next_pop_scores = np.zeros(P)

        # 6. Evolución
        
        # a) Elitismo: Los 'n_elite' mejores pasan directamente
        next_pop_X[:n_elite] = pop_X[elite_indices]
        next_pop_scores[:n_elite] = pop_scores[elite_indices]
        
        # b) Cruce (Mating): Generamos 'n_crossover' hijos
        for i in range(n_crossover):
            # Seleccionamos un padre de la élite y otro de la no-élite
            idx_parent_e = rng.choice(elite_indices)
            idx_parent_n = rng.choice(non_elite_indices)
            
            parent_e = pop_X[idx_parent_e]
            parent_n = pop_X[idx_parent_n]
            
            # Cruce sesgado (Biased Crossover)
            child = np.zeros(D)
            inherit_mask = rng.random(D) <= params.p_crossover_bias
            child = np.where(inherit_mask, parent_e, parent_n)  # True si hereda del élite, False del no-élite
            
            # Guardar el hijo y su score
            sol_child = tools.decode_rk_to_solution(
                child, inst, pre, layout, 
                target_days_default=params.target_days_default,
                weights=params.weights
                )
            next_pop_X[n_elite + i] = child
            next_pop_scores[n_elite + i] = float(sol_child.score)
            
            # Actualizar el mejor global (usando sol_child.score)
            if sol_child.score > best_sol.score:
                best_sol = sol_child
                best_chromosome = child

        # c) Mutantes: Generamos 'n_mutants' individuos aleatorios
        for i in range(n_mutants):
            mutant_X = rng.random(D)
            sol_mutant = tools.decode_rk_to_solution(
                            mutant_X, inst, pre, layout, 
                            target_days_default=params.target_days_default,
                            weights=params.weights
                        )     
    
            idx = n_elite + n_crossover + i
            next_pop_X[idx] = mutant_X
            next_pop_scores[idx] = float(sol_mutant.score)

            # Actualizar el mejor global 
            if sol_mutant.score > best_sol.score:
                best_sol = sol_mutant
                best_chromosome = mutant_X
        
        # La nueva generación reemplaza a la antigua
        pop_X = next_pop_X
        pop_scores = next_pop_scores
        
        # (Opcional) Imprimir progreso
        if (g + 1) % 20 == 0:
            print(f"Generación {g+1}/{params.gens} | Mejor Score: {best_sol.score:.2f}")

    # Retornamos la mejor Solution y su cromosoma asociado
    return best_sol, best_chromosome