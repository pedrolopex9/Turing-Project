import sys
import subprocess
import numpy as np
import json
import argparse
import re
import time
import random
import os

# Força UTF-8 na saída padrão (Python 3.7+)
sys.stdout.reconfigure(encoding='utf-8')

def log_debug(msg):
    # Log sem acentos para evitar crash no Windows console
    clean_msg = msg.encode('ascii', 'ignore').decode('ascii')
    print(f"DEBUG: {clean_msg}", file=sys.stdout, flush=True)

# ⭐️ ATUALIZAÇÃO DO ARGPARSE: Adiciona 'simplex' e 'battle' completo
parser = argparse.ArgumentParser()
parser.add_argument("--exe", required=True)
parser.add_argument("--method", choices=["pso", "ga", "simplex", "battle"], default="pso")
parser.add_argument("--dim", type=int, default=9)
parser.add_argument("--pop", type=int, default=20)
parser.add_argument("--iter", type=int, default=50)
parser.add_argument("--goal", default="min")
args = parser.parse_args()

# --- VALIDAÇÃO DO EXECUTÁVEL ---
if not os.path.exists(args.exe):
    log_debug(f"ERRO FATAL: O arquivo '{args.exe}' nao foi encontrado na pasta!")
    log_debug(f"Pasta atual: {os.getcwd()}")
    # Envia erro JSON para o front saber
    print(json.dumps({"erro": "Executavel nao encontrado"}), flush=True)
    sys.exit(1)
else:
    log_debug(f"Executavel encontrado: {args.exe}")

log_debug(f"Config: {args.method}, Dim: {args.dim}")

# --- EXECUÇÃO ---
def run_blackbox(params):
    try:
        # A função run_blackbox é idêntica à original
        cmd = [os.path.abspath(args.exe), "medio"] + [str(int(p)) for p in params]
        
        # Timeout curto para não travar a interface
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3, encoding='latin-1')
        
        # Se o executável falhar
        if result.returncode != 0:
            return float('inf')

        output = (result.stdout + result.stderr).replace(',', '.')
        
        nums = re.findall(r"[-+]?\d*\.?\d+", output)
        if nums:
            val = float(nums[-1])
            return val if args.goal == "min" else -val
            
        return float('inf')
    except Exception as e:
        # Não logar erro a cada tentativa para não sujar o console, retorna inf
        return float('inf')

# --- ALGORITMOS ---
# --- PSO (EXISTENTE) ---
def run_pso_step(particles, velocities, pbest, pbest_val, gbest, gbest_val):
    current_vals = []
    w, c1, c2 = 0.7, 1.4, 1.4

    for idx, p in enumerate(particles):
        val = run_blackbox(p)
        current_vals.append(val)

        if val < pbest_val[idx]:
            pbest_val[idx] = val
            pbest[idx] = p.copy()
            if val < gbest_val:
                gbest_val = val
                gbest = p.copy()

    # Atualiza posições
    r1, r2 = np.random.rand(args.pop, 1), np.random.rand(args.pop, 1)
    velocities = w * velocities + c1 * r1 * (pbest - particles) + c2 * r2 * (gbest - particles)
    particles = np.round(particles + velocities)
    particles = np.clip(particles, 1, 100)

    return particles, velocities, pbest, pbest_val, gbest, gbest_val, np.mean(current_vals)

# --- GA (EXISTENTE) ---
def run_ga_step(population):
    fitnesses = []
    for ind in population:
        val = run_blackbox(ind)
        fitnesses.append(val)
    
    min_fit = min(fitnesses)
    best_ind = population[fitnesses.index(min_fit)].copy()
    
    # Cria nova população
    new_pop = []
    new_pop.append(best_ind) # Elitismo
    
    while len(new_pop) < args.pop:
        # Torneio simples
        idx1, idx2 = random.randint(0, args.pop-1), random.randint(0, args.pop-1)
        parent1 = population[idx1] if fitnesses[idx1] < fitnesses[idx2] else population[idx2]
        
        idx3, idx4 = random.randint(0, args.pop-1), random.randint(0, args.pop-1)
        parent2 = population[idx3] if fitnesses[idx3] < fitnesses[idx4] else population[idx4]
        
        # Crossover
        cut = random.randint(1, args.dim-1)
        child = np.concatenate((parent1[:cut], parent2[cut:]))
        
        # Mutação
        if random.random() < 0.2:
            m_idx = random.randint(0, args.dim-1)
            child[m_idx] = random.randint(1, 100)
            
        new_pop.append(child)
        
    return np.array(new_pop), best_ind, min_fit, np.mean(fitnesses)

# ⭐️ NOVO ALGORITMO: SIMPLEX DE NELDER-MEAD
def run_simplex_step(simplex_points):
    # Simplex é um conjunto de N+1 pontos, onde N é args.dim
    
    # 1. Avalia o Simplex
    fitnesses = [(p, run_blackbox(p)) for p in simplex_points]
    
    # 2. Ordena os pontos: melhor (low), penúltimo (next_high), pior (high)
    fitnesses.sort(key=lambda x: x[1])
    
    p_low, f_low = fitnesses[0]
    p_high, f_high = fitnesses[-1]
    
    # Se o melhor valor for infinito, ou primeira iteração, retorna inalterado
    if f_low == float('inf'):
        # Usamos o primeiro ponto para evitar erros no centroid
        return simplex_points, float('inf'), float('inf') 

    # 3. Calcula o centroide (exceto o pior ponto)
    centroid = np.mean([p for p, f in fitnesses[:-1]], axis=0)
    
    # 4. Reflexão (alpha = 1.0)
    p_reflect = centroid + 1.0 * (centroid - p_high)
    p_reflect = np.clip(np.round(p_reflect), 1, 100)
    f_reflect = run_blackbox(p_reflect)

    # Verifica se a reflexão é melhor que o penúltimo, mas não o melhor
    if f_low <= f_reflect < fitnesses[-2][1]: 
        # Substitui o pior ponto pelo ponto refletido
        new_simplex = [p for p, f in fitnesses[:-1]] + [p_reflect]
        return np.array(new_simplex), f_low, np.mean([f for p, f in fitnesses])

    elif f_reflect < f_low: # Reflexão é o novo melhor ponto -> Tenta Expansão
        # 5. Expansão (gamma = 2.0)
        p_expand = centroid + 2.0 * (centroid - p_high)
        p_expand = np.clip(np.round(p_expand), 1, 100)
        f_expand = run_blackbox(p_expand)

        if f_expand < f_reflect:
            new_simplex = [p for p, f in fitnesses[:-1]] + [p_expand]
            return np.array(new_simplex), f_expand, np.mean([f for p, f in fitnesses])
        else:
            new_simplex = [p for p, f in fitnesses[:-1]] + [p_reflect]
            return np.array(new_simplex), f_low, np.mean([f for p, f in fitnesses])

    else: # f_reflect >= fitnesses[-2][1] -> Contração
        
        # 6. Contração (Beta = 0.5)
        if f_reflect < f_high: # Contração Externa
            p_contract = centroid + 0.5 * (p_reflect - centroid)
        else: # Contração Interna (pior que o ponto mais alto, usa o ponto mais alto)
            p_contract = centroid - 0.5 * (centroid - p_high)

        p_contract = np.clip(np.round(p_contract), 1, 100)
        f_contract = run_blackbox(p_contract)

        if f_contract < min(f_reflect, f_high): # Aceita a contração
            new_simplex = [p for p, f in fitnesses[:-1]] + [p_contract]
            return np.array(new_simplex), f_low, np.mean([f for p, f in fitnesses])

        # 7. Redução (Encolhimento)
        # Nada melhorou, encolhe o simplex em direção ao melhor ponto (p_low)
        new_simplex = [p_low]
        for p, f in fitnesses[1:]:
            p_new = p_low + 0.5 * (p - p_low)
            new_simplex.append(np.clip(np.round(p_new), 1, 100))
        
        # O melhor ponto é mantido
        return np.array(new_simplex), f_low, np.mean([f for p, f in fitnesses])


def main():
    # Setup Inicial (PSO e GA)
    pso_parts = np.random.randint(1, 100, (args.pop, args.dim))
    pso_vel = np.zeros_like(pso_parts)
    pso_pbest = pso_parts.copy()
    pso_pbest_val = [float('inf')] * args.pop
    pso_gbest = pso_parts[0]
    pso_gbest_val = float('inf')

    ga_pop = np.random.randint(1, 100, (args.pop, args.dim))
    ga_best_val = float('inf')
    
    # ⭐️ Setup Simplex (N+1 pontos)
    simplex_pop = np.random.randint(1, 100, (args.dim + 1, args.dim))
    simplex_best_val = float('inf')

    log_debug("Iniciando loop principal...")

    for i in range(args.iter):
        data_out = {"iteracao": i + 1}
        
        # Executa PSO
        if args.method in ["pso", "battle"]:
            pso_parts, pso_vel, pso_pbest, pso_pbest_val, pso_gbest, pso_gbest_val, pso_mean = \
                run_pso_step(pso_parts, pso_vel, pso_pbest, pso_pbest_val, pso_gbest, pso_gbest_val)
            
            real_best = pso_gbest_val if args.goal == "min" else -pso_gbest_val
            data_out["pso_best"] = real_best if real_best != float('inf') else 0

        # Executa GA
        if args.method in ["ga", "battle"]:
            ga_pop, ga_best_ind, ga_iter_best, ga_mean = run_ga_step(ga_pop)
            if ga_iter_best < ga_best_val:
                ga_best_val = ga_iter_best
            
            real_best_ga = ga_best_val if args.goal == "min" else -ga_best_val
            data_out["ga_best"] = real_best_ga if real_best_ga != float('inf') else 0

        # ⭐️ Executa Simplex
        if args.method in ["simplex", "battle"]:
            simplex_pop, simplex_iter_best, simplex_mean = run_simplex_step(simplex_pop)
            
            if simplex_iter_best < simplex_best_val:
                simplex_best_val = simplex_iter_best
            
            real_best_simplex = simplex_best_val if args.goal == "min" else -simplex_best_val
            data_out["simplex_best"] = real_best_simplex if real_best_simplex != float('inf') else 0

        # Imprime JSON
        print(json.dumps(data_out), flush=True)

    log_debug("Finalizado com sucesso.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_debug(f"ERRO CRITICO NO MAIN: {e}")
