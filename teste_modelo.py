# particle_swarm_fixed_v2.py
import random
import numpy as np
import subprocess
import re

def run_executable_int(exe_path, params):
    """
    Executa o programa com parâmetros - primeiro parâmetro é string, os demais são inteiros.
    """
    try:
        # Converte os parâmetros: o primeiro mantém como string, os demais como inteiros
        str_params = [str(params[0])]  # 'medio', 'baixo', 'alto'
        int_params = [str(int(p)) for p in params[1:]]  # x2 até x10 como inteiros
        
        cmd = [exe_path] + str_params + int_params
        
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30
        )
        
        output = (result.stdout or "") + (result.stderr or "")
        print(f"Params: {str_params + int_params} -> Output: {output.strip()}")
        
        # Tenta encontrar números na saída
        match = re.search(r"Valor\s*de\s*sa[ií]da\s*[:=]?\s*([-+]?\d*\.?\d+)", output)
        if match:
            return float(match.group(1))
            
        # Procura por qualquer número
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", output)
        if numbers:
            return float(numbers[-1])
            
        return float("inf")
        
    except Exception as e:
        print(f"Erro: {e}")
        return float("inf")

def particle_swarm_optimization(
    exe_path,
    x1_text,
    param_ranges,
    num_particles=10,
    iterations=10,
    inertia=0.5,
    cognitive=1.5,
    social=1.5,
):
    """
    PSO para números INTEIROS - versão corrigida
    """
    num_params = len(param_ranges)
    
    # Gera população inicial INTEIRA (apenas para x2 até x10)
    particles = []
    for _ in range(num_particles):
        particle = [random.randint(int(r[0]), int(r[1])) for r in param_ranges]
        particles.append(particle)
    
    particles = np.array(particles, dtype=int)
    velocities = np.zeros_like(particles, dtype=float)

    # Avaliação inicial
    personal_best = particles.copy()
    personal_best_values = []
    
    for i, p in enumerate(particles):
        # Combina x1_text (string) com os parâmetros numéricos
        full_params = [x1_text] + p.tolist()
        val = run_executable_int(exe_path, full_params)
        personal_best_values.append(val)
        print(f"Partícula {i}: {full_params} -> {val}")

    personal_best_values = np.array(personal_best_values)
    global_best_idx = np.argmin(personal_best_values)
    global_best = personal_best[global_best_idx].copy()
    global_best_value = personal_best_values[global_best_idx]
    
    print(f"Melhor inicial: {[x1_text] + global_best.tolist()} -> {global_best_value}")

    # Iterações
    for it in range(iterations):
        improved = False
        for i in range(num_particles):
            # Atualiza velocidade
            r1, r2 = random.random(), random.random()
            velocities[i] = (
                inertia * velocities[i]
                + cognitive * r1 * (personal_best[i] - particles[i])
                + social * r2 * (global_best - particles[i])
            )

            # Nova posição (arredonda para inteiro)
            new_pos_float = particles[i] + velocities[i]
            new_pos = np.round(new_pos_float).astype(int)
            
            # Limita aos valores permitidos
            for j in range(num_params):
                low, high = param_ranges[j]
                new_pos[j] = np.clip(new_pos[j], int(low), int(high))
            
            particles[i] = new_pos

            # Avalia nova posição
            full_params = [x1_text] + particles[i].tolist()
            val = run_executable_int(exe_path, full_params)
            
            # Atualiza melhores
            if val < personal_best_values[i]:
                personal_best[i] = particles[i].copy()
                personal_best_values[i] = val
                
                if val < global_best_value:
                    global_best = particles[i].copy()
                    global_best_value = val
                    improved = True
                    print(f"🔥 Iteração {it+1}: NOVO MELHOR = {val}")

        if not improved:
            print(f"Iteração {it+1}: sem melhoria (melhor = {global_best_value})")
        else:
            print(f"Iteração {it+1}: melhor valor = {global_best_value}")

    return global_best.tolist(), float(global_best_value)

# Teste rápido
if __name__ == "__main__":
    result = particle_swarm_optimization('modelo10.exe', 'medio', [(0,100)]*9, 5, 3)
    print("Resultado final:", result)
    print("Melhores parâmetros (x2-x10):", result[0])
    print("Melhor valor:", result[1])