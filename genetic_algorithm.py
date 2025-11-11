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
        
        # Verifica se há restrições violadas
        if "x2 e x3 devem estar entre 1 e 100" in output:
            return float("inf")
        
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

def genetic_algorithm(
    exe_path, x1_text, param_ranges, pop_size=20, generations=20, mutation_rate=0.2,
    elite_size=4, tournament_size=3
):
    """
    Otimização via Algoritmo Genético para números INTEIROS.
    param_ranges: lista de tuplas [(min, max), ...] para x2..x10
    """
    num_params = len(param_ranges)

    def create_individual():
        """Cria um indivíduo com parâmetros inteiros"""
        individual = []
        for j, (low, high) in enumerate(param_ranges):
            # Para x2 e x3 (índices 0 e 1), garante que estejam entre 1-100
            if j in [0, 1]:  # x2 e x3
                individual.append(random.randint(1, 100))
            else:
                individual.append(random.randint(int(low), int(high)))
        return individual

    def fitness(individual):
        """Função de fitness - queremos MINIMIZAR o valor"""
        val = run_executable_int(exe_path, [x1_text] + individual)
        # Quanto menor o valor, maior o fitness
        if val == float('inf'):
            return -1000000  # Penalidade muito alta para soluções inválidas
        return -val  # Negativo porque queremos minimizar

    def tournament_selection(population, fitnesses, tournament_size):
        """Seleção por torneio"""
        contestants = random.sample(list(zip(population, fitnesses)), tournament_size)
        contestants.sort(key=lambda x: x[1], reverse=True)
        return contestants[0][0]

    def crossover(parent1, parent2):
        """Cruzamento de um ponto"""
        if random.random() < 0.8:  # 80% de chance de crossover
            point = random.randint(1, num_params - 1)
            child1 = parent1[:point] + parent2[point:]
            child2 = parent2[:point] + parent1[point:]
            return child1, child2
        return parent1[:], parent2[:]

    def mutate(individual):
        """Mutação para inteiros"""
        mutated = individual[:]
        for i in range(num_params):
            if random.random() < mutation_rate:
                low, high = param_ranges[i]
                if i in [0, 1]:  # x2 e x3
                    mutated[i] = random.randint(1, 100)
                else:
                    mutated[i] = random.randint(int(low), int(high))
        return mutated

    # Cria população inicial
    population = [create_individual() for _ in range(pop_size)]
    
    best_individual = None
    best_fitness = float('-inf')
    best_value = float('inf')

    print("=== INICIANDO ALGORITMO GENÉTICO ===")
    
    for gen in range(generations):
        # Avaliar fitness da população
        fitnesses = [fitness(ind) for ind in population]
        
        # Encontrar o melhor da geração
        gen_best_idx = np.argmax(fitnesses)
        gen_best_fitness = fitnesses[gen_best_idx]
        gen_best_individual = population[gen_best_idx]
        gen_best_value = -gen_best_fitness  # Converte de volta para o valor real
        
        # Atualizar melhor global
        if gen_best_fitness > best_fitness:
            best_fitness = gen_best_fitness
            best_individual = gen_best_individual[:]
            best_value = gen_best_value
            print(f"🔥 Geração {gen+1}: NOVO MELHOR = {best_value:.4f}")
        else:
            print(f"Geração {gen+1}: melhor valor = {gen_best_value:.4f}")

        # Criar nova população
        new_population = []
        
        # Elitismo: mantém os melhores indivíduos
        elite_indices = np.argsort(fitnesses)[-elite_size:]
        for idx in elite_indices:
            new_population.append(population[idx][:])
        
        # Preenche o resto da população com crossover e mutação
        while len(new_population) < pop_size:
            # Seleção por torneio
            parent1 = tournament_selection(population, fitnesses, tournament_size)
            parent2 = tournament_selection(population, fitnesses, tournament_size)
            
            # Crossover
            child1, child2 = crossover(parent1, parent2)
            
            # Mutação
            child1 = mutate(child1)
            child2 = mutate(child2)
            
            new_population.extend([child1, child2])
        
        # Garante que a população não exceda o tamanho
        population = new_population[:pop_size]

    print("=== RESULTADO FINAL ===")
    print(f"Melhor valor encontrado: {best_value:.4f}")
    print(f"Melhores parâmetros (x2-x10): {best_individual}")
    
    return best_individual, best_value

# Teste rápido
if __name__ == "__main__":
    # Define os ranges para x2-x10
    # x2 e x3: entre 1 e 100, x4-x10: entre 0 e 100
    param_ranges = [(1, 100), (1, 100)] + [(0, 100)] * 7
    
    result = genetic_algorithm(
        exe_path='modelo10.exe',
        x1_text='medio',
        param_ranges=param_ranges,
        pop_size=20,
        generations=20,
        mutation_rate=0.15,
        elite_size=4,
        tournament_size=3
    )
    
    print("\n=== PARÂMETROS FINAIS ===")
    print(f"x1: medio")
    for i, param in enumerate(result[0], 2):
        print(f"x{i}: {param}")
    print(f"Valor final: {result[1]:.4f}")