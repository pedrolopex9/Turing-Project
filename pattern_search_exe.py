import subprocess
import argparse
import csv
import numpy as np
import re

def run_executable(exe_path, params):
    """
    Executa o modelo externo e retorna o valor numérico da saída.
    """
    cmd = [exe_path] + list(map(str, params))
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout.strip()

    # Tenta encontrar um número após "Valor de saída:"
    match = re.search(r"Valor de saída:\s*([-+]?\d*\.?\d+)", output)
    if match:
        return float(match.group(1))

    # Caso não encontre, tenta achar qualquer número isolado
    match_any = re.search(r"([-+]?\d*\.?\d+)", output)
    if match_any:
        return float(match_any.group(1))

    return np.inf  # se nada for encontrado

def pattern_search(exe_path, x1_text, init_nums, step, tol, max_iter, timeout):
    """
    Faz busca local (pattern search) apenas nos parâmetros numéricos.
    """
    current = np.array(init_nums, dtype=float)
    best_val = run_executable(exe_path, [x1_text] + list(current))
    iter_count = 0

    while step > tol and iter_count < max_iter:
        improved = False
        for i in range(len(current)):
            for delta in [-step, step]:
                new_params = current.copy()
                new_params[i] += delta
                new_val = run_executable(exe_path, [x1_text] + list(new_params))
                if new_val < best_val:
                    best_val = new_val
                    current = new_params
                    improved = True
        if not improved:
            step /= 2
        iter_count += 1

    return current, best_val, iter_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", required=True, help="Caminho do executável")
    parser.add_argument("--x1", type=str, default="medio", help="nível: baixo, medio, médio, alto")
    parser.add_argument("--step", type=float, default=1.0)
    parser.add_argument("--tol", type=float, default=0.01)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--out-csv", type=str, default="resultado.csv")
    args = parser.parse_args()

    # parâmetros iniciais numéricos (x2..x10)
    init_nums = [10, 20, 30, 40, 50, 60, 70, 80, 90]

    best_nums, best_val, iters = pattern_search(
        args.exe, args.x1, init_nums, args.step, args.tol, args.max_iter, args.timeout
    )

    best_params = [args.x1] + list(best_nums)

    print("========== RESULTADOS ==========")
    print("Melhores parâmetros:", best_params)
    print("Melhor valor encontrado:", best_val)
    print("Iterações executadas:", iters)

    # salva CSV
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Parâmetros", "Valor"])
        writer.writerow([best_params, best_val])

    print("Resultados salvos em:", args.out_csv)
