import subprocess
import re
import math

def run_executable(exe_path, params, timeout=30):
    """
    Executa um arquivo .exe com parâmetros e extrai o valor numérico retornado.
    Retorna float('inf') se houver erro, timeout ou saída inválida.

    Args:
        exe_path (str): Caminho do executável (ex: 'modelo10.exe')
        params (list): Lista de parâmetros, ex: ['medio', 10, 20, ...]
        timeout (int): Tempo máximo em segundos

    Returns:
        float: Valor extraído do executável ou float('inf') se falhar
    """
    try:
        # Monta o comando completo
        cmd = [exe_path] + [str(p) for p in params]

        # Executa o processo
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )

        output = (result.stdout or "") + (result.stderr or "")

        # Tenta encontrar um número após "Valor de saída:"
        match = re.search(r"Valor\s*de\s*sa[ií]da\s*[:=]?\s*([-+]?\d*\.?\d+)", output)
        if match:
            return float(match.group(1))

        # Caso o programa tenha retornado diretamente o valor (sem texto)
        match_direct = re.search(r"([-+]?\d*\.?\d+)", output)
        if match_direct:
            return float(match_direct.group(1))

        # Nenhum valor encontrado → erro lógico
        print(f"[ERRO] Nenhum valor numérico encontrado.\nSaída bruta:\n{output}")
        return float("inf")

    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] Execução excedeu {timeout}s → {params}")
        return float("inf")

    except Exception as e:
        print(f"[ERRO] Falha ao executar {exe_path}: {e}")
        return float("inf")
