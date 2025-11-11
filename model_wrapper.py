# model_wrapper.py
import subprocess
import re
from typing import Tuple, Optional

def parse_output_for_value(output: str) -> Optional[float]:
    """
    Tenta extrair um valor numérico da saída do executável.
    Procura por padrões comuns como:
      - "Valor: 102.16"
      - "Valor de saída: 102.16"
      - "Value: 102.16"
    Se não encontrar rótulos, retorna o último número float encontrado.
    """
    # Normaliza linha
    out = output.replace(",", ".")  # em caso de vírgulas em decimais
    # Procura rótulos
    patterns = [
        r"valor(?: de saída)?\s*[:=]\s*([-+]?\d*\.\d+|\d+)",
        r"value\s*[:=]\s*([-+]?\d*\.\d+|\d+)",
        r"resultado\s*[:=]\s*([-+]?\d*\.\d+|\d+)",
    ]
    for pat in patterns:
        m = re.search(pat, out, flags=re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except:
                pass

    # Se não encontrou por label, pega o último float na saída
    all_nums = re.findall(r"([-+]?\d*\.\d+|\d+)", out)
    if all_nums:
        try:
            return float(all_nums[-1])
        except:
            return None
    return None


def run_exe_with_args(exe_path: str, params: list[str], timeout: int = 30) -> Tuple[Optional[float], str]:
    """
    Tenta rodar o executável passando os parâmetros como argumentos de linha de comando.
    Retorna (valor_extraido, stdout_text)
    """
    try:
        cmd = [exe_path] + [str(p) for p in params]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, shell=False)
        stdout = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")
        val = parse_output_for_value(stdout)
        return val, stdout
    except Exception as e:
        return None, f"ERROR (args mode): {e}"


def run_exe_with_stdin(exe_path: str, params: list[str], timeout: int = 30) -> Tuple[Optional[float], str]:
    """
    Tenta rodar o executável e enviar os parâmetros via stdin (cada parâmetro em nova linha).
    Retorna (valor_extraido, stdout_text)
    """
    try:
        cmd = [exe_path]
        input_text = "\n".join(str(p) for p in params) + "\n"
        proc = subprocess.run(cmd, input=input_text, capture_output=True, text=True, timeout=timeout, shell=False)
        stdout = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")
        val = parse_output_for_value(stdout)
        return val, stdout
    except Exception as e:
        return None, f"ERROR (stdin mode): {e}"


def evaluate_exe(exe_path: str, params: list[str], timeout: int = 30, prefer_args: bool = True) -> Tuple[Optional[float], str]:
    """
    Tenta executar o exe e retornar o valor. Estratégia:
    1) se prefer_args: tenta args primeiro, depois stdin
    2) se not prefer_args: tenta stdin primeiro
    Retorna (valor, raw_output). Se não conseguir, valor é None e raw_output contém mensagem de erro.
    """
    if prefer_args:
        val, out = run_exe_with_args(exe_path, params, timeout=timeout)
        if val is not None:
            return val, out
        # fallback para stdin
        val2, out2 = run_exe_with_stdin(exe_path, params, timeout=timeout)
        if val2 is not None:
            return val2, out2
        # nenhum dos dois funcionou
        return None, f"No numeric value extracted.\nArgs attempt output:\n{out}\n\nStdin attempt output:\n{out2}"
    else:
        val, out = run_exe_with_stdin(exe_path, params, timeout=timeout)
        if val is not None:
            return val, out
        val2, out2 = run_exe_with_args(exe_path, params, timeout=timeout)
        if val2 is not None:
            return val2, out2
        return None, f"No numeric value extracted.\nStdin attempt output:\n{out}\n\nArgs attempt output:\n{out2}"


if __name__ == "__main__":
    # exemplo rápido de uso
    exe = "modelo10.exe"
    params = ["10", "20", "30"]
    val, out = evaluate_exe(exe, params, timeout=20)
    print("VAL:", val)
    print("OUTPUT (trecho):")
    print(out[:1000])
