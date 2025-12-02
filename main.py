import subprocess
import sys
import time
import hashlib
import json
import random
from typing import List, Union, Tuple

class BancoMemoria:
    """Gerenciador de Cache para evitar recálculo de soluções já testadas."""
    _armazenamento = {}

    @classmethod
    def gerar_hash(cls, executavel: str, parametros: List) -> str:
        """Cria uma assinatura única para a combinação de parâmetros."""
        dados_str = json.dumps(parametros, sort_keys=True)
        return hashlib.md5(f"{executavel}:{dados_str}".encode()).hexdigest()

    @classmethod
    def consultar(cls, executavel: str, parametros: List):
        """Verifica se já existe resultado salvo."""
        chave = cls.gerar_hash(executavel, parametros)
        return cls._armazenamento.get(chave)

    @classmethod
    def salvar(cls, executavel: str, parametros: List, resultado: float):
        """Guarda o resultado no dicionário estático."""
        chave = cls.gerar_hash(executavel, parametros)
        cls._armazenamento[chave] = resultado

    @classmethod
    def total_salvo(cls) -> int:
        return len(cls._armazenamento)


class BuscaDiretaPadrao:
    """Algoritmo de Busca por Padrão (Pattern Search) com suporte a tipos mistos."""

    def __init__(self, alvo_exe: str, objetivo: str = 'maximizar'):
        self.alvo_exe = alvo_exe
        self.objetivo = objetivo.lower()
        # Define o recorde inicial base
        self.recorde_atual = float('-inf') if self.objetivo == 'maximizar' else float('inf')
        self.config_recorde = None
        self.log_execucao = []
        self.contagem_runs = 0
        self.hits_memoria = 0

    def analisar_entrada(self, texto_input: str) -> Tuple[List, List[str]]:
        """Identifica tipos de dados (int, float, string) a partir do input."""
        partes = texto_input.strip().split()
        lista_valores = []
        lista_tipos = []

        for item in partes:
            # Tenta converter para Inteiro
            try:
                val_i = int(item)
                if val_i == 0: val_i = 1 # Evita zero inicial
                lista_valores.append(val_i)
                lista_tipos.append('int')
                continue
            except ValueError:
                pass

            # Tenta converter para Float
            try:
                val_f = float(item)
                if val_f == 0.0: val_f = 1.0
                lista_valores.append(val_f)
                lista_tipos.append('float')
                continue
            except ValueError:
                pass

            # Assume String
            lista_valores.append(item.lower())
            lista_tipos.append('str')

        return lista_valores, lista_tipos

    def computar_simulacao(self, params: List) -> float:
        """Roda o processo externo e captura o retorno."""
        # Checagem de memória antes de rodar
        valor_cache = BancoMemoria.consultar(self.alvo_exe, params)
        if valor_cache is not None:
            self.hits_memoria += 1
            return valor_cache

        try:
            args_cmd = [str(p) for p in params]
            
            processo = subprocess.run(
                [self.alvo_exe] + args_cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            saida_raw = processo.stdout.strip()

            # Parser da saída
            if ":" in saida_raw:
                resultado_num = float(saida_raw.split(":")[-1].strip())
            else:
                resultado_num = float(saida_raw)

            self.contagem_runs += 1
            BancoMemoria.salvar(self.alvo_exe, params, resultado_num)

            # Lógica de Recorde
            houve_melhoria = False
            if self.objetivo == 'maximizar':
                if resultado_num > self.recorde_atual: houve_melhoria = True
            else:
                if resultado_num < self.recorde_atual: houve_melhoria = True

            if houve_melhoria:
                self.recorde_atual = resultado_num
                self.config_recorde = params.copy()
                legenda = "MÁXIMO" if self.objetivo == 'maximizar' else "MÍNIMO"
                print(f" >> Novo {legenda}: {resultado_num:.6f} | Config: {params}")

            self.log_execucao.append((params.copy(), resultado_num))
            return resultado_num

        except ValueError:
            # Captura erro de conversão (provável erro do exe)
            # print(f"[DEBUG] Retorno inválido: {processo.stdout.strip()}") 
            return float('-inf')
        except Exception as e:
            print(f"[ERRO CRÍTICO] Falha na execução: {e}")
            return float('-inf')

    def criar_adjacentes(self, config: List, tipos: List[str], tamanho_passo: Union[int, float], min_lim: float, max_lim: float) -> List[List]:
        """Gera configurações vizinhas alterando parâmetros."""
        candidatos = []

        for i in range(len(config)):
            tipo_dado = tipos[i]

            if tipo_dado == 'int':
                passo = max(1, int(tamanho_passo))
                
                # Vizinho Superior
                copia_mais = config.copy()
                val_mais = config[i] + passo
                if val_mais <= max_lim:
                    copia_mais[i] = val_mais
                    candidatos.append(copia_mais)

                # Vizinho Inferior
                copia_menos = config.copy()
                val_menos = config[i] - passo
                if val_menos >= min_lim:
                    copia_menos[i] = val_menos
                    candidatos.append(copia_menos)

            elif tipo_dado == 'float':
                passo = float(tamanho_passo)
                
                # Vizinho Superior
                copia_mais = config.copy()
                val_mais = round(config[i] + passo, 6)
                if val_mais <= max_lim:
                    copia_mais[i] = val_mais
                    candidatos.append(copia_mais)

                # Vizinho Inferior
                copia_menos = config.copy()
                val_menos = round(config[i] - passo, 6)
                if val_menos >= min_lim:
                    copia_menos[i] = val_menos
                    candidatos.append(copia_menos)

            elif tipo_dado == 'str':
                # Mapa de substituições comuns
                mapa_strings = {
                    'baixo': ['medio', 'alto'],
                    'medio': ['baixo', 'alto'],
                    'alto': ['baixo', 'medio'],
                    'true': ['false'], 'false': ['true'],
                    'sim': ['nao'], 'nao': ['sim'],
                    'yes': ['no'], 'no': ['yes']
                }
                
                val_atual = str(config[i]).lower()
                if val_atual in mapa_strings:
                    for substituto in mapa_strings[val_atual]:
                        copia_alt = config.copy()
                        copia_alt[i] = substituto
                        candidatos.append(copia_alt)

        return candidatos

    def iniciar_busca(self,
                      dados_iniciais: List,
                      meta_tipos: List[str],
                      passo_start: Union[int, float] = 20,
                      passo_min: Union[int, float] = 1,
                      redutor: float = 0.5,
                      limite_iteracoes: int = 1000,
                      piso: Union[int, float] = 1,
                      teto: Union[int, float] = 100):
        
        print("\n" + "-"*60)
        print(" INICIANDO BUSCA DIRETA (PATTERN SEARCH) ")
        print("-"*60)
        print(f" Objetivo: {self.objetivo.upper()}")
        print(f" Config Inicial: {dados_iniciais}")
        print(f" Variáveis: {len(dados_iniciais)} | Passo Inicial: {passo_start}")
        print("-"*60 + "\n")

        inicio_t = time.time()
        
        curr_params = dados_iniciais.copy()
        curr_passo = passo_start
        rodada = 0
        estagnacao = 0
        limite_estagnacao = 10

        print(">> Avaliando ponto de partida...")
        curr_valor = self.computar_simulacao(curr_params)
        print(f">> Valor Base: {curr_valor:.6f}\n")

        while rodada < limite_iteracoes and curr_passo >= passo_min:
            rodada += 1
            print(f"--- Rodada {rodada} (Passo: {curr_passo}) ---")

            lista_vizinhos = self.criar_adjacentes(curr_params, meta_tipos, curr_passo, piso, teto)

            if not lista_vizinhos:
                print(" [!] Sem vizinhos válidos.")
                if curr_passo <= passo_min: break
                
                curr_passo *= redutor
                if 'int' in meta_tipos: curr_passo = max(passo_min, int(curr_passo))
                estagnacao += 1
                continue

            melhor_local = None
            valor_melhor_local = curr_valor

            # Avalia vizinhança
            for viz in lista_vizinhos:
                res = self.computar_simulacao(viz)
                
                if self.objetivo == 'maximizar':
                    if res > valor_melhor_local:
                        valor_melhor_local = res
                        melhor_local = viz
                else:
                    if res < valor_melhor_local:
                        valor_melhor_local = res
                        melhor_local = viz

            # Decisão de movimento
            if melhor_local is not None:
                curr_params = melhor_local
                curr_valor = valor_melhor_local
                print(f" > Avançando: {curr_params} -> {curr_valor:.6f}")
                estagnacao = 0
            else:
                estagnacao += 1
                if curr_passo <= passo_min:
                    print(" [!] Passo mínimo atingido sem melhoria. Encerrando.")
                    break
                
                curr_passo *= redutor
                if 'int' in meta_tipos: curr_passo = max(passo_min, int(curr_passo))
                print(f" > Sem avanço. Refinando passo para: {curr_passo}")

                if estagnacao >= limite_estagnacao:
                    print(f" [!] Estagnação detectada ({limite_estagnacao}x). Parando.")
                    break

        duracao = time.time() - inicio_t
        print("\n" + "="*60)
        print(" RESULTADO FINAL DA OTIMIZAÇÃO ")
        print("="*60)
        print(f" Tempo: {duracao:.2f}s")
        print(f" Execuções Totais: {self.contagem_runs}")
        print(f" Uso de Cache: {self.hits_memoria}")
        print(f" Recorde ({self.objetivo}): {self.recorde_atual:.6f}")
        print(f" Configuração Vencedora: {self.config_recorde}")
        print("="*60 + "\n")

        return self.config_recorde, self.recorde_atual

    def exportar_dados(self, nome_arq: str = "log_pattern_search.txt"):
        with open(nome_arq, 'w', encoding='utf-8') as arquivo:
            arquivo.write("LOG DE PROCESSAMENTO - PATTERN SEARCH\n")
            arquivo.write("-"*50 + "\n")
            arquivo.write(f"Melhor Resultado: {self.recorde_atual:.6f}\n")
            arquivo.write(f"Parâmetros: {self.config_recorde}\n")
            arquivo.write(f"Execuções: {self.contagem_runs}\n\n")
            arquivo.write("HISTÓRICO DE TENTATIVAS\n")
            for i, (p, v) in enumerate(self.log_execucao, 1):
                arquivo.write(f"#{i} | {p} -> {v:.6f}\n")
        print(f"[INFO] Dados exportados para: {nome_arq}")


class AlgoritmoGeneticoEvolutivo:
    """Implementação de AG (Algoritmo Genético) com Elitismo e Cache."""

    def __init__(self, alvo_exe: str, objetivo: str = 'maximizar'):
        self.alvo_exe = alvo_exe
        self.objetivo = objetivo.lower()
        self.recorde_atual = float('-inf') if self.objetivo == 'maximizar' else float('inf')
        self.config_recorde = None
        self.log_execucao = []
        self.contagem_runs = 0
        self.hits_memoria = 0

    # Reutiliza lógica de detecção
    def analisar_entrada(self, padrao: str) -> Tuple[List, List[str]]:
        # Mesma lógica da classe anterior, copiada para manter independência
        valores_str = padrao.strip().split()
        valores = []
        tipos = []
        for val in valores_str:
            try:
                v_int = int(val)
                if v_int == 0: v_int = 1
                valores.append(v_int)
                tipos.append('int')
                continue
            except: pass
            try:
                v_flt = float(val)
                if v_flt == 0.0: v_flt = 1.0
                valores.append(v_flt)
                tipos.append('float')
                continue
            except: pass
            valores.append(val.lower())
            tipos.append('str')
        return valores, tipos

    def computar_simulacao(self, params: List) -> float:
        # Verifica cache
        cached = BancoMemoria.consultar(self.alvo_exe, params)
        if cached is not None:
            self.hits_memoria += 1
            return cached

        try:
            args = [str(p) for p in params]
            proc = subprocess.run([self.alvo_exe] + args, capture_output=True, text=True, timeout=30)
            out = proc.stdout.strip()
            val = float(out.split(":")[-1].strip()) if ":" in out else float(out)

            self.contagem_runs += 1
            BancoMemoria.salvar(self.alvo_exe, params, val)

            melhorou = False
            if self.objetivo == 'maximizar':
                if val > self.recorde_atual: melhorou = True
            else:
                if val < self.recorde_atual: melhorou = True

            if melhorou:
                self.recorde_atual = val
                self.config_recorde = params.copy()
                tag = "MÁXIMO" if self.objetivo == 'maximizar' else "MÍNIMO"
                print(f" >> AG Novo {tag}: {val:.6f} | Genes: {params}")

            self.log_execucao.append((params.copy(), val))
            return val
        except:
            return float('-inf') if self.objetivo == 'maximizar' else float('inf')

    def criar_cromossomo(self, tipos: List[str], min_v: float, max_v: float, ref: List = None) -> List:
        genes = []
        banco_strs = ['baixo', 'medio', 'alto']
        for i, tp in enumerate(tipos):
            if tp == 'int':
                genes.append(random.randint(int(min_v), int(max_v)))
            elif tp == 'float':
                genes.append(round(random.uniform(min_v, max_v), 6))
            elif tp == 'str':
                genes.append(random.choice(banco_strs))
        return genes

    def cruzamento(self, pai_a: List, pai_b: List, tipos: List[str]) -> Tuple[List, List]:
        corte = random.randint(1, len(pai_a) - 1)
        filho_a = pai_a[:corte] + pai_b[corte:]
        filho_b = pai_b[:corte] + pai_a[corte:]
        return filho_a, filho_b

    def aplicar_mutacao(self, cromossomo: List, tipos: List[str], taxa: float, min_v: float, max_v: float) -> List:
        mutante = cromossomo.copy()
        banco_strs = ['baixo', 'medio', 'alto']
        
        for i in range(len(mutante)):
            if random.random() < taxa:
                if tipos[i] == 'int':
                    delta = random.randint(-20, 20)
                    novo = mutante[i] + delta
                    mutante[i] = max(int(min_v), min(int(max_v), novo))
                elif tipos[i] == 'float':
                    delta = random.uniform(-10.0, 10.0)
                    novo = mutante[i] + delta
                    mutante[i] = round(max(min_v, min(max_v, novo)), 6)
                elif tipos[i] == 'str':
                    mutante[i] = random.choice(banco_strs)
        return mutante

    def selecao_torneio(self, pop: List[List], fit: List[float], k: int = 3) -> List:
        competidores = random.sample(range(len(pop)), min(k, len(pop)))
        if self.objetivo == 'maximizar':
            vencedor = max(competidores, key=lambda i: fit[i])
        else:
            vencedor = min(competidores, key=lambda i: fit[i])
        return pop[vencedor]

    def iniciar_busca(self, dados_iniciais: List, meta_tipos: List[str],
                      tam_pop: int = 100, geracoes: int = 150,
                      prob_cross: float = 0.85, prob_mut: float = 0.2,
                      piso: float = 1, teto: float = 100):
        
        print("\n" + "-"*60)
        print(" ALGORITMO GENÉTICO (EVOLUÇÃO) ")
        print("-"*60)
        
        if 'str' in meta_tipos:
            print(" [AVISO] Strings detectadas. Serão mutadas aleatoriamente.")

        t_start = time.time()
        print("Criando população ancestral...")
        
        populacao = [self.criar_cromossomo(meta_tipos, piso, teto) for _ in range(tam_pop)]
        populacao[0] = dados_iniciais.copy() # Garante que o input do usuário está na mistura

        for g in range(geracoes):
            print(f"\n--- Geração {g + 1} / {geracoes} ---")
            
            fitness = [self.computar_simulacao(ind) for ind in populacao]
            
            # Identifica o alpha da geração
            if self.objetivo == 'maximizar':
                idx_alpha = max(range(len(fitness)), key=lambda i: fitness[i])
            else:
                idx_alpha = min(range(len(fitness)), key=lambda i: fitness[i])
            
            print(f" > Melhor da Geração: {fitness[idx_alpha]:.6f}")

            nova_pop = []
            nova_pop.append(populacao[idx_alpha].copy()) # Elitismo

            while len(nova_pop) < tam_pop:
                p1 = self.selecao_torneio(populacao, fitness)
                p2 = self.selecao_torneio(populacao, fitness)

                if random.random() < prob_cross:
                    f1, f2 = self.cruzamento(p1, p2, meta_tipos)
                else:
                    f1, f2 = p1.copy(), p2.copy()

                f1 = self.aplicar_mutacao(f1, meta_tipos, prob_mut, piso, teto)
                f2 = self.aplicar_mutacao(f2, meta_tipos, prob_mut, piso, teto)

                nova_pop.append(f1)
                if len(nova_pop) < tam_pop: nova_pop.append(f2)
            
            populacao = nova_pop

        duracao = time.time() - t_start
        print("\n" + "="*60)
        print(" EVOLUÇÃO CONCLUÍDA ")
        print(f" Recorde: {self.recorde_atual:.6f}")
        print(f" Genes Vencedores: {self.config_recorde}")
        print(f" Tempo: {duracao:.2f}s")
        print("="*60 + "\n")
        return self.config_recorde, self.recorde_atual

    def exportar_dados(self, nome="log_genetico.txt"):
        with open(nome, 'w', encoding='utf-8') as f:
            f.write("LOG GENETICO\n")
            f.write(f"Resultado: {self.recorde_atual}\n")
            f.write(f"Genes: {self.config_recorde}\n")
            for i, (p, v) in enumerate(self.log_execucao, 1):
                f.write(f"{i}. {p} -> {v}\n")
        print(f"[INFO] Log salvo em {nome}")


class EnxameParticulasPSO:
    """Otimização por Enxame de Partículas (PSO)."""

    def __init__(self, alvo_exe: str, objetivo: str = 'maximizar'):
        self.alvo_exe = alvo_exe
        self.objetivo = objetivo.lower()
        self.recorde_atual = float('-inf') if self.objetivo == 'maximizar' else float('inf')
        self.config_recorde = None
        self.log_execucao = []
        self.contagem_runs = 0
        self.hits_memoria = 0

    # Reutiliza lógica de detecção
    def analisar_entrada(self, padrao: str) -> Tuple[List, List[str]]:
        valores_str = padrao.strip().split()
        valores = []
        tipos = []
        for val in valores_str:
            try:
                v_int = int(val)
                if v_int == 0: v_int = 1
                valores.append(v_int)
                tipos.append('int')
                continue
            except: pass
            try:
                v_flt = float(val)
                if v_flt == 0.0: v_flt = 1.0
                valores.append(v_flt)
                tipos.append('float')
                continue
            except: pass
            valores.append(val.lower())
            tipos.append('str')
        return valores, tipos

    def computar_simulacao(self, params: List) -> float:
        cached = BancoMemoria.consultar(self.alvo_exe, params)
        if cached is not None:
            self.hits_memoria += 1
            return cached

        try:
            args = [str(p) for p in params]
            proc = subprocess.run([self.alvo_exe] + args, capture_output=True, text=True, timeout=30)
            out = proc.stdout.strip()
            val = float(out.split(":")[-1].strip()) if ":" in out else float(out)

            self.contagem_runs += 1
            BancoMemoria.salvar(self.alvo_exe, params, val)

            melhorou = False
            if self.objetivo == 'maximizar':
                if val > self.recorde_atual: melhorou = True
            else:
                if val < self.recorde_atual: melhorou = True

            if melhorou:
                self.recorde_atual = val
                self.config_recorde = params.copy()
                print(f" >> PSO Novo Extremo: {val:.6f} | Pos: {params}")

            self.log_execucao.append((params.copy(), val))
            return val
        except:
            return float('-inf') if self.objetivo == 'maximizar' else float('inf')

    def iniciar_posicao(self, tipos: List[str], min_v: float, max_v: float, ref: List = None) -> List:
        particula = []
        strs = ['baixo', 'medio', 'alto']
        for i, tp in enumerate(tipos):
            if tp == 'int':
                particula.append(random.randint(int(min_v), int(max_v)))
            elif tp == 'float':
                particula.append(round(random.uniform(min_v, max_v), 6))
            elif tp == 'str':
                particula.append(random.choice(strs))
        return particula

    def calcular_velocidade(self, vel_atual: List, pos_atual: List, melhor_pessoal: List, melhor_global: List,
                            tipos: List[str], w: float, c1: float, c2: float) -> List:
        nova_vel = []
        for i in range(len(vel_atual)):
            tp = tipos[i]
            if tp in ['int', 'float']:
                r1, r2 = random.random(), random.random()
                inercia = w * vel_atual[i]
                cognitivo = c1 * r1 * (melhor_pessoal[i] - pos_atual[i])
                social = c2 * r2 * (melhor_global[i] - pos_atual[i])
                
                v = inercia + cognitivo + social
                limite = 10.0 if tp == 'float' else 10
                v = max(-limite, min(limite, v))
                
                if tp == 'int': v = int(v)
                else: v = round(v, 6)
                nova_vel.append(v)
            else:
                nova_vel.append(0)
        return nova_vel

    def atualizar_pos(self, pos: List, vel: List, tipos: List[str], pBest: List, gBest: List, min_v: float, max_v: float) -> List:
        nova_p = []
        strs = ['baixo', 'medio', 'alto']
        for i in range(len(pos)):
            tp = tipos[i]
            if tp == 'int':
                val = pos[i] + vel[i]
                val = max(int(min_v), min(int(max_v), int(val)))
                nova_p.append(val)
            elif tp == 'float':
                val = pos[i] + vel[i]
                val = round(max(min_v, min(max_v), val), 6)
                nova_p.append(val)
            elif tp == 'str':
                # Estratégia probabilística para strings
                if random.random() < 0.3:
                    if random.random() < 0.7: nova_p.append(gBest[i])
                    else: nova_p.append(random.choice(strs))
                else:
                    nova_p.append(pos[i])
        return nova_p

    def iniciar_busca(self, dados_iniciais: List, meta_tipos: List[str],
                      n_particulas: int = 50, n_iter: int = 150,
                      w: float = 0.7, c1: float = 2.0, c2: float = 2.0,
                      piso: float = 1, teto: float = 100):
        
        print("\n" + "-"*60)
        print(" OTIMIZAÇÃO POR ENXAME DE PARTÍCULAS (PSO) ")
        print("-"*60)
        
        t_start = time.time()
        
        # População
        enxame = [self.iniciar_posicao(meta_tipos, piso, teto, dados_iniciais) for _ in range(n_particulas)]
        velocidades = [[0]*len(dados_iniciais) for _ in range(n_particulas)]
        for v in velocidades: # Randomiza velocidade inicial levemente
             for k, tp in enumerate(meta_tipos):
                 if tp == 'int': v[k] = random.randint(-5, 5)
                 elif tp == 'float': v[k] = random.uniform(-1, 1)

        enxame[0] = dados_iniciais.copy()

        # Avaliação inicial
        fitness = [self.computar_simulacao(p) for p in enxame]
        
        pBest = [p.copy() for p in enxame]
        pBest_fit = fitness.copy()

        if self.objetivo == 'maximizar':
            idx_g = max(range(len(fitness)), key=lambda i: fitness[i])
        else:
            idx_g = min(range(len(fitness)), key=lambda i: fitness[i])
        
        gBest = enxame[idx_g].copy()
        gBest_fit = fitness[idx_g]

        for it in range(n_iter):
            print(f"\n--- Iteração {it + 1} / {n_iter} ---")
            
            for i in range(n_particulas):
                velocidades[i] = self.calcular_velocidade(velocidades[i], enxame[i], pBest[i], gBest, meta_tipos, w, c1, c2)
                enxame[i] = self.atualizar_pos(enxame[i], velocidades[i], meta_tipos, pBest[i], gBest, piso, teto)
                
                fitness[i] = self.computar_simulacao(enxame[i])

                # Atualiza Pessoal
                melhora_p = False
                if self.objetivo == 'maximizar':
                    if fitness[i] > pBest_fit[i]: melhora_p = True
                else:
                    if fitness[i] < pBest_fit[i]: melhora_p = True
                
                if melhora_p:
                    pBest[i] = enxame[i].copy()
                    pBest_fit[i] = fitness[i]
                    
                    # Atualiza Global
                    melhora_g = False
                    if self.objetivo == 'maximizar':
                        if fitness[i] > gBest_fit: melhora_g = True
                    else:
                        if fitness[i] < gBest_fit: melhora_g = True
                    
                    if melhora_g:
                        gBest = enxame[i].copy()
                        gBest_fit = fitness[i]
            
            print(f" > Líder Global Atual: {gBest_fit:.6f}")

        duracao = time.time() - t_start
        print("\n" + "="*60)
        print(" PSO FINALIZADO ")
        print(f" Recorde: {self.recorde_atual:.6f}")
        print("="*60 + "\n")
        return self.config_recorde, self.recorde_atual

    def exportar_dados(self, nome="log_pso.txt"):
        with open(nome, 'w', encoding='utf-8') as f:
            f.write("LOG PSO\n")
            f.write(f"Max: {self.recorde_atual}\n")
            for i, (p, v) in enumerate(self.log_execucao, 1):
                f.write(f"{i}. {p} -> {v}\n")
        print(f"[INFO] Salvo em {nome}")


class SistemaHibrido:
    """
    ESTRATÉGIA HÍBRIDA DE OTIMIZAÇÃO
    1. PSO (Exploração Ampla)
    2. Nelder-Mead (Focagem)
    3. Pattern Search (Refinamento Fino)
    """

    def __init__(self, alvo_exe: str, objetivo: str = 'maximizar'):
        self.alvo_exe = alvo_exe
        self.objetivo = objetivo.lower()
        self.recorde_atual = float('-inf') if self.objetivo == 'maximizar' else float('inf')
        self.config_recorde = None
        self.log_execucao = []
        self.contagem_runs = 0
        self.hits_memoria = 0
        self.top_candidatos = []

    # Reutiliza lógica
    def analisar_entrada(self, padrao: str) -> Tuple[List, List[str]]:
        valores_str = padrao.strip().split()
        valores = []
        tipos = []
        for val in valores_str:
            try:
                v_int = int(val)
                if v_int == 0: v_int = 1
                valores.append(v_int)
                tipos.append('int')
                continue
            except: pass
            try:
                v_flt = float(val)
                if v_flt == 0.0: v_flt = 1.0
                valores.append(v_flt)
                tipos.append('float')
                continue
            except: pass
            valores.append(val.lower())
            tipos.append('str')
        return valores, tipos

    def computar_simulacao(self, params: List) -> float:
        cached = BancoMemoria.consultar(self.alvo_exe, params)
        if cached is not None:
            self.hits_memoria += 1
            return cached

        try:
            args = [str(p) for p in params]
            proc = subprocess.run([self.alvo_exe] + args, capture_output=True, text=True, timeout=30)
            out = proc.stdout.strip()
            val = float(out.split(":")[-1].strip()) if ":" in out else float(out)

            self.contagem_runs += 1
            BancoMemoria.salvar(self.alvo_exe, params, val)

            melhorou = False
            if self.objetivo == 'maximizar':
                if val > self.recorde_atual: melhorou = True
            else:
                if val < self.recorde_atual: melhorou = True

            if melhorou:
                self.recorde_atual = val
                self.config_recorde = params.copy()
                print(f"    [NOVO RECORDE] {val:.6f} | {params}")

            self.log_execucao.append((params.copy(), val))
            self._atualizar_ranking(params, val)
            return val
        except:
            return float('-inf') if self.objetivo == 'maximizar' else float('inf')

    def _atualizar_ranking(self, params: List, val: float):
        self.top_candidatos.append((params.copy(), val))
        if self.objetivo == 'maximizar':
            self.top_candidatos.sort(key=lambda x: x[1], reverse=True)
        else:
            self.top_candidatos.sort(key=lambda x: x[1])
        self.top_candidatos = self.top_candidatos[:10]

    def _criar_aleatorio(self, tipos: List[str], min_v: float, max_v: float) -> List:
        ind = []
        strs = ['baixo', 'medio', 'alto']
        for tp in tipos:
            if tp == 'int': ind.append(random.randint(int(min_v), int(max_v)))
            elif tp == 'float': ind.append(round(random.uniform(min_v, max_v), 6))
            elif tp == 'str': ind.append(random.choice(strs))
        return ind

    def _executar_fase1(self, inicio: List, tipos: List[str], min_v: float, max_v: float, n_part: int = 40, n_it: int = 50):
        print("\n" + "="*60)
        print(" FASE 1: VARREDURA GLOBAL (PSO) ")
        print("="*60)
        
        strs = ['baixo', 'medio', 'alto']
        parts = []
        vels = []

        for i in range(n_part):
            if i == 0: parts.append(inicio.copy())
            else: parts.append(self._criar_aleatorio(tipos, min_v, max_v))
            
            v = []
            for tp in tipos:
                if tp == 'int': v.append(random.randint(-10, 10))
                elif tp == 'float': v.append(random.uniform(-5, 5))
                else: v.append(0)
            vels.append(v)

        fitness = [self.computar_simulacao(p) for p in parts]
        pBest = [p.copy() for p in parts]
        pBest_fit = fitness.copy()

        if self.objetivo == 'maximizar':
            best_i = max(range(len(fitness)), key=lambda i: fitness[i])
        else:
            best_i = min(range(len(fitness)), key=lambda i: fitness[i])
        
        gBest = parts[best_i].copy()
        gBest_fit = fitness[best_i]

        w, c1, c2 = 0.9, 1.5, 2.0

        for it in range(n_it):
            w = 0.9 - (0.5 * it / n_it)
            for i in range(n_part):
                for j in range(len(tipos)):
                    if tipos[j] in ['int', 'float']:
                        r1, r2 = random.random(), random.random()
                        vels[i][j] = (w * vels[i][j] + c1*r1*(pBest[i][j] - parts[i][j]) + c2*r2*(gBest[j] - parts[i][j]))
                        limite = (max_v - min_v) * 0.3
                        vels[i][j] = max(-limite, min(limite, vels[i][j]))
                
                for j in range(len(tipos)):
                    if tipos[j] == 'int':
                        parts[i][j] = int(parts[i][j] + vels[i][j])
                        parts[i][j] = max(int(min_v), min(int(max_v), parts[i][j]))
                    elif tipos[j] == 'float':
                        parts[i][j] = parts[i][j] + vels[i][j]
                        parts[i][j] = round(max(min_v, min(max_v, parts[i][j])), 6)
                    elif tipos[j] == 'str':
                        if random.random() < 0.2: parts[i][j] = random.choice(strs)
                
                fitness[i] = self.computar_simulacao(parts[i])

                melhora = False
                if self.objetivo == 'maximizar':
                    if fitness[i] > pBest_fit[i]: melhora = True
                else:
                    if fitness[i] < pBest_fit[i]: melhora = True
                
                if melhora:
                    pBest[i] = parts[i].copy()
                    pBest_fit[i] = fitness[i]
                    melhora_g = False
                    if self.objetivo == 'maximizar':
                        if fitness[i] > gBest_fit: melhora_g = True
                    else:
                        if fitness[i] < gBest_fit: melhora_g = True
                    if melhora_g:
                        gBest = parts[i].copy()
                        gBest_fit = fitness[i]
            
            if (it+1) % 10 == 0:
                print(f"  Varredura {it+1}/{n_it}: Líder = {gBest_fit:.6f}")

    def _executar_fase2(self, tipos: List[str], min_v: float, max_v: float, max_it: int = 100):
        print("\n" + "="*60)
        print(" FASE 2: CONCENTRAÇÃO (NELDER-MEAD) ")
        print("="*60)
        
        elite = self.top_candidatos[:5]
        print(f"Otimizando os {len(elite)} melhores candidatos...")
        
        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5

        for idx, (p_ini, v_ini) in enumerate(elite):
            print(f"\n  [Candidato {idx+1}] Base: {v_ini:.6f}")
            n = len(p_ini)
            step = (max_v - min_v) * 0.1
            
            simplex = [p_ini.copy()]
            for i in range(n):
                vtx = p_ini.copy()
                if tipos[i] == 'int': vtx[i] = min(int(max_v), vtx[i] + max(1, int(step)))
                elif tipos[i] == 'float': vtx[i] = min(max_v, vtx[i] + step)
                simplex.append(vtx)
            
            vals = [self.computar_simulacao(v) for v in simplex]

            for _ in range(max_it):
                if self.objetivo == 'maximizar': ids = sorted(range(len(vals)), key=lambda i: vals[i], reverse=True)
                else: ids = sorted(range(len(vals)), key=lambda i: vals[i])
                
                simplex = [simplex[i] for i in ids]
                vals = [vals[i] for i in ids]

                if max(vals) - min(vals) < 1e-6: break

                centro = []
                for j in range(n):
                    if tipos[j] in ['int', 'float']:
                        c = sum(simplex[i][j] for i in range(n)) / n
                        centro.append(c)
                    else: centro.append(simplex[0][j])
                
                pior = simplex[-1]
                refl = []
                for j in range(n):
                    if tipos[j] == 'int':
                        val = int(centro[j] + alpha*(centro[j] - pior[j]))
                        val = max(int(min_v), min(int(max_v), val))
                        refl.append(val)
                    elif tipos[j] == 'float':
                        val = centro[j] + alpha*(centro[j] - pior[j])
                        val = max(min_v, min(max_v, val))
                        refl.append(round(val, 6))
                    else: refl.append(pior[j])
                
                fr = self.computar_simulacao(refl)

                aceita_refl = False
                if self.objetivo == 'maximizar':
                    if fr > vals[-1]:
                        simplex[-1] = refl
                        vals[-1] = fr
                else:
                    if fr < vals[-1]:
                        simplex[-1] = refl
                        vals[-1] = fr

    def _executar_fase3(self, tipos: List[str], min_v: float, max_v: float):
        print("\n" + "="*60)
        print(" FASE 3: POLIMENTO FINAL (AJUSTE FINO) ")
        print("="*60)

        if self.config_recorde is None: return
        
        curr_p = self.config_recorde.copy()
        curr_v = self.recorde_atual
        passos = [10, 5, 2, 1]

        for p in passos:
            print(f"\n  Ajuste Fino com Passo {p}")
            melhorou = True
            while melhorou:
                melhorou = False
                for i in range(len(curr_p)):
                    if tipos[i] == 'str':
                        for op in ['baixo', 'medio', 'alto']:
                            if op != curr_p[i]:
                                teste = curr_p.copy()
                                teste[i] = op
                                v_teste = self.computar_simulacao(teste)
                                ok = False
                                if self.objetivo == 'maximizar' and v_teste > curr_v: ok = True
                                if self.objetivo == 'minimizar' and v_teste < curr_v: ok = True
                                if ok:
                                    curr_p, curr_v = teste, v_teste
                                    melhorou = True
                    elif tipos[i] in ['int', 'float']:
                        for d in [p, -p]:
                            teste = curr_p.copy()
                            if tipos[i] == 'int':
                                teste[i] = max(int(min_v), min(int(max_v), teste[i] + d))
                            else:
                                teste[i] = max(min_v, min(max_v, teste[i] + d))
                            
                            v_teste = self.computar_simulacao(teste)
                            ok = False
                            if self.objetivo == 'maximizar' and v_teste > curr_v: ok = True
                            if self.objetivo == 'minimizar' and v_teste < curr_v: ok = True
                            if ok:
                                curr_p, curr_v = teste, v_teste
                                melhorou = True
                                break

    def iniciar_busca(self, dados_iniciais: List, meta_tipos: List[str], piso: float = 1, teto: float = 100):
        print("\n" + "="*60)
        print("       INICIANDO SISTEMA HÍBRIDO ")
        print("="*60)
        print(f" Modo: {self.objetivo.upper()}")
        print(f" Config Base: {dados_iniciais}")
        
        t_total = time.time()
        
        self._executar_fase1(dados_iniciais, meta_tipos, piso, teto, 40, 60)
        self._executar_fase2(meta_tipos, piso, teto, 80)
        self._executar_fase3(meta_tipos, piso, teto)

        duracao = time.time() - t_total
        print("\n" + "="*60)
        print(" PROCESSO HÍBRIDO ENCERRADO ")
        print("="*60)
        print(f" Tempo Total: {duracao:.2f}s")
        print(f" Execuções: {self.contagem_runs}")
        print(f" Recorde Final: {self.recorde_atual:.6f}")
        print(f" Melhor Configuração: {self.config_recorde}")
        print("="*60 + "\n")
        return self.config_recorde, self.recorde_atual

    def exportar_dados(self, nome="log_hibrido.txt"):
        with open(nome, 'w', encoding='utf-8') as f:
            f.write("LOG HIBRIDO\n")
            f.write(f"Final: {self.recorde_atual:.6f}\n")
            f.write(f"Params: {self.config_recorde}\n")
            f.write("TOP 10:\n")
            for i, (p, v) in enumerate(self.top_candidatos, 1):
                f.write(f"{i}. {p} -> {v}\n")
        print(f"[INFO] Arquivo salvo: {nome}")


class SimplexNelder:
    """Método Nelder-Mead Simplex direto."""

    def __init__(self, alvo_exe: str, objetivo: str = 'maximizar'):
        self.alvo_exe = alvo_exe
        self.objetivo = objetivo.lower()
        self.recorde_atual = float('-inf') if self.objetivo == 'maximizar' else float('inf')
        self.config_recorde = None
        self.log_execucao = []
        self.contagem_runs = 0
        self.hits_memoria = 0

    def analisar_entrada(self, padrao: str) -> Tuple[List, List[str]]:
        valores_str = padrao.strip().split()
        valores = []
        tipos = []
        for val in valores_str:
            try:
                v_int = int(val)
                if v_int == 0: v_int = 1
                valores.append(v_int)
                tipos.append('int')
                continue
            except: pass
            try:
                v_flt = float(val)
                if v_flt == 0.0: v_flt = 1.0
                valores.append(v_flt)
                tipos.append('float')
                continue
            except: pass
            valores.append(val.lower())
            tipos.append('str')
        return valores, tipos

    def computar_simulacao(self, params: List, usar_cache: bool = True) -> float:
        if usar_cache:
            cached = BancoMemoria.consultar(self.alvo_exe, params)
            if cached is not None:
                self.hits_memoria += 1
                return cached

        try:
            args = [str(p) for p in params]
            proc = subprocess.run([self.alvo_exe] + args, capture_output=True, text=True, timeout=30)
            out = proc.stdout.strip()
            val = float(out.split(":")[-1].strip()) if ":" in out else float(out)

            self.contagem_runs += 1
            if usar_cache: BancoMemoria.salvar(self.alvo_exe, params, val)

            melhorou = False
            if self.objetivo == 'maximizar':
                if val > self.recorde_atual: melhorou = True
            else:
                if val < self.recorde_atual: melhorou = True

            if melhorou:
                self.recorde_atual = val
                self.config_recorde = params.copy()
                print(f" >> NM Recorde: {val:.6f} | {params}")

            self.log_execucao.append((params.copy(), val))
            return val
        except:
            return float('-inf') if self.objetivo == 'maximizar' else float('inf')

    def montar_simplex(self, inicio: List, tipos: List[str], min_v: float, max_v: float, tamanho: float = 5.0) -> List[List]:
        n = len(inicio)
        simplex = [inicio.copy()]
        for i in range(n):
            vtx = inicio.copy()
            if tipos[i] == 'int':
                delta = max(1, int(tamanho))
                novo = vtx[i] + delta
                vtx[i] = max(int(min_v), min(int(max_v), novo))
            elif tipos[i] == 'float':
                novo = vtx[i] + tamanho
                vtx[i] = round(max(min_v, min(max_v, novo)), 6)
            simplex.append(vtx)
        return simplex

    def centroide(self, pts: List[List], skip: int = -1) -> List:
        if skip >= 0: pts = [p for i, p in enumerate(pts) if i != skip]
        dim = len(pts[0])
        centro = []
        for j in range(dim):
            vals = [p[j] for p in pts if isinstance(p[j], (int, float))]
            if vals: centro.append(sum(vals)/len(vals))
            else: centro.append(pts[0][j])
        return centro

    def refletir(self, ruim: List, centro: List, a: float, tipos: List[str], min_v: float, max_v: float) -> List:
        res = []
        for i in range(len(ruim)):
            if tipos[i] in ['int', 'float']:
                val = centro[i] + a * (centro[i] - ruim[i])
                if tipos[i] == 'int': val = int(max(int(min_v), min(int(max_v), val)))
                else: val = round(max(min_v, min(max_v), val), 6)
                res.append(val)
            else: res.append(ruim[i])
        return res

    def expandir(self, centro: List, refl: List, g: float, tipos: List[str], min_v: float, max_v: float) -> List:
        exp = []
        for i in range(len(refl)):
            if tipos[i] in ['int', 'float']:
                val = centro[i] + g * (refl[i] - centro[i])
                if tipos[i] == 'int': val = int(max(int(min_v), min(int(max_v), val)))
                else: val = round(max(min_v, min(max_v), val), 6)
                exp.append(val)
            else: exp.append(refl[i])
        return exp

    def contrair(self, centro: List, pt: List, rho: float, tipos: List[str], min_v: float, max_v: float) -> List:
        con = []
        for i in range(len(pt)):
            if tipos[i] in ['int', 'float']:
                val = centro[i] + rho * (pt[i] - centro[i])
                if tipos[i] == 'int': val = int(max(int(min_v), min(int(max_v), val)))
                else: val = round(max(min_v, min(max_v), val), 6)
                con.append(val)
            else: con.append(pt[i])
        return con

    def encolher(self, simpl: List[List], idx_bom: int, sig: float, tipos: List[str], min_v: float, max_v: float) -> List[List]:
        bom = simpl[idx_bom]
        novo_s = [bom]
        for i, pt in enumerate(simpl):
            if i == idx_bom: continue
            novo_pt = []
            for j in range(len(pt)):
                if tipos[j] in ['int', 'float']:
                    val = bom[j] + sig * (pt[j] - bom[j])
                    if tipos[j] == 'int': val = int(max(int(min_v), min(int(max_v), val)))
                    else: val = round(max(min_v, min(max_v), val), 6)
                    novo_pt.append(val)
                else: novo_pt.append(pt[j])
            novo_s.append(novo_pt)
        return novo_s

    def iniciar_busca(self, dados_iniciais: List, meta_tipos: List[str],
                      max_it: int = 500, passo_ini: float = 30.0,
                      alpha: float = 1.0, gamma: float = 2.0, rho: float = 0.5, sigma: float = 0.5,
                      tol: float = 1e-6, piso: float = 1, teto: float = 100):
        
        print("\n" + "="*60)
        print(" NELDER-MEAD SIMPLEX (MULTI-START) ")
        print("="*60)
        
        pontos_partida = [dados_iniciais.copy()]
        if 'str' in meta_tipos:
            print(" [Multi-Start] Variações de string ativadas.")
            for op in ['baixo', 'medio', 'alto']:
                var = dados_iniciais.copy()
                for k, t in enumerate(meta_tipos):
                    if t == 'str': var[k] = op
                if var != dados_iniciais: pontos_partida.append(var)

        total_partidas = len(pontos_partida)
        t_start = time.time()

        for num_run, p_start in enumerate(pontos_partida, 1):
            print(f"\n --- Execução {num_run}/{total_partidas} --- Início: {p_start}")
            
            simplex = self.montar_simplex(p_start, meta_tipos, piso, teto, passo_ini)
            valores = [self.computar_simulacao(v) for v in simplex]

            it = 0
            sem_melhora = 0
            v_anterior = self.recorde_atual

            while it < max_it:
                it += 1
                if self.objetivo == 'maximizar': order = sorted(range(len(valores)), key=lambda i: valores[i], reverse=True)
                else: order = sorted(range(len(valores)), key=lambda i: valores[i])
                
                simplex = [simplex[i] for i in order]
                valores = [valores[i] for i in order]

                idx_bom, idx_ruim, idx_ruim2 = 0, len(valores)-1, len(valores)-2

                if max(valores) - min(valores) < tol:
                    print(" [Convergência Atingida]")
                    break

                if it % 20 == 0:
                    print(f" Iter {it}: Extremo = {self.recorde_atual:.6f}")
                    if abs(self.recorde_atual - v_anterior) < tol*10:
                        sem_melhora += 1
                        if sem_melhora >= 5:
                            print(" [Parada] Estagnação prolongada.")
                            break
                    else: sem_melhora = 0
                    v_anterior = self.recorde_atual

                c = self.centroide(simplex, idx_ruim)
                xr = self.refletir(simplex[idx_ruim], c, alpha, meta_tipos, piso, teto)
                fr = self.computar_simulacao(xr)

                if self.objetivo == 'maximizar':
                    if fr > valores[idx_ruim2] and fr <= valores[idx_bom]:
                        simplex[idx_ruim], valores[idx_ruim] = xr, fr
                        continue
                    if fr > valores[idx_bom]:
                        xe = self.expandir(c, xr, gamma, meta_tipos, piso, teto)
                        fe = self.computar_simulacao(xe)
                        if fe > fr: simplex[idx_ruim], valores[idx_ruim] = xe, fe
                        else: simplex[idx_ruim], valores[idx_ruim] = xr, fr
                        continue
                    xc = self.contrair(c, simplex[idx_ruim], rho, meta_tipos, piso, teto)
                    fc = self.computar_simulacao(xc)
                    if fc > valores[idx_ruim]:
                        simplex[idx_ruim], valores[idx_ruim] = xc, fc
                        continue
                else:
                    if fr < valores[idx_ruim2] and fr >= valores[idx_bom]:
                        simplex[idx_ruim], valores[idx_ruim] = xr, fr
                        continue
                    if fr < valores[idx_bom]:
                        xe = self.expandir(c, xr, gamma, meta_tipos, piso, teto)
                        fe = self.computar_simulacao(xe)
                        if fe < fr: simplex[idx_ruim], valores[idx_ruim] = xe, fe
                        else: simplex[idx_ruim], valores[idx_ruim] = xr, fr
                        continue
                    xc = self.contrair(c, simplex[idx_ruim], rho, meta_tipos, piso, teto)
                    fc = self.computar_simulacao(xc)
                    if fc < valores[idx_ruim]:
                        simplex[idx_ruim], valores[idx_ruim] = xc, fc
                        continue

                simplex = self.encolher(simplex, idx_bom, sigma, meta_tipos, piso, teto)
                valores = [self.computar_simulacao(v) for v in simplex]

        print("\n" + "="*60)
        print(f" Finalizado em {time.time()-t_start:.2f}s ")
        print(f" Recorde Global: {self.recorde_atual:.6f}")
        print("="*60 + "\n")
        return self.config_recorde, self.recorde_atual

    def exportar_dados(self, nome="log_simplex.txt"):
        with open(nome, 'w', encoding='utf-8') as f:
            f.write("LOG SIMPLEX\n")
            f.write(f"Recorde: {self.recorde_atual}\n")
            f.write(f"Params: {self.config_recorde}\n")
            for i, (p, v) in enumerate(self.log_execucao, 1):
                f.write(f"{i}. {p} -> {v}\n")
        print(f"[INFO] Exportado: {nome}")


def interface_principal():
    print("\n" + "="*60)
    print("="*60)
    print("\nEstratégias Disponíveis:\n")
    print("  [1] Busca Direta Pattern")
    print("  [2] Algoritmo Genético")
    print("  [3] Enxame de Partículas (PSO)")
    print("  [4] Método Simplex (Nelder-Mead)")
    print("  [5] Sistema Híbrido (Inteligente)")
    print("\n  [0] Encerrar")
    print(f"\n  Resultados em Cache: {BancoMemoria.total_salvo()}")
    print("="*60)

def setup_busca_direta():
    print("\n--- CONFIGURAR BUSCA DIRETA ---")
    print("Insira o executável e os valores iniciais (ex: app.exe 10 20 30):")
    entrada = input(" >> ").strip()
    if not entrada:
        entrada = "modelo10.exe baixo 1 1 1 1 1 1 1 1 1"
        print(f" [Default] {entrada}")
    
    partes = entrada.split()
    if len(partes) < 2: return False
    
    exe, padrao = partes[0], " ".join(partes[1:])
    otimizador = BuscaDiretaPadrao(exe)
    ini_params, tipos = otimizador.analisar_entrada(padrao)
    
    print(f" [Info] Variáveis: {len(ini_params)} | Tipos: {tipos}")
    
    print("\nObjetivo:")
    print(" [1] Maximizar (Padrão)")
    print(" [2] Minimizar")
    if input(" >> ") == '2': otimizador.objetivo = 'minimizar'
    
    try:
        print("\nParâmetros (Enter para padrão):")
        passo = input(" Passo inicial [20]: ").strip()
        passo = float(passo) if passo else 20
        
        lim_min = input(" Limite Min [1]: ").strip()
        lim_min = float(lim_min) if lim_min else 1
        
        lim_max = input(" Limite Max [100]: ").strip()
        lim_max = float(lim_max) if lim_max else 100
    except:
        passo, lim_min, lim_max = 20, 1, 100
    
    try:
        otimizador.iniciar_busca(ini_params, tipos, passo_start=passo, piso=lim_min, teto=lim_max)
        otimizador.exportar_dados()
        print(f"\nComando Vencedor:\n{exe} " + " ".join(str(x) for x in otimizador.config_recorde))
        return True
    except KeyboardInterrupt:
        print("\n [Interrompido pelo usuário]")
        return False
    except Exception as e:
        print(f" [Erro] {e}")
        return False

def setup_genetico():
    print("\n--- CONFIGURAR GENÉTICO ---")
    entrada = input(" Executável + Padrão: ").strip()
    if not entrada: entrada = "modelo10.exe baixo 1 1 1 1 1 1 1 1 1"
    
    partes = entrada.split()
    if len(partes) < 2: return False
    exe, padrao = partes[0], " ".join(partes[1:])
    
    ag = AlgoritmoGeneticoEvolutivo(exe)
    ini, tipos = ag.analisar_entrada(padrao)
    
    if input(" [1]Max / [2]Min: ") == '2': ag.objetivo = 'minimizar'
    
    try:
        pop = int(input(" População [50]: ") or 50)
        gen = int(input(" Gerações [100]: ") or 100)
        min_v = float(input(" Min [1]: ") or 1)
        max_v = float(input(" Max [100]: ") or 100)
    except: pop, gen, min_v, max_v = 50, 100, 1, 100
    
    try:
        ag.iniciar_busca(ini, tipos, tam_pop=pop, geracoes=gen, piso=min_v, teto=max_v)
        ag.exportar_dados()
        print(f"\nComando Vencedor:\n{exe} " + " ".join(str(x) for x in ag.config_recorde))
        return True
    except KeyboardInterrupt:
        print("\n [Interrompido]")
        return False

def setup_pso():
    print("\n--- CONFIGURAR PSO ---")
    entrada = input(" Executável + Padrão: ").strip()
    if not entrada: entrada = "modelo10.exe baixo 1 1 1 1 1 1 1 1 1"
    
    partes = entrada.split()
    if len(partes) < 2: return False
    exe, padrao = partes[0], " ".join(partes[1:])
    
    pso = EnxameParticulasPSO(exe)
    ini, tipos = pso.analisar_entrada(padrao)
    
    if input(" [1]Max / [2]Min: ") == '2': pso.objetivo = 'minimizar'
    
    try:
        part = int(input(" Partículas [30]: ") or 30)
        it = int(input(" Iterações [100]: ") or 100)
        min_v = float(input(" Min [1]: ") or 1)
        max_v = float(input(" Max [100]: ") or 100)
    except: part, it, min_v, max_v = 30, 100, 1, 100
    
    try:
        pso.iniciar_busca(ini, tipos, n_particulas=part, n_iter=it, piso=min_v, teto=max_v)
        pso.exportar_dados()
        print(f"\nComando Vencedor:\n{exe} " + " ".join(str(x) for x in pso.config_recorde))
        return True
    except KeyboardInterrupt: return False

def setup_simplex():
    print("\n--- CONFIGURAR SIMPLEX ---")
    entrada = input(" Executável + Padrão: ").strip()
    if not entrada: entrada = "modelo10.exe baixo 1 1 1 1 1 1 1 1 1"
    
    partes = entrada.split()
    if len(partes) < 2: return False
    exe, padrao = partes[0], " ".join(partes[1:])
    
    nm = SimplexNelder(exe)
    ini, tipos = nm.analisar_entrada(padrao)
    
    if input(" [1]Max / [2]Min: ") == '2': nm.objetivo = 'minimizar'
    
    try:
        it = int(input(" Iterações [500]: ") or 500)
        step = float(input(" Step Inicial [10]: ") or 10)
        min_v = float(input(" Min [1]: ") or 1)
        max_v = float(input(" Max [100]: ") or 100)
    except: it, step, min_v, max_v = 500, 10, 1, 100
    
    try:
        nm.iniciar_busca(ini, tipos, max_it=it, passo_ini=step, piso=min_v, teto=max_v)
        nm.exportar_dados()
        print(f"\nComando Vencedor:\n{exe} " + " ".join(str(x) for x in nm.config_recorde))
        return True
    except KeyboardInterrupt: return False

def setup_hibrido():
    print("\n--- CONFIGURAR SISTEMA HÍBRIDO ---")
    print(" (PSO + Nelder-Mead + Pattern Search)")
    entrada = input(" Executável + Padrão: ").strip()
    if not entrada: entrada = "modelo10.exe baixo 1 1 1 1 1 1 1 1 1"
    
    partes = entrada.split()
    if len(partes) < 2: return False
    exe, padrao = partes[0], " ".join(partes[1:])
    
    sis = SistemaHibrido(exe)
    ini, tipos = sis.analisar_entrada(padrao)
    
    if input(" [1]Max / [2]Min: ") == '2': sis.objetivo = 'minimizar'
    
    try:
        min_v = float(input(" Min [1]: ") or 1)
        max_v = float(input(" Max [100]: ") or 100)
    except: min_v, max_v = 1, 100
    
    try:
        sis.iniciar_busca(ini, tipos, piso=min_v, teto=max_v)
        sis.exportar_dados()
        print(f"\nComando Vencedor:\n{exe} " + " ".join(str(x) for x in sis.config_recorde))
        return True
    except KeyboardInterrupt: return False

def loop_principal():
    while True:
        interface_principal()
        op = input("\n Opção: ").strip()
        
        if op == '0':
            print("\n [Saindo...] Até logo!")
            sys.exit(0)
        elif op == '1':
            setup_busca_direta()
            input("\n [Enter] para voltar...")
        elif op == '2':
            setup_genetico()
            input("\n [Enter] para voltar...")
        elif op == '3':
            setup_pso()
            input("\n [Enter] para voltar...")
        elif op == '4':
            setup_simplex()
            input("\n [Enter] para voltar...")
        elif op == '5':
            setup_hibrido()
            input("\n [Enter] para voltar...")
        else:
            print(" Opção inválida.")
            time.sleep(1)

if __name__ == "__main__":

    loop_principal()
