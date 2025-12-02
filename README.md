🚀 Ferramenta de Otimização Universal (Black Box)

Este é um sistema de otimização automatizado desenvolvido em Python. Ele foi projetado para calibrar e encontrar os melhores parâmetros para qualquer executável externo (caixa-preta), independentemente da lógica interna ou da quantidade de parâmetros.

O sistema utiliza algoritmos meta-heurísticos avançados para maximizar (ou minimizar) a pontuação retornada pelo software alvo.

✨ Principais Funcionalidades

Universal: Funciona com qualquer executável (.exe).

Auto-Adaptável: Detecta automaticamente a quantidade de parâmetros e seus tipos (Inteiros, Decimais ou Texto).

Sem Dependências: Roda com Python puro (Standard Library), sem necessidade de pip install.

Cache Inteligente: Memoriza resultados passados para não perder tempo recalculando a mesma configuração.

Multi-Estratégia: Inclui 5 algoritmos, desde buscas locais simples até sistemas híbridos complexos.

🛠️ Como Usar

1. Preparação

Certifique-se de que o arquivo do seu executável (ex: simulado.exe, modelo.exe) esteja na mesma pasta que este script Python.

2. Execução

Abra o terminal (ou VS Code) na pasta do arquivo e execute:

python otimizador_completo.py


3. O Menu Principal

Escolha a estratégia de otimização digitando o número correspondente:

[1] Pattern Search: Rápido e preciso para ajustes finos. Bom para problemas simples.

[3] PSO (Enxame): Excelente para explorar o mapa todo e não ficar preso em falsos topos.

[5] Híbrido (Recomendado): Combina PSO + Nelder-Mead + Pattern Search. É o mais robusto e garante o melhor resultado, embora demore mais.


4. Definindo o Padrão (O Passo Mais Importante)

O programa pedirá: Cmd + Params.
Você deve digitar o nome do executável seguido de valores de exemplo para os parâmetros.

O script usará esses valores para entender quantos parâmetros existem e qual o tipo de cada um.

Cenário A: Apenas Números Inteiros

Se o programa exige 5 números inteiros:

simulado.exe 10 10 10 10 10


(O sistema entende: "Otimizar 5 variáveis do tipo Inteiro").

Cenário B: Texto + Números

Se o programa exige uma configuração (baixo/alto) e 3 números:

modelo_fabrica.exe baixo 100 50 20


(O sistema entende: "A 1ª variável é Texto (vai testar variações como medio/alto), as outras 3 são Inteiros").

⚙️ Configurações Adicionais

Após definir o padrão, o sistema fará perguntas rápidas de configuração. Se tiver dúvida, apenas pressione ENTER para usar o padrão recomendado.

Objetivo:

1 para Maximizar (Buscar maior nota/lucro).

2 para Minimizar (Buscar menor erro/custo).

Limites (Min/Max):

Define as fronteiras da busca.

Ex: Se os parâmetros só podem ir de 0 a 100, mantenha o padrão. Se podem ir até 1000, digite 1000 no Max.

🧠 Explicação das Estratégias

Estratégia

Quando usar?

Descrição

Pattern Search

Testes rápidos

Tenta somar e subtrair valores vizinhos. Se melhora, avança. Se não, diminui o passo. É como tatear no escuro.

Algoritmo Genético

Problemas complexos

Simula a evolução natural. Cria uma "população" de soluções que se cruzam e sofrem mutações ao longo das gerações.

PSO (Enxame)

Exploração Global

Simula um bando de pássaros. Ótimo para encontrar a região "geral" onde está a melhor solução, evitando armadilhas locais.

Nelder-Mead

Refinamento

Usa geometria (triângulos) para escalar montanhas matemáticas rapidamente. Ótimo para "subir" valores de forma agressiva.

Híbrido

PROVA / FINAL

Executa PSO (para achar a região) -> Nelder-Mead (para subir o pico) -> Pattern Search (para ajuste fino no topo)


📋 Exemplo de Saída (Log)

Ao final, o programa exibe o melhor resultado e salva um arquivo .txt:

============================================================
 RESULTADO FINAL DA OTIMIZAÇÃO 
============================================================
 Tempo: 1180.76s
 Execuções Totais: 3240
 Recorde Final: 1101.550000
 Melhor Configuração: ['alto', 100, 100, 100, 1, 1, 3, 100, 100, 100]
============================================================
[INFO] Arquivo salvo: log_hibrido.txt
