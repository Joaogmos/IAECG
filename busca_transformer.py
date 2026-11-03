# =============================================================================
# BUSCA DE HIPERPARÂMETROS DO TRANSFORMER (prevista no artigo do TCC 1, Seção 3.4)
#
# Grade: cabeças (4, 8) × camadas (4, 6) × dimensão (128, 256) × patch (32, 64)
#        × LR (1e-4, 3e-4, 1e-3) = 48 combinações.
# Cada combinação treina num subconjunto fixo (--limite, mesmo split para todas)
# e é comparada pela AP macro de VALIDAÇÃO — o teste não entra na escolha.
# A melhor depois deve ser treinada na base inteira com 3 sementes:
#   python transformer_ecg.py --patch P --dimensao D --camadas L --cabecas H --lr LR
#
# Pula combinações já treinadas; dá para parar e continuar.
# Ranking em docs/tabelas/v3/busca_transformer.md
#
# Uso:  python busca_transformer.py                    (48 combinações, 60 mil exames)
#       python busca_transformer.py --amostra 12       (12 combinações sorteadas)
# =============================================================================

import os
import sys
import json
import random
import argparse
import itertools
import subprocess

PASTA_RAIZ  = os.path.dirname(os.path.abspath(__file__))
PASTA_RAPIDO = os.path.join(PASTA_RAIZ, 'resultados', 'teste_rapido')
ARQUIVO_RANKING = os.path.join(PASTA_RAIZ, 'docs', 'tabelas', 'v3', 'busca_transformer.md')

GRADE = {'cabecas': [4, 8], 'camadas': [4, 6], 'dimensao': [128, 256], 'patch': [32, 64],
         'lr': [1e-4, 3e-4, 1e-3]}


def nome_da_pasta(c):
    padrao = (c['patch'], c['dimensao'], c['camadas'], c['cabecas'], c['lr']) == (64, 128, 4, 4, 3e-4)
    if padrao:
        return 'transformer_s42'
    return f"transformer_p{c['patch']}_d{c['dimensao']}_l{c['camadas']}_h{c['cabecas']}_lr{c['lr']:g}_s42"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--limite', type=int, default=60000)
    parser.add_argument('--amostra', type=int, default=None, help='sorteia N combinações (semente 0)')
    parser.add_argument('--partes', type=int, nargs='*', default=None)
    args = parser.parse_args()

    combinacoes = [dict(zip(GRADE, valores)) for valores in itertools.product(*GRADE.values())]
    if args.amostra:
        combinacoes = random.Random(0).sample(combinacoes, args.amostra)

    for i, c in enumerate(combinacoes, start=1):
        pasta = os.path.join(PASTA_RAPIDO, nome_da_pasta(c))
        if os.path.exists(os.path.join(pasta, 'metricas.json')):
            continue
        print(f"[{i}/{len(combinacoes)}] {c}", flush=True)
        comando = [sys.executable, '-u', 'transformer_ecg.py', '--limite', str(args.limite),
                   '--patch', str(c['patch']), '--dimensao', str(c['dimensao']),
                   '--camadas', str(c['camadas']), '--cabecas', str(c['cabecas']), '--lr', str(c['lr'])]
        if args.partes is not None:
            comando += ['--partes', *map(str, args.partes)]
        os.makedirs(pasta, exist_ok=True)
        with open(os.path.join(pasta, 'log.txt'), 'a', encoding='utf-8') as log:
            subprocess.run(comando, cwd=PASTA_RAIZ, stdout=log, stderr=subprocess.STDOUT)

    linhas = []
    for c in combinacoes:
        arquivo = os.path.join(PASTA_RAPIDO, nome_da_pasta(c), 'metricas.json')
        if os.path.exists(arquivo):
            with open(arquivo, encoding='utf-8') as f:
                m = json.load(f)
            linhas.append((m['melhor_ap_val'], c, m))
    linhas.sort(key=lambda t: -t[0])

    os.makedirs(os.path.dirname(ARQUIVO_RANKING), exist_ok=True)
    texto = ['# Busca de hiperparâmetros do Transformer', '',
             f'{len(linhas)} combinações, {args.limite:,} exames, semente 42. Ordenado pela AP macro de validação.'.replace(',', '.'),
             '', '| # | Patch | Dimensão | Camadas | Cabeças | LR | Parâmetros | AP val | AUC teste | F1 teste |',
             '|---|---|---|---|---|---|---|---|---|---|']
    for k, (ap, c, m) in enumerate(linhas, start=1):
        texto.append(f"| {k} | {c['patch']} | {c['dimensao']} | {c['camadas']} | {c['cabecas']} | {c['lr']:g} | "
                     f"{m['parametros']:,} | {ap:.4f} | {m['auc_macro']:.4f} | {m['f1_macro']:.4f} |".replace(',', '.'))
    with open(ARQUIVO_RANKING, 'w', encoding='utf-8') as f:
        f.write('\n'.join(texto) + '\n')
    print('\n'.join(texto))
