# =============================================================================
# TREINAR TODOS — fila com os 3 modelos × 3 sementes do protocolo v3
#
# Roda um treino por vez (a GPU é uma só), com log em resultados/v3/log_*.txt.
# Pula o que já tem metricas.json, e um treino interrompido retoma do
# checkpoint, então dá para parar e rodar de novo sem perder nada.
# No fim, gera as tabelas (gerar_tabelas.py).
#
# Uso:  python treinar_todos.py
#       python treinar_todos.py --sementes 42        (só uma semente)
# =============================================================================

import os
import sys
import argparse
import subprocess

PASTA_RAIZ = os.path.dirname(os.path.abspath(__file__))
PASTA_V3   = os.path.join(PASTA_RAIZ, 'resultados', 'v3')

MODELOS = [  # (nome da pasta, script, argumentos extras) — o modelo do produto primeiro
    ('cnn1d',               'cnn1d_ecg.py',       []),
    ('transformer_hibrido', 'transformer_ecg.py', ['--hibrido']),
    ('transformer',         'transformer_ecg.py', []),
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sementes', type=int, nargs='*', default=[42, 7, 123])
    args = parser.parse_args()
    os.makedirs(PASTA_V3, exist_ok=True)

    for semente in args.sementes:                  # semente por fora: cada modelo ganha resultado cedo
        for nome, script, extras in MODELOS:
            if os.path.exists(os.path.join(PASTA_V3, f'{nome}_s{semente}', 'metricas.json')):
                print(f"{nome} s{semente}: já treinado — pulando")
                continue
            log = os.path.join(PASTA_V3, f'log_{nome}_s{semente}.txt')
            print(f"{nome} s{semente}: treinando (log em {log})", flush=True)
            with open(log, 'a', encoding='utf-8') as saida:
                retorno = subprocess.run([sys.executable, '-u', script, '--semente', str(semente), *extras],
                                         cwd=PASTA_RAIZ, stdout=saida, stderr=subprocess.STDOUT)
            if retorno.returncode != 0:
                raise SystemExit(f"{nome} s{semente} falhou (código {retorno.returncode}); veja {log}")

    subprocess.run([sys.executable, 'gerar_tabelas.py'], cwd=PASTA_RAIZ)
