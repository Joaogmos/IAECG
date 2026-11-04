# =============================================================================
# TRIAGEM EM LOTE — analisa uma pasta de exames (ou um HDF5 com vários) e grava
# uma fila em CSV, com os exames alterados primeiro
#
# É o uso pensado para um serviço com fila de laudos: o cardiologista abre
# primeiro os exames que o modelo sinalizou, e os de sinal ruim aparecem
# marcados para repetir.
#
# Ordem da fila:
#   1. sinal inadequado (alerta de gravidade alta): repetir o exame
#   2. alterado (alguma das 6 doenças acima do limiar), pela maior probabilidade
#   3. outra alteração (nem as 6 nem "normal")
#   4. normal
#
# Uso:
#   python triagem_lote.py pasta_com_exames/ --frequencia 500 --saida fila.csv
#   python triagem_lote.py dados/code_test/data/ecg_tracings.hdf5 --modo triagem
# WFDB: o .hea e o .dat de mesmo nome são lidos juntos.
# =============================================================================

import os
import csv
import argparse
from glob import glob

import h5py
import torch

from comum_ecg import DOENCAS
from inferencia_ecg import carregar_modelo, ler_arquivos, analisar

EXTENSOES = ('.hdf5', '.h5', '.npy', '.csv', '.txt', '.hea')
ORDEM_VEREDITO = {'repetir': 0, 'alterado': 1, 'outra_alteracao': 2, 'normal': 3}


def listar_exames(caminhos):
    """Devolve [(identificação, {arquivo: bytes}, índice no HDF5)] de arquivos e pastas."""
    arquivos = []
    for c in caminhos:
        if os.path.isdir(c):
            arquivos += sorted(f for f in glob(os.path.join(c, '**', '*'), recursive=True)
                               if f.lower().endswith(EXTENSOES))
        else:
            arquivos.append(c)

    exames = []
    for arquivo in arquivos:
        extensao = os.path.splitext(arquivo)[1].lower()
        if extensao == '.hea':
            base = os.path.splitext(arquivo)[0]
            par = {f: open(f, 'rb').read() for f in (arquivo, base + '.dat') if os.path.exists(f)}
            exames.append((os.path.basename(base), par, 0))
        elif extensao in ('.hdf5', '.h5'):
            with h5py.File(arquivo, 'r') as f:
                n = len(f['tracings'])
            conteudo = open(arquivo, 'rb').read()
            for i in range(n):
                exames.append((f"{os.path.basename(arquivo)}#{i}", {arquivo: conteudo}, i))
        else:
            exames.append((os.path.basename(arquivo), {arquivo: open(arquivo, 'rb').read()}, 0))
    return exames


def triar(exames, modelo, info, frequencia, modo):
    linhas = []
    for n, (identificacao, conteudos, indice) in enumerate(exames, 1):
        try:
            sinal, fs = ler_arquivos(conteudos, frequencia, indice)
            r = analisar(sinal, fs, modelo, info, modo, com_saliencia=False)
        except Exception as erro:          # arquivo ilegível não derruba a fila
            linhas.append({'exame': identificacao, 'veredito': 'erro', 'observacao': str(erro)})
            continue

        probs = {x['sigla']: x['probabilidade'] for x in r['resultados']}
        veredito = r['veredito'] if r['qualidade']['adequado'] else 'repetir'
        linhas.append({
            'exame':       identificacao,
            'veredito':    veredito,
            'achados':     ' '.join(r['achados']),
            'maior_prob':  round(max(probs[d] for d in DOENCAS), 4),
            **{f'p_{s}': round(p, 4) for s, p in probs.items()},
            'fc_bpm':      r['ritmo']['frequencia_bpm'] if r['ritmo'] else '',
            'observacao':  '; '.join(a['mensagem'] for a in r['qualidade']['alertas']),
        })
        if n % 50 == 0:
            print(f"  {n}/{len(exames)} exames")

    linhas.sort(key=lambda l: (ORDEM_VEREDITO.get(l['veredito'], 4), -l.get('maior_prob', 0)))
    for posicao, l in enumerate(linhas, 1):
        l['posicao'] = posicao
    return linhas


def gravar(linhas, saida):
    colunas = ['posicao', 'exame', 'veredito', 'achados', 'maior_prob'] \
        + list(dict.fromkeys(k for l in linhas for k in l if k.startswith('p_'))) + ['fc_bpm', 'observacao']
    with open(saida, 'w', newline='', encoding='utf-8-sig') as f:   # BOM: o Excel abre com acento
        escritor = csv.DictWriter(f, fieldnames=colunas, delimiter=';', extrasaction='ignore')
        escritor.writeheader()
        escritor.writerows(linhas)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('caminhos', nargs='+', help='arquivos ou pastas de exames')
    parser.add_argument('--frequencia', type=float, default=400, help='Hz dos .csv/.npy')
    parser.add_argument('--modo', choices=['equilibrado', 'triagem'], default='equilibrado')
    parser.add_argument('--modelo', default=None, help='caminho do modelo_final.pt')
    parser.add_argument('--saida', default='fila_triagem.csv')
    args = parser.parse_args()

    dispositivo = 'cuda' if torch.cuda.is_available() else 'cpu'
    modelo, info = carregar_modelo(args.modelo, dispositivo)
    exames = listar_exames(args.caminhos)
    print(f"{len(exames)} exames | modelo {info['arquitetura']} | modo {args.modo} | {dispositivo}")

    linhas = triar(exames, modelo, info, args.frequencia, args.modo)
    gravar(linhas, args.saida)

    contagem = {}
    for l in linhas:
        contagem[l['veredito']] = contagem.get(l['veredito'], 0) + 1
    print(f"Fila gravada em {args.saida}: " + ', '.join(f"{v} {k}" for k, v in contagem.items()))
