# =============================================================================
# ROBUSTEZ — quanto o modelo piora com os defeitos de um ECG de verdade
#
# No pronto-socorro o traçado chega com ruído, linha de base oscilando,
# interferência da rede elétrica, eletrodo solto ou de um aparelho com
# frequência de amostragem menor. Este script aplica cada defeito nos 827
# exames do CODE-test (laudo de cardiologistas) e mede AUC, F1 e
# sensibilidade com os mesmos limiares escolhidos na validação.
#
# Os defeitos entram no sinal em mV, antes do z-score, como num aparelho real.
# Semente fixa: todo modelo vê exatamente o mesmo ruído.
#
# Uso:
#   python robustez_ecg.py                       # melhor modelo v3
#   python robustez_ecg.py --modelo resultados/v3/cnn1d_s42/modelo_final.pt
# Saída: <pasta do modelo>/robustez.json, robustez.md e robustez.png
# =============================================================================

import os
import json
import argparse
from fractions import Fraction

import h5py
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import resample_poly
from sklearn.metrics import roc_auc_score

from comum_ecg import (PASTA_CODE_TEST, ORDEM_CODE_TEST, DOENCAS, FREQUENCIA_HZ,
                       NUM_AMOSTRAS, TAMANHO_BATCH, normalizar_sinais, metricas_por_limiar)
from inferencia_ecg import carregar_modelo, remover_linha_de_base

SEMENTE = 42
TEMPO = np.arange(NUM_AMOSTRAS) / FREQUENCIA_HZ


def potencia(sinal):
    """Potência média de cada derivação no trecho com sinal (o CODE completa com zeros)."""
    ativo = np.abs(sinal).sum(axis=2, keepdims=True) > 0
    return (sinal ** 2 * ativo).sum(axis=1, keepdims=True) / np.maximum(ativo.sum(axis=1, keepdims=True), 1)


def ruido(snr_db):
    def aplicar(s, rng):
        sigma = np.sqrt(potencia(s) / 10 ** (snr_db / 10))
        return s + rng.standard_normal(s.shape).astype(np.float32) * sigma
    return aplicar


def linha_de_base(amplitude_mv):
    """Oscilação lenta (respiração, movimento): 0,15–0,5 Hz, fase sorteada por derivação."""
    def aplicar(s, rng):
        f = rng.uniform(0.15, 0.5, (len(s), 1, 12))
        fase = rng.uniform(0, 2 * np.pi, (len(s), 1, 12))
        return s + amplitude_mv * np.sin(2 * np.pi * f * TEMPO[None, :, None] + fase).astype(np.float32)
    return aplicar


def rede_eletrica(amplitude_mv, hz=60):
    def aplicar(s, rng):
        fase = rng.uniform(0, 2 * np.pi, (len(s), 1, 1))
        return s + amplitude_mv * np.sin(2 * np.pi * hz * TEMPO[None, :, None] + fase).astype(np.float32)
    return aplicar


def derivacoes_soltas(quantas, so_precordiais=False):
    """Zera derivações sorteadas por exame (eletrodo desconectado)."""
    def aplicar(s, rng):
        s = s.copy()
        opcoes = np.arange(6, 12) if so_precordiais else np.arange(12)
        for i in range(len(s)):
            s[i][:, rng.choice(opcoes, quantas, replace=False)] = 0
        return s
    return aplicar


def amostragem(hz):
    """Ida e volta por um aparelho de menor frequência: perde o que passa de hz/2."""
    def aplicar(s, rng):
        r = Fraction(hz / FREQUENCIA_HZ).limit_denominator(1000)
        baixo = resample_poly(s, r.numerator, r.denominator, axis=1)
        volta = resample_poly(baixo, r.denominator, r.numerator, axis=1)[:, :NUM_AMOSTRAS]
        return volta.astype(np.float32)
    return aplicar


def registro_curto(segundos):
    """Mantém só os primeiros segundos de sinal e completa com zero, como o CODE faz."""
    def aplicar(s, rng):
        s = s.copy()
        for i in range(len(s)):
            ativo = np.flatnonzero(np.abs(s[i]).sum(axis=1) > 0)
            if len(ativo):
                s[i][ativo[0] + int(segundos * FREQUENCIA_HZ):] = 0
        return s
    return aplicar


# (grupo, condição, função). A ordem é a da tabela e do gráfico.
CONDICOES = [
    ('Original',               'sem alteração',          None),
    ('Ruído branco',           'SNR 20 dB',              ruido(20)),
    ('Ruído branco',           'SNR 10 dB',              ruido(10)),
    ('Ruído branco',           'SNR 5 dB',               ruido(5)),
    ('Linha de base',          '0,5 mV',                 linha_de_base(0.5)),
    ('Linha de base',          '1 mV',                   linha_de_base(1.0)),
    ('Rede elétrica (60 Hz)',  '0,1 mV',                 rede_eletrica(0.1)),
    ('Rede elétrica (60 Hz)',  '0,3 mV',                 rede_eletrica(0.3)),
    ('Eletrodo solto',         '1 derivação',            derivacoes_soltas(1)),
    ('Eletrodo solto',         '3 derivações',           derivacoes_soltas(3)),
    ('Eletrodo solto',         '6 precordiais',          derivacoes_soltas(6, so_precordiais=True)),
    ('Amostragem menor',       '250 Hz',                 amostragem(250)),
    ('Amostragem menor',       '150 Hz',                 amostragem(150)),
    ('Registro curto',         '5 s',                    registro_curto(5)),
    ('Registro curto',         '3 s',                    registro_curto(3)),
]


@torch.no_grad()
def probabilidades(modelo, sinais, dispositivo):
    saidas = []
    for i in range(0, len(sinais), TAMANHO_BATCH):
        x = torch.from_numpy(normalizar_sinais(sinais[i:i + TAMANHO_BATCH])).to(dispositivo)
        saidas.append(torch.sigmoid(modelo(x).float()).cpu().numpy())
    return np.concatenate(saidas)


def medir(probs, rotulos, limiares):
    d = len(DOENCAS)
    lim = np.array(limiares['equilibrado'][:d])
    auc = [roc_auc_score(rotulos[:, j], probs[:, j]) for j in range(d)]
    por_limiar = [metricas_por_limiar(rotulos[:, j], probs[:, j], lim[j]) for j in range(d)]
    alterado = rotulos.max(axis=1)
    lim_triagem = np.array(limiares['triagem'][:d])
    triagem = metricas_por_limiar(alterado, (probs[:, :d] >= lim_triagem).max(axis=1).astype(float), 0.5)
    return {
        'auc_macro': float(np.mean(auc)),
        'f1_macro': float(np.mean([m['f1'] for m in por_limiar])),
        'sensibilidade_macro': float(np.mean([m['sensibilidade'] for m in por_limiar])),
        'f1_por_classe': {s: float(m['f1']) for s, m in zip(DOENCAS, por_limiar)},
        'sensibilidade_alterado_triagem': float(triagem['sensibilidade']),
        'especificidade_alterado_triagem': float(triagem['especificidade']),
    }


def tabela(resultados, info):
    base = resultados[0]
    linhas = [f"# Robustez no CODE-test ({info['arquitetura']})", '',
              f"Modelo: `{os.path.relpath(info['caminho'])}`. 827 exames, limiares da validação "
              f"(F1 e sensibilidade no equilibrado; a última coluna é a triagem \"tem alguma das 6?\").", '',
              '| Defeito | Nível | AUC macro | F1 macro | Δ F1 | Sens. macro | Sens. triagem (alterado) |',
              '|---|---|---|---|---|---|---|']
    for r in resultados:
        m = r['metricas']
        delta = '' if r is base else f"{m['f1_macro'] - base['metricas']['f1_macro']:+.3f}"
        linhas.append(f"| {r['grupo']} | {r['condicao']} | {m['auc_macro']:.3f} | {m['f1_macro']:.3f} | {delta} "
                      f"| {m['sensibilidade_macro']:.3f} | {m['sensibilidade_alterado_triagem']:.3f} |")
    linhas += ['', '## F1 por doença', '', '| Defeito | Nível | ' + ' | '.join(DOENCAS) + ' |',
               '|---|---|' + '---|' * len(DOENCAS)]
    for r in resultados:
        linhas.append(f"| {r['grupo']} | {r['condicao']} | "
                      + ' | '.join(f"{r['metricas']['f1_por_classe'][d]:.2f}" for d in DOENCAS) + ' |')
    return '\n'.join(linhas) + '\n'


def grafico(resultados, arquivo):
    """F1 macro por condição; a linha tracejada é o exame original."""
    rotulos = [f"{r['grupo']}: {r['condicao']}" if r['grupo'] != 'Original' else 'Original'
               for r in resultados][::-1]
    f1 = [r['metricas']['f1_macro'] for r in resultados][::-1]
    base = resultados[0]['metricas']['f1_macro']

    fig, ax = plt.subplots(figsize=(8, 0.38 * len(rotulos) + 1.2))
    ax.barh(rotulos, f1, height=0.6, color='#3b6ea5')
    ax.axvline(base, color='#555555', linestyle='--', linewidth=1)
    for y, v in enumerate(f1):
        ax.text(v + 0.01, y, f"{v:.2f}", va='center', fontsize=8, color='#333333')
    ax.set_xlim(0, 1.05)
    ax.set_xlabel('F1 macro no CODE-test (limiar equilibrado)')
    ax.set_title('Desempenho com defeitos comuns do traçado', loc='left', fontsize=11)
    for lado in ['top', 'right']:
        ax.spines[lado].set_visible(False)
    ax.grid(axis='x', color='#e5e5e5', linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(arquivo, dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelo', default=None, help='caminho do modelo_final.pt (padrão: melhor v3)')
    parser.add_argument('--filtro', action='store_true', help='passa-alta de 0,5 Hz antes do modelo, como na inferência')
    parser.add_argument('--dispositivo', default='cpu', help="'cpu' não disputa a GPU com um treino")
    args = parser.parse_args()

    modelo, info = carregar_modelo(args.modelo, args.dispositivo)
    with h5py.File(os.path.join(PASTA_CODE_TEST, 'ecg_tracings.hdf5'), 'r') as f:
        brutos = f['tracings'][:][..., ORDEM_CODE_TEST].astype(np.float32)   # mV, ordem do treino
    rotulos = pd.read_csv(os.path.join(PASTA_CODE_TEST, 'annotations', 'gold_standard.csv'))[DOENCAS].to_numpy()

    resultados = []
    for grupo, condicao, funcao in CONDICOES:
        rng = np.random.default_rng(SEMENTE)
        sinais = brutos if funcao is None else funcao(brutos, rng)
        if args.filtro:
            sinais = remover_linha_de_base(sinais)
        m = medir(probabilidades(modelo, sinais, args.dispositivo), rotulos, info['limiares'])
        resultados.append({'grupo': grupo, 'condicao': condicao, 'metricas': m})
        print(f"{grupo:24s} {condicao:16s} AUC {m['auc_macro']:.3f}  F1 {m['f1_macro']:.3f}  "
              f"sens. triagem {m['sensibilidade_alterado_triagem']:.3f}")

    pasta = os.path.dirname(info['caminho'])
    sufixo = '_filtro' if args.filtro else ''
    with open(os.path.join(pasta, f'robustez{sufixo}.json'), 'w', encoding='utf-8') as f:
        json.dump({'modelo': info['caminho'], 'semente': SEMENTE, 'filtro': args.filtro, 'resultados': resultados},
                  f, ensure_ascii=False, indent=2)
    with open(os.path.join(pasta, f'robustez{sufixo}.md'), 'w', encoding='utf-8') as f:
        f.write(tabela(resultados, info))
    grafico(resultados, os.path.join(pasta, f'robustez{sufixo}.png'))
    print(f"Resultados em {pasta}")
