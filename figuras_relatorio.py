# =============================================================================
# FIGURAS DO RELATÓRIO/PÔSTER — não treina nada, só desenha
#
#   figuras/relatorio/ecg_12_derivacoes.png   ← exemplo de ECG normal (referencial teórico)
#   figuras/relatorio/ecg_por_classe.png      ← derivação DII de um exame de cada classe
#   figuras/relatorio/fluxo_projeto.png       ← figura da introdução (pedida pelo professor)
#   figuras/relatorio/distribuicao_classes.png← desbalanceamento original do CODE-15%
#
# Uso: python figuras_relatorio.py
# =============================================================================

import os
from glob import glob

import numpy as np
import pandas as pd
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from comum_ecg import (PASTA_RAIZ, PASTA_DADOS, ARQUIVO_CSV, CLASSES_ORDENADAS,
                       NOMES_ORDENADOS, NUM_CLASSES, atribuir_classes)

PASTA_SAIDA = os.path.join(PASTA_RAIZ, 'figuras', 'relatorio')
os.makedirs(PASTA_SAIDA, exist_ok=True)

DERIVACOES = ['DI', 'DII', 'DIII', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
FREQUENCIA = 400   # Hz (CODE-15%)

df = pd.read_csv(ARQUIVO_CSV)
df['classe'] = atribuir_classes(df)
arquivos = {os.path.basename(p) for p in glob(os.path.join(PASTA_DADOS, '*.hdf5'))}
df_local = df[df['trace_file'].isin(arquivos) & (df['classe'] >= 0)]


def ler_exame(linha):
    with h5py.File(os.path.join(PASTA_DADOS, linha['trace_file']), 'r') as arquivo:
        posicao = int(np.where(arquivo['exam_id'][:] == linha['exam_id'])[0][0])
        return arquivo['tracings'][posicao]


def recortar(sinal, segundos):
    """Tira o preenchimento com zeros das pontas e pega `segundos` do meio."""
    ativo = np.where(np.abs(sinal).sum(axis=1) > 0)[0]
    inicio = ativo[0] if len(ativo) else 0
    return sinal[inicio:inicio + int(segundos * FREQUENCIA)]


# ----- 1) ECG normal com as 12 derivações -----
exemplo = df_local[df_local['classe'] == 0].sample(1, random_state=3).iloc[0]
sinal   = recortar(ler_exame(exemplo), 4)
tempo   = np.arange(len(sinal)) / FREQUENCIA

fig, eixos = plt.subplots(6, 2, figsize=(14, 11), sharex=True)
for i, ax in enumerate(eixos.T.flatten()):
    ax.plot(tempo, sinal[:, i], color='black', lw=0.8)
    ax.set_ylabel(DERIVACOES[i], rotation=0, labelpad=18, fontweight='bold')
    ax.set_yticks([])
    ax.grid(which='major', color='#F4A6A6', lw=0.6)
    ax.set_xticks(np.arange(0, 4.01, 0.2), minor=True)
    ax.grid(which='minor', color='#FBDADA', lw=0.3)
    for lado in ['top', 'right', 'left']:
        ax.spines[lado].set_visible(False)
for ax in eixos[-1]:
    ax.set_xlabel('Tempo (s)')
fig.suptitle('ECG de 12 derivações — exame normal do CODE-15% (400 Hz)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PASTA_SAIDA, 'ecg_12_derivacoes.png'), dpi=200)
plt.close(fig)

# ----- 2) Derivação DII de um exame de cada classe -----
fig, eixos = plt.subplots(NUM_CLASSES, 1, figsize=(13, 13), sharex=True)
for i, ax in enumerate(eixos):
    linha = df_local[df_local['classe'] == i].sample(1, random_state=1).iloc[0]
    sinal = recortar(ler_exame(linha), 5)[:, 1]
    ax.plot(np.arange(len(sinal)) / FREQUENCIA, sinal, color='black', lw=0.8)
    rotulo = NOMES_ORDENADOS[i] if i == 0 else f'{NOMES_ORDENADOS[i]} ({CLASSES_ORDENADAS[i]})'
    ax.set_title(rotulo, loc='left',
                 fontsize=11, fontweight='bold')
    ax.set_yticks([])
    ax.grid(color='#F4A6A6', lw=0.5)
    for lado in ['top', 'right', 'left']:
        ax.spines[lado].set_visible(False)
eixos[-1].set_xlabel('Tempo (s)')
fig.suptitle('Derivação DII — um exemplo de cada classe', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PASTA_SAIDA, 'ecg_por_classe.png'), dpi=200)
plt.close(fig)

# ----- 3) Distribuição original das classes (CODE-15% completo) -----
contagem = [(df['classe'] == i).sum() for i in range(NUM_CLASSES)]
fig, ax = plt.subplots(figsize=(10, 5))
barras = ax.bar(NOMES_ORDENADOS, contagem, color=['#2E7D32'] + ['#1565C0'] * 6)
ax.bar_label(barras, labels=[f'{c:,}'.replace(',', '.') for c in contagem], fontsize=9, padding=2)
ax.set_yscale('log')
ax.set_ylabel('Nº de exames (escala log)')
ax.set_title('Distribuição das classes no CODE-15% (345.779 exames)', fontweight='bold')
plt.xticks(rotation=20, ha='right')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PASTA_SAIDA, 'distribuicao_classes.png'), dpi=200)
plt.close(fig)

# ----- 4) Fluxo do projeto (figura da introdução) -----
etapas = [
    ('1. Coleta',            'CODE-15%\nTelessaúde MG\nECG 12 derivações'),
    ('2. Pré-processamento', 'Rótulos em 7 classes\nBalanceamento\nNormalização z-score\nDivisão por paciente'),
    ('3. Treinamento',       'Transformer 1D\n(principal)\n×\nCNN 1D ResNet\n(baseline)'),
    ('4. Avaliação',         'Acurácia, F1,\nsensibilidade,\nespecificidade, AUC\nMatriz de confusão'),
    ('5. Aplicação',         'Apoio ao diagnóstico\ne priorização\nno pronto-socorro'),
]
fig, ax = plt.subplots(figsize=(16, 4.2))
ax.set_xlim(0, len(etapas) * 3.2)
ax.set_ylim(0, 4)
ax.axis('off')
for k, (titulo, texto) in enumerate(etapas):
    x = k * 3.2 + 0.2
    ax.add_patch(FancyBboxPatch((x, 0.3), 2.6, 3.2, boxstyle='round,pad=0.05,rounding_size=0.2',
                                fc='#E3F2FD' if k != 2 else '#FFEBEE', ec='#0D47A1', lw=1.5))
    ax.text(x + 1.3, 3.1, titulo, ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(x + 1.3, 1.7, texto, ha='center', va='center', fontsize=13, linespacing=1.4)
    if k < len(etapas) - 1:
        ax.annotate('', xy=(x + 3.1, 1.9), xytext=(x + 2.7, 1.9),
                    arrowprops=dict(arrowstyle='->', lw=2, color='#0D47A1'))
plt.tight_layout()
plt.savefig(os.path.join(PASTA_SAIDA, 'fluxo_projeto.png'), dpi=200)
plt.close(fig)

print(f"Figuras salvas em {PASTA_SAIDA}")
