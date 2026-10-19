# =============================================================================
# GERAR TABELAS — junta os resultados de todos os treinos para pôster/relatório
#
# Lê resultados/*/metricas.json (uma pasta por modelo e semente) e gera:
#   docs/tabelas/tabelas.md              ← tabelas prontas em Markdown
#   docs/tabelas/tabela_comparacao.csv   ← para colar no Word/Canva/Excel
#   docs/tabelas/tabela_f1_por_classe.csv
#   docs/tabelas/tabela_sens_espec.csv
#   docs/tabelas/tabela_dataset.csv
#   figuras/<modelo>/*.png               ← matriz de confusão e curvas (semente 42)
#   figuras/comparacao_f1_por_classe.png
#
# Uso: python gerar_tabelas.py
# =============================================================================

import os
import json
import shutil
from glob import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from comum_ecg import PASTA_RAIZ, PASTA_RESULTADOS, CLASSES_ORDENADAS, NOMES_ORDENADOS

PASTA_FIGURAS = os.path.join(PASTA_RAIZ, 'figuras')
PASTA_TABELAS = os.path.join(PASTA_RAIZ, 'docs', 'tabelas')
os.makedirs(PASTA_TABELAS, exist_ok=True)
ORDEM_MODELOS = ['transformer', 'transformer_hibrido', 'cnn1d']
CORES         = {'transformer': '#90CAF9', 'transformer_hibrido': '#1565C0', 'cnn1d': '#C62828'}
ROTULOS_CURTOS = {'transformer': 'Transf.', 'transformer_hibrido': 'Híbrido', 'cnn1d': 'CNN'}


def media_dp(valores, porcentagem=True):
    """'91.2 ± 0.4' quando há várias sementes, '91.2' quando há só uma."""
    v = np.array(valores) * (100 if porcentagem else 1)
    casas = 1 if porcentagem else 3
    if len(v) == 1:
        return f"{v[0]:.{casas}f}"
    return f"{v.mean():.{casas}f} ± {v.std(ddof=1):.{casas}f}"


def para_markdown(df):
    linhas = ['| ' + ' | '.join(df.columns) + ' |',
              '|' + '|'.join(['---'] * len(df.columns)) + '|']
    linhas += ['| ' + ' | '.join(str(x) for x in linha) + ' |' for linha in df.values]
    return '\n'.join(linhas)


execucoes = []
for arquivo in sorted(glob(os.path.join(PASTA_RESULTADOS, '*', 'metricas.json'))):
    with open(arquivo, encoding='utf-8') as f:
        execucoes.append(json.load(f))

if not execucoes:
    raise SystemExit("Nenhum resultados/*/metricas.json encontrado — rode os treinos primeiro.")

assinaturas = {e['assinatura_dados'] for e in execucoes}
if len(assinaturas) > 1:
    print(f"AVISO: execuções com dados diferentes ({assinaturas}) — a comparação não é justa.")

por_modelo = {}
for e in execucoes:
    por_modelo.setdefault(e['nome'], []).append(e)
modelos = [m for m in ORDEM_MODELOS if m in por_modelo] + \
          [m for m in por_modelo if m not in ORDEM_MODELOS]

# ----- Tabela 1: dataset -----
ref = execucoes[0]
tabela_dataset = pd.DataFrame({
    'Classe': NOMES_ORDENADOS,
    'Sigla':  CLASSES_ORDENADAS,
    'Treino':    [ref['distribuicao']['treino'][c]    for c in CLASSES_ORDENADAS],
    'Validação': [ref['distribuicao']['validacao'][c] for c in CLASSES_ORDENADAS],
    'Teste':     [ref['distribuicao']['teste'][c]     for c in CLASSES_ORDENADAS],
})
tabela_dataset['Total'] = tabela_dataset[['Treino', 'Validação', 'Teste']].sum(axis=1)
tabela_dataset.loc[len(tabela_dataset)] = ['Total', '', *tabela_dataset[['Treino', 'Validação', 'Teste', 'Total']].sum()]

# ----- Tabela 2: comparação geral -----
linhas = []
for m in modelos:
    ex = por_modelo[m]
    linhas.append({
        'Modelo':            ex[0]['modelo'],
        'Execuções (sementes)': len(ex),
        'Acurácia (%)':      media_dp([e['acuracia'] for e in ex]),
        'F1 macro (%)':      media_dp([e['f1_macro'] for e in ex]),
        'AUC macro':         media_dp([e['auc_macro'] for e in ex], porcentagem=False),
        'Acc treino (%)':    media_dp([e['acc_treino_final'] for e in ex]),
        'Parâmetros':        f"{ex[0]['parametros']:,}".replace(',', '.'),
        'Épocas':            f"{np.mean([e['epocas_treinadas'] for e in ex]):.0f}",
    })
tabela_comparacao = pd.DataFrame(linhas)

# ----- Tabela 3: F1 por classe / Tabela 4: sensibilidade e especificidade -----
def tabela_por_classe(metricas_e_nomes):
    linhas = []
    for c in CLASSES_ORDENADAS:
        linha = {'Classe': por_modelo[modelos[0]][0]['por_classe'][c]['nome']}
        for m in modelos:
            for chave, nome in metricas_e_nomes:
                coluna = f'{nome} {ROTULOS_CURTOS.get(m, m)} (%)'.strip()
                linha[coluna] = media_dp([e['por_classe'][c][chave] for e in por_modelo[m]])
        linhas.append(linha)
    return pd.DataFrame(linhas)

tabela_f1      = tabela_por_classe([('f1', 'F1')])
tabela_clinica = tabela_por_classe([('sensibilidade', 'Sens.'), ('especificidade', 'Espec.')])

# ----- Salvar -----
for nome_arquivo, tabela in [('tabela_dataset.csv', tabela_dataset),
                             ('tabela_comparacao.csv', tabela_comparacao),
                             ('tabela_f1_por_classe.csv', tabela_f1),
                             ('tabela_sens_espec.csv', tabela_clinica)]:
    tabela.to_csv(os.path.join(PASTA_TABELAS, nome_arquivo), index=False, encoding='utf-8-sig')

texto = [
    '# Tabelas de resultados — IAECG',
    '',
    'Dados: CODE-15%, ' + f"{ref['n_treino'] + ref['n_validacao'] + ref['n_teste']:,} exames de ".replace(',', '.') +
    f"{ref['pacientes']:,} pacientes".replace(',', '.') + f", divisão 70/15/15 por paciente (assinatura `{ref['assinatura_dados']}`). "
    'Métricas no conjunto de TESTE. Com mais de uma semente: média ± desvio padrão.',
    '',
    '## Tabela 1 — Distribuição dos exames por classe', '', para_markdown(tabela_dataset), '',
    '## Tabela 2 — Comparação geral dos modelos', '', para_markdown(tabela_comparacao), '',
    '## Tabela 3 — F1-score por classe', '', para_markdown(tabela_f1), '',
    '## Tabela 4 — Sensibilidade e especificidade por classe', '', para_markdown(tabela_clinica), '',
]
with open(os.path.join(PASTA_TABELAS, 'tabelas.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(texto))
print('\n'.join(texto))

# ----- Figuras para o pôster -----
for m in modelos:
    principal = next((e for e in por_modelo[m] if e['semente'] == 42), por_modelo[m][0])
    origem  = os.path.join(PASTA_RESULTADOS, f"{m}_s{principal['semente']}")
    destino = os.path.join(PASTA_FIGURAS, m)
    os.makedirs(destino, exist_ok=True)
    for png in ['matriz_confusao.png', 'curvas_aprendizado.png']:
        shutil.copy(os.path.join(origem, png), os.path.join(destino, png))

fig, ax = plt.subplots(figsize=(12, 5.8))
largura = 0.8 / len(modelos)
x = np.arange(len(CLASSES_ORDENADAS))
for k, m in enumerate(modelos):
    f1 = np.array([[e['por_classe'][c]['f1'] for c in CLASSES_ORDENADAS] for e in por_modelo[m]])
    barras = ax.bar(x + (k - (len(modelos) - 1) / 2) * largura, f1.mean(axis=0) * 100, largura,
                    yerr=f1.std(axis=0, ddof=1) * 100 if len(f1) > 1 else None, capsize=3,
                    label=por_modelo[m][0]['modelo'], color=CORES.get(m))
    ax.bar_label(barras, fmt='%.0f', fontsize=8, padding=2)
ax.set_xticks(x, NOMES_ORDENADOS, rotation=20, ha='right')
ax.set_ylabel('F1-score no teste (%)')
ax.set_ylim(0, 108)
ax.set_title('F1-score por classe no conjunto de teste', fontweight='bold')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=len(modelos), frameon=False)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PASTA_FIGURAS, 'comparacao_f1_por_classe.png'), dpi=200)
plt.close(fig)

print(f"\nTabelas em {PASTA_TABELAS} | figuras em {PASTA_FIGURAS}")
