# =============================================================================
# GERAR TABELAS — junta os resultados de todos os treinos v3 para o relatório
#
# Lê resultados/v3/*/metricas.json (uma pasta por modelo e semente) e gera:
#   docs/tabelas/v3/tabelas.md                ← tabelas prontas em Markdown
#   docs/tabelas/v3/tabela_*.csv              ← para colar no Word/Canva/Excel
#   figuras/v3/<modelo>/*.png                 ← figuras da semente 42
#   figuras/v3/comparacao_f1_por_classe.png   ← teste interno
#   figuras/v3/comparacao_code_test.png       ← modelos x especialistas no CODE-test
#
# As tabelas do protocolo v2 (TCC 1) são geradas pela versão deste script na main.
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

from comum_ecg import PASTA_RAIZ, PASTA_RESULTADOS, PROTOCOLO, DOENCAS, SAIDAS, NOMES_COMPLETOS

PASTA_FIGURAS = os.path.join(PASTA_RAIZ, 'figuras', PROTOCOLO)
PASTA_TABELAS = os.path.join(PASTA_RAIZ, 'docs', 'tabelas', PROTOCOLO)
os.makedirs(PASTA_TABELAS, exist_ok=True)
os.makedirs(PASTA_FIGURAS, exist_ok=True)
ORDEM_MODELOS  = ['transformer', 'transformer_hibrido', 'cnn1d', 'cnn1d_robusto']
CORES          = {'transformer': '#90CAF9', 'transformer_hibrido': '#1565C0', 'cnn1d': '#C62828', 'cnn1d_robusto': '#7B1F1F'}
ROTULOS_CURTOS = {'transformer': 'Transf.', 'transformer_hibrido': 'Híbrido', 'cnn1d': 'CNN', 'cnn1d_robusto': 'CNN rob.'}
CORES_LEITORES = {'DNN Ribeiro et al. (2020)': '#424242', 'Residentes de cardiologia': '#8D6E63',
                  'Residentes de emergência': '#BCAAA4', 'Estudantes de medicina': '#D7CCC8'}


def media_dp(valores, porcentagem=True):
    """'91.2 ± 0.4' quando há várias sementes, '91.2' quando há só uma."""
    v = np.array(valores, dtype=float) * (100 if porcentagem else 1)
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
for arquivo in sorted(glob(os.path.join(PASTA_RESULTADOS, PROTOCOLO, '*', 'metricas.json'))):
    with open(arquivo, encoding='utf-8') as f:
        execucoes.append(json.load(f))

if not execucoes:
    raise SystemExit(f"Nenhum resultados/{PROTOCOLO}/*/metricas.json encontrado — rode os treinos primeiro.")

assinaturas = {e['assinatura_dados'] for e in execucoes}
if len(assinaturas) > 1:
    print(f"AVISO: execuções com dados diferentes ({assinaturas}) — a comparação não é justa.")

por_modelo = {}
for e in execucoes:
    por_modelo.setdefault(e['nome'], []).append(e)
modelos = [m for m in ORDEM_MODELOS if m in por_modelo] + \
          [m for m in por_modelo if m not in ORDEM_MODELOS]
com_code_test = all(e.get('code_test') for e in execucoes)

# ----- Tabela 1: dataset (positivos por classe; um exame pode ter mais de um rótulo) -----
ref = execucoes[0]
tabela_dataset = pd.DataFrame({
    'Classe':    [NOMES_COMPLETOS[s] for s in SAIDAS],
    'Sigla':     SAIDAS,
    'Treino':    [ref['positivos']['treino'][s]    for s in SAIDAS],
    'Validação': [ref['positivos']['validacao'][s] for s in SAIDAS],
    'Teste':     [ref['positivos']['teste'][s]     for s in SAIDAS],
})
tabela_dataset['Total'] = tabela_dataset[['Treino', 'Validação', 'Teste']].sum(axis=1)
tabela_dataset.loc[len(tabela_dataset)] = ['Exames', '', ref['n_treino'], ref['n_validacao'], ref['n_teste'],
                                           ref['n_treino'] + ref['n_validacao'] + ref['n_teste']]

# ----- Tabela 2: comparação geral no teste interno -----
linhas = []
for m in modelos:
    ex = por_modelo[m]
    linhas.append({
        'Modelo':                ex[0]['modelo'],
        'Sementes':              len(ex),
        'AUC macro':             media_dp([e['auc_macro'] for e in ex], porcentagem=False),
        'AP macro':              media_dp([e['ap_macro'] for e in ex], porcentagem=False),
        'F1 macro (%)':          media_dp([e['f1_macro'] for e in ex]),
        'ECE macro':             media_dp([e.get('ece_macro', np.nan) for e in ex], porcentagem=False),
        'Sens. macro (%)':       media_dp([e['sensibilidade_macro'] for e in ex]),
        'Espec. macro (%)':      media_dp([e['especificidade_macro'] for e in ex]),
        'Sens. alterado (%)':    media_dp([e['alterado_x_normal_equilibrado']['sensibilidade'] for e in ex]),
        'Sens. alterado triagem (%)': media_dp([e['alterado_x_normal_triagem']['sensibilidade'] for e in ex]),
        'Parâmetros':            f"{ex[0]['parametros']:,}".replace(',', '.'),
        'Épocas':                (f"{np.mean([e['epocas_treinadas'] for e in ex]):.0f}"
                                  if ex[0]['epocas_treinadas'] else '-'),   # ensemble não treina
    })
tabela_comparacao = pd.DataFrame(linhas)

# ----- Tabelas 3 e 4: por classe no teste interno -----
def tabela_por_classe(extrair, nome, siglas=SAIDAS, porcentagem=True, fonte=lambda e: e['por_classe']):
    linhas = []
    for s in siglas:
        linha = {'Classe': NOMES_COMPLETOS[s]}
        for m in modelos:
            linha[f'{nome} {ROTULOS_CURTOS.get(m, m)}'] = media_dp(
                [extrair(fonte(e)[s]) for e in por_modelo[m]], porcentagem)
        linhas.append(linha)
    return pd.DataFrame(linhas)

tabela_auc  = tabela_por_classe(lambda c: c['auc'], 'AUC', porcentagem=False)
tabela_f1   = tabela_por_classe(lambda c: c['equilibrado']['f1'], 'F1 (%)')
tabela_sens = tabela_por_classe(lambda c: c['equilibrado']['sensibilidade'], 'Sens. (%)')
tabela_espec = tabela_por_classe(lambda c: c['equilibrado']['especificidade'], 'Espec. (%)')
tabela_clinica = pd.concat([tabela_sens, tabela_espec.drop(columns='Classe')], axis=1)

# ----- Tabela 5: CODE-test (externo) com especialistas -----
tabela_code_test = None
if com_code_test:
    linhas = []
    for d in DOENCAS + ['macro']:
        linha = {'Classe': NOMES_COMPLETOS.get(d, 'Média (macro)')}
        for m in modelos:
            linha[f'F1 {ROTULOS_CURTOS.get(m, m)} (%)'] = media_dp(
                [e['code_test']['comparacao_f1']['Este modelo'][d] for e in por_modelo[m]])
        for leitor in CORES_LEITORES:
            linha[f'F1 {leitor} (%)'] = media_dp([ref['code_test']['comparacao_f1'][leitor][d]])
        linhas.append(linha)
    tabela_code_test = pd.DataFrame(linhas)
    tabela_code_test_auc = tabela_por_classe(lambda c: c['auc'], 'AUC', siglas=DOENCAS, porcentagem=False,
                                             fonte=lambda e: e['code_test']['por_classe'])

# ----- Tabela 8: PTB-XL (Alemanha, outro equipamento) -----
tabela_ptbxl = None
if all(e.get('ptbxl') for e in execucoes):
    fonte_px = lambda e: e['ptbxl']['por_classe']
    tabela_ptbxl = pd.concat([
        tabela_por_classe(lambda c: c['auc'], 'AUC', porcentagem=False, fonte=fonte_px),
        tabela_por_classe(lambda c: c['equilibrado']['f1'], 'F1 (%)', fonte=fonte_px).drop(columns='Classe'),
        tabela_por_classe(lambda c: c['equilibrado']['sensibilidade'], 'Sens. (%)', fonte=fonte_px).drop(columns='Classe'),
    ], axis=1)
    tabela_ptbxl.insert(1, 'Positivos', [ref['ptbxl']['por_classe'][s]['suporte'] for s in SAIDAS])

# ----- Tabela 9: robustez no CODE-test (robustez_ecg.py), F1 macro por defeito -----
tabela_robustez = None
colunas_rob = {}
for arquivo in sorted(glob(os.path.join(PASTA_RESULTADOS, PROTOCOLO, '*', 'robustez*.json'))):
    pasta = os.path.dirname(arquivo)
    if not os.path.exists(os.path.join(pasta, 'metricas.json')):
        continue
    with open(os.path.join(pasta, 'metricas.json'), encoding='utf-8') as f:
        rotulo = f"{json.load(f)['modelo']} ({os.path.basename(pasta)})"
    with open(arquivo, encoding='utf-8') as f:
        rob = json.load(f)
    rotulo += ' + passa-alta' if rob.get('filtro') else ''
    colunas_rob[rotulo] = {(r['grupo'], r['condicao']): r['metricas']['f1_macro'] for r in rob['resultados']}
if colunas_rob:
    condicoes = list(next(iter(colunas_rob.values())))
    tabela_robustez = pd.DataFrame([{'Defeito': g, 'Nível': c,
                                     **{k: f"{v[(g, c)] * 100:.1f}" for k, v in colunas_rob.items()}}
                                    for g, c in condicoes])

# ----- Salvar -----
tabelas = [('tabela_dataset.csv', tabela_dataset), ('tabela_comparacao.csv', tabela_comparacao),
           ('tabela_auc_por_classe.csv', tabela_auc), ('tabela_f1_por_classe.csv', tabela_f1),
           ('tabela_sens_espec.csv', tabela_clinica)]
if tabela_code_test is not None:
    tabelas += [('tabela_code_test_f1.csv', tabela_code_test), ('tabela_code_test_auc.csv', tabela_code_test_auc)]
if tabela_ptbxl is not None:
    tabelas += [('tabela_ptbxl.csv', tabela_ptbxl)]
if tabela_robustez is not None:
    tabelas += [('tabela_robustez.csv', tabela_robustez)]
for nome_arquivo, tabela in tabelas:
    tabela.to_csv(os.path.join(PASTA_TABELAS, nome_arquivo), index=False, encoding='utf-8-sig')

total = ref['n_treino'] + ref['n_validacao'] + ref['n_teste']
texto = [
    f'# Tabelas de resultados — IAECG (protocolo {PROTOCOLO})',
    '',
    f"Dados: CODE-15%, {total:,} exames de {ref['pacientes']:,} pacientes".replace(',', '.') +
    f", prevalência real, divisão 70/15/15 por paciente (assinatura `{ref['assinatura_dados']}`). "
    'Multirrótulo: um exame pode ter mais de uma alteração. Limiares escolhidos na validação; '
    'métricas no TESTE. Com mais de uma semente: média ± desvio padrão.',
    '',
    '## Tabela 1 — Exames positivos por classe e subconjunto', '', para_markdown(tabela_dataset), '',
    '## Tabela 2 — Comparação geral no teste interno (6 alterações)', '', para_markdown(tabela_comparacao), '',
    '"Sens. alterado": dos exames com alguma das 6 alterações, quantos o modelo sinaliza com pelo menos um achado. "ECE": erro de calibração (0 = a probabilidade mostrada bate com a frequência real).', '',
    '## Tabela 3 — AUC por classe (teste interno)', '', para_markdown(tabela_auc), '',
    '## Tabela 4 — F1-score por classe (teste interno)', '', para_markdown(tabela_f1), '',
    '## Tabela 5 — Sensibilidade e especificidade por classe (teste interno)', '', para_markdown(tabela_clinica), '',
]
if tabela_code_test is not None:
    texto += ['## Tabela 6 — CODE-test (827 exames, laudo de cardiologistas): F1 dos modelos e de médicos', '',
              para_markdown(tabela_code_test), '',
              '## Tabela 7 — CODE-test: AUC por classe', '', para_markdown(tabela_code_test_auc), '']
if tabela_ptbxl is not None:
    texto += ['## Tabela 8 — PTB-XL (Alemanha, 21.799 exames, outro equipamento): validação externa', '',
              'Limiares os mesmos do CODE-15% (sem reajuste). RBBB/LBBB = códigos CRBBB/CLBBB (bloqueio completo).', '',
              para_markdown(tabela_ptbxl), '']
if tabela_robustez is not None:
    texto += ['## Tabela 9 — Robustez no CODE-test: F1 macro (%) com defeitos comuns do traçado', '',
              'Defeitos aplicados em mV, semente fixa, limiares da validação (robustez_ecg.py). '
              '"+ passa-alta": com o filtro de 0,5 Hz que a inferência aplica.', '',
              para_markdown(tabela_robustez), '']
with open(os.path.join(PASTA_TABELAS, 'tabelas.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(texto))
print('\n'.join(texto))

# ----- Figuras -----
for m in modelos:
    principal = next((e for e in por_modelo[m] if e['semente'] == 42), por_modelo[m][0])
    origem  = os.path.join(PASTA_RESULTADOS, PROTOCOLO, f"{m}_s{principal['semente']}")
    destino = os.path.join(PASTA_FIGURAS, m)
    os.makedirs(destino, exist_ok=True)
    for png in ['matrizes_confusao.png', 'curvas_aprendizado.png', 'comparacao_code_test.png']:
        if os.path.exists(os.path.join(origem, png)):
            shutil.copy(os.path.join(origem, png), os.path.join(destino, png))


def grafico_barras(series, titulo, arquivo, siglas):
    """series: lista de (rótulo, matriz sementes × classes, cor)."""
    fig, ax = plt.subplots(figsize=(13, 5.8))
    largura = 0.8 / len(series)
    x = np.arange(len(siglas))
    for k, (rotulo, valores, cor) in enumerate(series):
        valores = np.atleast_2d(valores) * 100
        barras = ax.bar(x + (k - (len(series) - 1) / 2) * largura, valores.mean(axis=0), largura,
                        yerr=valores.std(axis=0, ddof=1) if len(valores) > 1 else None, capsize=3,
                        label=rotulo, color=cor)
        ax.bar_label(barras, fmt='%.0f', fontsize=7, padding=2)
    ax.set_xticks(x, [NOMES_COMPLETOS[s] for s in siglas], rotation=20, ha='right')
    ax.set_ylabel('F1-score (%)')
    ax.set_ylim(0, 108)
    ax.set_title(titulo, fontweight='bold')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.24), ncol=min(4, len(series)), frameon=False)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_FIGURAS, arquivo), dpi=200)
    plt.close(fig)


grafico_barras([(por_modelo[m][0]['modelo'],
                 [[e['por_classe'][s]['equilibrado']['f1'] for s in SAIDAS] for e in por_modelo[m]],
                 CORES.get(m)) for m in modelos],
               'F1-score por classe no teste interno (prevalência real)', 'comparacao_f1_por_classe.png', SAIDAS)

if com_code_test:
    series = [(por_modelo[m][0]['modelo'],
               [[e['code_test']['comparacao_f1']['Este modelo'][d] for d in DOENCAS] for e in por_modelo[m]],
               CORES.get(m)) for m in modelos]
    series += [(leitor, [ref['code_test']['comparacao_f1'][leitor][d] for d in DOENCAS], cor)
               for leitor, cor in CORES_LEITORES.items()]
    grafico_barras(series, 'CODE-test: F1-score dos modelos e de médicos (laudo de 2+ cardiologistas)',
                   'comparacao_code_test.png', DOENCAS)

print(f"\nTabelas em {PASTA_TABELAS} | figuras em {PASTA_FIGURAS}")
