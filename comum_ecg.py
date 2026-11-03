# =============================================================================
# COMUM — dados, treino e avaliação compartilhados pelos modelos (protocolo v3)
#
# transformer_ecg.py e cnn1d_ecg.py só definem a arquitetura (Etapa 5) e
# chamam treinar_e_avaliar(). Todo o resto mora aqui, então a comparação
# Transformer x CNN usa garantidamente os mesmos exames, a mesma divisão,
# o mesmo augmentation, a mesma loss, o mesmo agendador e o mesmo early stopping.
#
# Protocolo v3 (TCC 2) — perto do uso real:
#   - TODOS os exames do CODE-15% disponíveis, com a prevalência natural
#     (nada de balanceamento nem de descarte de exames com outras alterações);
#   - multirrótulo: uma sigmoide por doença + uma para "ECG normal". Um exame
#     pode ter duas doenças, e um exame sem as 6 doenças pode não ser normal
#     (outra alteração) — o modelo precisa saber dizer isso;
#   - limiar de decisão por classe escolhido na validação (F1 máximo e modo
#     triagem, com sensibilidade alvo);
#   - teste interno (CODE-15%, por paciente) + teste externo no CODE-test,
#     comparado com residentes, estudantes e a DNN de Ribeiro et al. (2020).
# O protocolo v2 (balanceado, 7 classes exclusivas) do TCC 1 está na main.
#
# Saídas de cada execução em resultados/v3/<nome>_s<semente>/ (os da v2 ficam
# em resultados/<nome>_s<semente>/; testes com --limite vão em resultados/teste_rapido/):
#   metricas.json, curvas_aprendizado.png, matrizes_confusao.png,
#   comparacao_code_test.png, melhor_modelo.pt, checkpoint.pt, modelo_final.pt
# =============================================================================

import os
import sys
import json
import math
import time
import hashlib
from glob import glob
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import h5py
import matplotlib
matplotlib.use('Agg')                        # salva figuras sem abrir janela
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

# =============================================================================
# CONFIGURAÇÕES GERAIS
# =============================================================================

PASTA_RAIZ       = os.path.dirname(os.path.abspath(__file__))
PASTA_DADOS      = os.path.join(PASTA_RAIZ, 'dados')
ARQUIVO_CSV      = os.path.join(PASTA_DADOS, 'exams.csv')
PASTA_CODE_TEST  = os.path.join(PASTA_DADOS, 'code_test', 'data')
PASTA_PTBXL      = os.path.join(PASTA_DADOS, 'ptbxl')
PASTA_RESULTADOS = os.path.join(PASTA_RAIZ, 'resultados')

PROTOCOLO = 'v3'

DOENCAS = ['1dAVb', 'RBBB', 'LBBB', 'SB', 'ST', 'AF']
SAIDAS  = DOENCAS + ['Normal']               # 7 sigmoides independentes
NUM_SAIDAS = len(SAIDAS)

NOMES_COMPLETOS = {
    '1dAVb':  'Bloqueio AV 1º Grau',
    'RBBB':   'Bloqueio Ramo Direito',
    'LBBB':   'Bloqueio Ramo Esquerdo',
    'SB':     'Bradicardia Sinusal',
    'ST':     'Taquicardia Sinusal',
    'AF':     'Fibrilação Atrial',
    'Normal': 'ECG Normal',
}
NOMES_SAIDAS = [NOMES_COMPLETOS[s] for s in SAIDAS]

# Pré-processamento (o mesmo é aplicado na inferência — ver inferencia_ecg.py)
FREQUENCIA_HZ  = 400
NUM_AMOSTRAS   = 4096                        # ~10 s a 400 Hz, com zeros nas pontas
NUM_DERIVACOES = 12

# Hiperparâmetros de treino (iguais para todos os modelos, exceto a LR)
TAMANHO_BATCH    = 64
EPOCAS           = 40     # com ~240 mil exames de treino, cada época já vê muitos dados
PACIENCIA        = 8
REDUCAO_LR       = 3      # LR ÷ 10 a cada 3 épocas sem melhora (2 reduções antes de parar)
MIN_EPOCAS       = 10
SEMENTE_DIVISAO  = 42     # fixa: todos os modelos (e todas as sementes) usam o mesmo split
SENSIBILIDADE_TRIAGEM = 0.95                 # alvo do limiar "triagem", escolhido na validação

# Logs redirecionados para arquivo no Windows usam cp1252 e quebram com ✓ / ⏹
sys.stdout.reconfigure(encoding='utf-8')

DISPOSITIVO = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USA_AMP     = DISPOSITIVO.type == 'cuda'     # bfloat16 na GPU: ~2× mais rápido, mesma acurácia

# =============================================================================
# ETAPA 1 — METADADOS, RÓTULOS E DIVISÃO POR PACIENTE
# =============================================================================

def carregar_metadados(arquivo_csv, pasta_dados, partes=None):
    """
    Lê o exams.csv e mantém só os exames cujo HDF5 está em dados/.
    Acrescenta a posição de cada exame dentro do HDF5 e ordena o DataFrame
    na ordem dos arquivos — assim o cache é escrito e lido em sequência.
    """
    df = pd.read_csv(arquivo_csv)
    arquivos = sorted((os.path.basename(p) for p in glob(os.path.join(pasta_dados, 'exams_part*.hdf5'))),
                      key=lambda n: int(n.split('part')[1].split('.')[0]))
    if partes is not None:                      # ex.: refazer um treino feito com menos partes
        arquivos = [n for n in arquivos if int(n.split('part')[1].split('.')[0]) in partes]
    print(f"Arquivos HDF5 encontrados ({len(arquivos)}): {arquivos}")

    posicoes = []
    for nome in arquivos:
        with h5py.File(os.path.join(pasta_dados, nome), 'r') as f:
            ids = f['exam_id'][:]
        posicoes.append(pd.DataFrame({'exam_id': ids, 'trace_file': nome,
                                      'posicao_hdf5': np.arange(len(ids))}))
    posicoes = pd.concat(posicoes)

    df = df.merge(posicoes, on=['exam_id', 'trace_file'], how='inner')
    df['ordem_arquivo'] = df['trace_file'].map({n: i for i, n in enumerate(arquivos)})
    df = (df.sort_values(['ordem_arquivo', 'posicao_hdf5'])
            .drop(columns='ordem_arquivo').reset_index(drop=True))
    print(f"Total de exames disponíveis: {len(df):,} de {df['patient_id'].nunique():,} pacientes")
    return df


def montar_rotulos(df):
    """Matriz (N, 7) de 0/1: as 6 doenças do laudo + normal_ecg."""
    rotulos = df[DOENCAS].to_numpy(dtype=np.float32)
    normal  = df['normal_ecg'].to_numpy(dtype=np.float32)[:, None]
    return np.concatenate([rotulos, normal], axis=1)


def chave_estratificacao(rotulos):
    """
    Uma classe por exame só para estratificar a divisão: a doença mais rara
    que o exame tiver; senão 'normal' (6) ou 'outra alteração' (7). Garante
    que as doenças raras fiquem proporcionais em treino/validação/teste.
    """
    prevalencia = rotulos[:, :len(DOENCAS)].mean(axis=0)
    ordem_rara  = np.argsort(prevalencia)
    chave = np.where(rotulos[:, len(DOENCAS)] == 1, len(DOENCAS), len(DOENCAS) + 1)
    for d in ordem_rara[::-1]:                  # a mais rara sobrescreve por último
        chave = np.where(rotulos[:, d] == 1, d, chave)
    return chave


def dividir_por_paciente(df, rotulos):
    """70/15/15 estratificado e agrupado por paciente (nenhum paciente em dois conjuntos)."""
    divisor = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=SEMENTE_DIVISAO)
    dobra   = np.empty(len(df), dtype=int)
    for i, (_, indices) in enumerate(divisor.split(df, chave_estratificacao(rotulos), df['patient_id'])):
        dobra[indices] = i
    return np.select([dobra < 3, dobra < 6], ['teste', 'validacao'], 'treino')

# =============================================================================
# ETAPA 2 — CACHE DOS SINAIS EM DISCO (memmap float16)
# =============================================================================

def normalizar_sinais(sinais):
    """Z-score por derivação, por exame. sinais: (..., 4096, 12)."""
    media  = sinais.mean(axis=-2, keepdims=True)
    desvio = sinais.std(axis=-2, keepdims=True)
    return ((sinais - media) / (desvio + 1e-8)).astype(np.float32)


def construir_cache(df, arquivo_cache, tamanho_bloco=1000):
    """
    Grava todos os sinais já normalizados num .npy float16 (~98 KB por exame;
    ~34 GB para o CODE-15% inteiro). O arquivo é lido por memmap no treino,
    então não precisa caber na RAM. Grava em .tmp e renomeia no fim, para
    um cache incompleto nunca ser usado.
    """
    temporario = arquivo_cache + '.tmp'
    cache = np.lib.format.open_memmap(temporario, mode='w+', dtype=np.float16,
                                      shape=(len(df), NUM_AMOSTRAS, NUM_DERIVACOES))
    for nome_arquivo, grupo in df.groupby('trace_file', sort=False):
        print(f"  Lendo {nome_arquivo} ({len(grupo):,} exames)...", flush=True)
        linhas   = grupo.index.to_numpy()
        posicoes = grupo['posicao_hdf5'].to_numpy()
        with h5py.File(os.path.join(PASTA_DADOS, nome_arquivo), 'r') as f:
            for inicio in range(0, len(posicoes), tamanho_bloco):
                p = posicoes[inicio:inicio + tamanho_bloco]
                bloco = f['tracings'][p.min():p.max() + 1]      # leitura contínua, rápida
                cache[linhas[inicio:inicio + tamanho_bloco]] = normalizar_sinais(bloco[p - p.min()])
    cache.flush()
    del cache
    os.replace(temporario, arquivo_cache)


def preparar_dados(limite_exames=None, partes=None):
    """
    Retorna dict com o memmap dos sinais, os rótulos (N, 7) e os índices de
    treino/validação/teste. limite_exames: subamostra para testes rápidos.
    """
    df = carregar_metadados(ARQUIVO_CSV, PASTA_DADOS, partes)
    if limite_exames and limite_exames < len(df):
        df = (df.sample(limite_exames, random_state=SEMENTE_DIVISAO)
                .sort_index().reset_index(drop=True))
        print(f"Subamostra para teste rápido: {len(df):,} exames")

    rotulos = montar_rotulos(df)
    df['particao'] = dividir_por_paciente(df, rotulos)

    print("\nPrevalência por classe (todos os exames):")
    for j, nome in enumerate(NOMES_SAIDAS):
        print(f"  {nome:25s}: {int(rotulos[:, j].sum()):7,} ({rotulos[:, j].mean():.2%})")
    sem_doenca = rotulos[:, :len(DOENCAS)].sum(axis=1) == 0
    print(f"  {'(outra alteração)':25s}: {int((sem_doenca & (rotulos[:, -1] == 0)).sum()):7,}")
    print(f"  {'(2+ doenças)':25s}: {int((rotulos[:, :len(DOENCAS)].sum(axis=1) > 1).sum()):7,}")

    assinatura = hashlib.md5(PROTOCOLO.encode() + df['exam_id'].to_numpy().tobytes()).hexdigest()[:10]
    arquivo_cache = os.path.join(PASTA_DADOS, f'cache_sinais_{PROTOCOLO}_{assinatura}.npy')
    if not os.path.exists(arquivo_cache):
        print(f"\nCriando cache de sinais ({len(df) * 98 / 2**20:.1f} GB): {arquivo_cache}")
        construir_cache(df, arquivo_cache)
    print(f"\nUsando cache de sinais: {arquivo_cache}")

    dados = {'assinatura': assinatura, 'df': df, 'rotulos': rotulos, 'rapido': bool(limite_exames),
             'sinais': np.load(arquivo_cache, mmap_mode='r')}
    for particao in ['treino', 'validacao', 'teste']:
        dados[particao] = np.flatnonzero((df['particao'] == particao).to_numpy())

    print(f"\nTreino: {len(dados['treino']):,} | Validação: {len(dados['validacao']):,} "
          f"| Teste: {len(dados['teste']):,}  (sem paciente repetido entre conjuntos)")
    return dados


# O CODE-test grava as aumentadas como aVL, aVF, aVR (conferido pela lei de
# Einthoven: aVR = -(DI+DII)/2); o CODE-15% usa aVR, aVL, aVF. Reordena para o do treino.
ORDEM_CODE_TEST = [0, 1, 2, 5, 3, 4, 6, 7, 8, 9, 10, 11]


def carregar_code_test():
    """CODE-test: 827 exames de pacientes fora do CODE-15%, com padrão-ouro de 2+ cardiologistas."""
    caminho = os.path.join(PASTA_CODE_TEST, 'ecg_tracings.hdf5')
    if not os.path.exists(caminho):
        return None
    with h5py.File(caminho, 'r') as f:
        sinais = normalizar_sinais(f['tracings'][:][..., ORDEM_CODE_TEST])
    ouro = pd.read_csv(os.path.join(PASTA_CODE_TEST, 'annotations', 'gold_standard.csv'))
    return sinais, ouro[DOENCAS].to_numpy(dtype=np.float32)


# Códigos SCP do PTB-XL equivalentes às nossas saídas. RBBB/LBBB = bloqueio completo
# (QRS >= 120 ms), como no CODE; o incompleto (IRBBB, ILBBB) não conta.
CODIGOS_PTBXL = {'1dAVb': '1AVB', 'RBBB': 'CRBBB', 'LBBB': 'CLBBB', 'SB': 'SBRAD',
                 'ST': 'STACH', 'AF': 'AFIB', 'Normal': 'NORM'}


def carregar_ptbxl():
    """
    PTB-XL (Alemanha, 21.799 ECGs de 10 s a 500 Hz, laudo de cardiologistas): validação
    externa com outro equipamento e outra população. Nenhum exame foi usado no treino,
    então entram todos. Reamostra para 400 Hz e centraliza em 4.096 amostras;
    cache em dados/ptbxl/cache_ptbxl.npy (float16, ~2 GB) para não reler os WFDB.
    """
    import ast
    from scipy.signal import resample_poly
    arquivo_csv = os.path.join(PASTA_PTBXL, 'ptbxl_database.csv')
    if not os.path.exists(arquivo_csv):
        return None
    df = pd.read_csv(arquivo_csv, index_col='ecg_id')
    codigos = df['scp_codes'].map(ast.literal_eval)
    rotulos = np.array([[float(CODIGOS_PTBXL[s] in c) for s in SAIDAS] for c in codigos], dtype=np.float32)

    arquivo_cache = os.path.join(PASTA_PTBXL, 'cache_ptbxl.npy')
    if not os.path.exists(arquivo_cache):
        import wfdb
        print(f"Lendo {len(df):,} exames do PTB-XL (só na 1ª vez)...", flush=True)
        cache = np.lib.format.open_memmap(arquivo_cache + '.tmp', mode='w+', dtype=np.float16,
                                          shape=(len(df), NUM_AMOSTRAS, NUM_DERIVACOES))
        for i, caminho in enumerate(df['filename_hr']):
            registro = wfdb.rdrecord(os.path.join(PASTA_PTBXL, caminho))
            assert [n.upper() for n in registro.sig_name] == \
                ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
            sinal = resample_poly(np.nan_to_num(registro.p_signal), 4, 5, axis=0)   # 500 → 400 Hz
            falta = NUM_AMOSTRAS - len(sinal)
            sinal = np.pad(sinal, ((falta // 2, falta - falta // 2), (0, 0)))
            cache[i] = normalizar_sinais(sinal)
        cache.flush()
        del cache
        os.replace(arquivo_cache + '.tmp', arquivo_cache)
    return np.load(arquivo_cache, mmap_mode='r'), rotulos

# =============================================================================
# ETAPA 4 — CARREGADOR DE LOTES E DATA AUGMENTATION
# =============================================================================

class CarregadorLotes:
    """
    Lê lotes direto do memmap com threads em paralelo (pré-busca), para a GPU
    não esperar o disco. Os índices de cada lote são ordenados para a leitura
    ser o mais sequencial possível.
    """
    def __init__(self, sinais, indices, rotulos, embaralhar=False, semente=0, pre_busca=8):
        self.sinais, self.indices, self.rotulos = sinais, np.asarray(indices), rotulos
        self.embaralhar, self.pre_busca = embaralhar, pre_busca
        self.gerador = np.random.default_rng(semente)

    def __len__(self):
        return math.ceil(len(self.indices) / TAMANHO_BATCH)

    def _ler_lote(self, idx):
        x = torch.from_numpy(np.asarray(self.sinais[idx]))
        y = torch.from_numpy(self.rotulos[idx])
        if torch.cuda.is_available():
            x, y = x.pin_memory(), y.pin_memory()
        return x, y

    def __iter__(self):
        ordem = self.gerador.permutation(self.indices) if self.embaralhar else self.indices
        lotes = [np.sort(ordem[i:i + TAMANHO_BATCH]) for i in range(0, len(ordem), TAMANHO_BATCH)]
        with ThreadPoolExecutor(max_workers=4) as executor:
            pendentes = deque()
            for lote in lotes:
                pendentes.append(executor.submit(self._ler_lote, lote))
                if len(pendentes) >= self.pre_busca:
                    yield pendentes.popleft().result()
            while pendentes:
                yield pendentes.popleft().result()


def lotes_em_memoria(sinais):
    """Lotes de um array já em memória (inferência / CODE-test)."""
    for i in range(0, len(sinais), TAMANHO_BATCH):
        yield torch.from_numpy(np.ascontiguousarray(sinais[i:i + TAMANHO_BATCH]))


def aumentar_batch(sinais):
    """
    Augmentation aplicado direto na GPU, por exame:
      - ruído gaussiano leve (σ = 0.02)
      - escala de amplitude entre 0.9 e 1.1
      - deslocamento temporal de até ±0.5 s (200 amostras a 400 Hz)
    Sem inversão de polaridade: ela inverte a morfologia do QRS, que é
    justamente o que diferencia bloqueio de ramo direito de esquerdo.
    """
    b = sinais.size(0)
    sinais = sinais + torch.randn_like(sinais) * 0.02
    sinais = sinais * (0.9 + 0.2 * torch.rand(b, 1, 1, device=sinais.device))
    deslocamento = int(torch.randint(-200, 201, (1,)))
    return torch.roll(sinais, shifts=deslocamento, dims=1)


@torch.no_grad()
def prever(modelo, lotes):
    """Probabilidades (N, 7) para um iterável de lotes de sinais."""
    modelo.eval()
    saidas = []
    for x in lotes:
        x = x[0] if isinstance(x, (tuple, list)) else x
        x = x.to(DISPOSITIVO, non_blocking=True).float()
        with torch.autocast(DISPOSITIVO.type, dtype=torch.bfloat16, enabled=USA_AMP):
            logits = modelo(x)
        saidas.append(torch.sigmoid(logits.float()).cpu())
    return torch.cat(saidas).numpy()

# =============================================================================
# ETAPA 9 — MÉTRICAS MULTIRRÓTULO E LIMIARES
# =============================================================================

def escolher_limiares(probs, rotulos):
    """
    Dois limiares por classe, escolhidos SÓ na validação:
      equilibrado: maximiza o F1 (mesmo critério de Ribeiro et al., 2020);
      triagem:     maior limiar com sensibilidade >= SENSIBILIDADE_TRIAGEM
                   (perder menos doentes, aceitando mais alertas falsos).
    """
    equilibrado, triagem = [], []
    for j in range(rotulos.shape[1]):
        if rotulos[:, j].sum() == 0:
            equilibrado.append(0.5); triagem.append(0.5)
            continue
        precisao, sensibilidade, limiares = precision_recall_curve(rotulos[:, j], probs[:, j])
        f1 = 2 * precisao * sensibilidade / np.maximum(precisao + sensibilidade, 1e-12)
        equilibrado.append(float(limiares[np.argmax(f1[:-1])]))
        ok = np.flatnonzero(sensibilidade[:-1] >= SENSIBILIDADE_TRIAGEM)
        limiar_triagem = float(limiares[ok[-1]]) if len(ok) else float(limiares[0])
        triagem.append(min(limiar_triagem, equilibrado[-1]))   # nunca menos sensível que o equilibrado
    return {'equilibrado': equilibrado, 'triagem': triagem}


def metricas_por_limiar(y, p, limiar):
    previsto = p >= limiar
    vp = int((previsto & (y == 1)).sum()); fp = int((previsto & (y == 0)).sum())
    fn = int((~previsto & (y == 1)).sum()); vn = int((~previsto & (y == 0)).sum())
    sens = vp / max(vp + fn, 1); prec = vp / max(vp + fp, 1)
    return {
        'sensibilidade':  sens,
        'especificidade': vn / max(vn + fp, 1),
        'precisao':       prec,                          # valor preditivo positivo
        'vpn':            vn / max(vn + fn, 1),
        'f1':             2 * prec * sens / max(prec + sens, 1e-12),
        'matriz':         [[vn, fp], [fn, vp]],
    }


def erro_calibracao(y, p, faixas=10):
    """
    ECE: diferença média entre a probabilidade dada e a frequência real, em 10
    faixas de probabilidade (ponderada pelo nº de exames). 0 = "70%" acerta 70% das vezes.
    """
    faixa = np.minimum((p * faixas).astype(int), faixas - 1)
    erro = 0.0
    for b in range(faixas):
        dentro = faixa == b
        if dentro.any():
            erro += dentro.mean() * abs(p[dentro].mean() - y[dentro].mean())
    return float(erro)


def avaliar(probs, rotulos, limiares, nomes=SAIDAS):
    """Métricas por classe (AUC, AP e as do limiar) + médias macro das 6 doenças."""
    por_classe = {}
    for j, sigla in enumerate(nomes):
        y, p = rotulos[:, j], probs[:, j]
        tem_os_dois = 0 < y.sum() < len(y)
        por_classe[sigla] = {
            'nome':        NOMES_COMPLETOS[sigla],
            'suporte':     int(y.sum()),
            'prevalencia': float(y.mean()),
            'auc':         float(roc_auc_score(y, p)) if tem_os_dois else None,
            'ap':          float(average_precision_score(y, p)) if tem_os_dois else None,
            'ece':         erro_calibracao(y, p),
            'equilibrado': metricas_por_limiar(y, p, limiares['equilibrado'][j]),
            'triagem':     metricas_por_limiar(y, p, limiares['triagem'][j]),
        }

    doencas = [s for s in nomes if s in DOENCAS and por_classe[s]['auc'] is not None]
    media = lambda f: float(np.mean([f(por_classe[s]) for s in doencas]))
    resumo = {
        'auc_macro':  media(lambda c: c['auc']),
        'ap_macro':   media(lambda c: c['ap']),
        'ece_macro':  media(lambda c: c['ece']),
        'f1_macro':   media(lambda c: c['equilibrado']['f1']),
        'sensibilidade_macro':  media(lambda c: c['equilibrado']['sensibilidade']),
        'especificidade_macro': media(lambda c: c['equilibrado']['especificidade']),
        'f1_macro_triagem':            media(lambda c: c['triagem']['f1']),
        'sensibilidade_macro_triagem': media(lambda c: c['triagem']['sensibilidade']),
        'especificidade_macro_triagem': media(lambda c: c['triagem']['especificidade']),
    }

    # Decisão de triagem: o exame tem alguma das 6 doenças? (o que importa na UPA)
    idx = [nomes.index(s) for s in DOENCAS if s in nomes]
    alterado = rotulos[:, idx].max(axis=1)
    for modo in ['equilibrado', 'triagem']:
        lim = np.array([limiares[modo][j] for j in idx])
        resumo[f'alterado_x_normal_{modo}'] = metricas_por_limiar(
            alterado, (probs[:, idx] >= lim).max(axis=1).astype(float), 0.5)
    return resumo, por_classe


def comparar_com_especialistas(probs, rotulos, limiares):
    """F1 por doença no CODE-test: modelo x residentes, estudantes e a DNN de Ribeiro et al."""
    pasta = os.path.join(PASTA_CODE_TEST, 'annotations')
    leitores = {'Este modelo': (probs[:, :len(DOENCAS)] >= np.array(limiares['equilibrado'][:len(DOENCAS)])).astype(float)}
    for arquivo, rotulo in [('dnn.csv', 'DNN Ribeiro et al. (2020)'),
                            ('cardiology_residents.csv', 'Residentes de cardiologia'),
                            ('emergency_residents.csv', 'Residentes de emergência'),
                            ('medical_students.csv', 'Estudantes de medicina')]:
        leitores[rotulo] = pd.read_csv(os.path.join(pasta, arquivo))[DOENCAS].to_numpy(dtype=float)

    comparacao = {}
    for rotulo, previsto in leitores.items():
        comparacao[rotulo] = {d: metricas_por_limiar(rotulos[:, j], previsto[:, j], 0.5)['f1']
                              for j, d in enumerate(DOENCAS)}
        comparacao[rotulo]['macro'] = float(np.mean([comparacao[rotulo][d] for d in DOENCAS]))
    return comparacao

# =============================================================================
# ETAPA 6 a 12 — TREINO, CHECKPOINT, AVALIAÇÃO E FIGURAS
# =============================================================================

def treinar_e_avaliar(nome, titulo, modelo, taxa_aprendizado, dados, semente, configuracoes):
    """
    nome:   identificador curto ('transformer', 'cnn1d') — vira a pasta de saída
    titulo: nome legível para os gráficos ('Transformer 1D', 'CNN 1D (ResNet)')
    """
    subpasta    = 'teste_rapido' if dados['rapido'] else PROTOCOLO
    pasta_saida = os.path.join(PASTA_RESULTADOS, subpasta, f'{nome}_s{semente}')
    os.makedirs(pasta_saida, exist_ok=True)
    arquivo_checkpoint = os.path.join(pasta_saida, 'checkpoint.pt')
    arquivo_melhor     = os.path.join(pasta_saida, 'melhor_modelo.pt')

    modelo = modelo.to(DISPOSITIVO)
    total_parametros = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    print(f"\n{titulo} | Parâmetros treináveis: {total_parametros:,} | Dispositivo: {DISPOSITIVO}")

    sinais, rotulos = dados['sinais'], dados['rotulos']
    carregador_treino    = CarregadorLotes(sinais, dados['treino'], rotulos, embaralhar=True, semente=semente)
    carregador_validacao = CarregadorLotes(sinais, dados['validacao'], rotulos)
    rotulos_val = rotulos[dados['validacao']]

    # ----- Etapa 6: loss, otimizador e agendador -----
    passos_por_epoca = len(carregador_treino)
    passos_warmup    = max(50, min(200, passos_por_epoca // 2))

    funcao_perda = nn.BCEWithLogitsLoss()
    otimizador   = torch.optim.AdamW(modelo.parameters(), lr=taxa_aprendizado, weight_decay=1e-3)

    # Warmup linear e depois LR ÷ 10 a cada REDUCAO_LR épocas sem melhora (como em
    # Ribeiro et al., 2020). Com a base inteira o early stopping chega cedo, e um
    # cosseno longo pararia com a LR ainda alta.
    escala = {'lr': 1.0}

    def calcular_taxa_aprendizado(passo_atual):
        return min(1.0, passo_atual / max(1, passos_warmup)) * escala['lr']

    agendador = torch.optim.lr_scheduler.LambdaLR(otimizador, calcular_taxa_aprendizado)

    # ----- Etapa 7: checkpoint (retoma de onde parou) -----
    historico = {'perda_treino': [], 'perda_val': [], 'ap_val': [], 'auc_val': [], 'tempo_epoca': []}
    epoca_inicial, melhor_ap, contador_paciencia = 1, 0.0, 0

    if os.path.exists(arquivo_checkpoint):
        estado = torch.load(arquivo_checkpoint, map_location=DISPOSITIVO)
        if estado.get('assinatura') == dados['assinatura']:
            modelo.load_state_dict(estado['modelo'])
            otimizador.load_state_dict(estado['otimizador'])
            agendador.load_state_dict(estado['agendador'])
            historico          = estado['historico']
            epoca_inicial      = estado['epoca'] + 1
            melhor_ap          = estado['melhor_ap']
            contador_paciencia = estado['contador_paciencia']
            escala['lr']       = estado.get('escala_lr', 1.0)
            print(f"Retomando da época {epoca_inicial} | Melhor AP val: {melhor_ap:.4f}")
        else:
            print("Checkpoint é de outro conjunto de dados — ignorando e começando do zero.")

    def salvar_checkpoint(epoca):
        torch.save({
            'epoca': epoca, 'assinatura': dados['assinatura'],
            'modelo': modelo.state_dict(), 'otimizador': otimizador.state_dict(),
            'agendador': agendador.state_dict(), 'melhor_ap': melhor_ap,
            'contador_paciencia': contador_paciencia, 'historico': historico,
            'escala_lr': escala['lr'],
        }, arquivo_checkpoint)

    # ----- Etapa 8: loop de treino -----
    def epoca_de_treino():
        modelo.train()
        perda_total, aviso = 0.0, max(1, passos_por_epoca // 5)
        for passo, (x, y) in enumerate(carregador_treino, start=1):
            x = aumentar_batch(x.to(DISPOSITIVO, non_blocking=True).float())
            y = y.to(DISPOSITIVO, non_blocking=True)
            with torch.autocast(DISPOSITIVO.type, dtype=torch.bfloat16, enabled=USA_AMP):
                logits = modelo(x)
            perda = funcao_perda(logits.float(), y)
            otimizador.zero_grad(set_to_none=True)
            perda.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
            otimizador.step()
            agendador.step()
            perda_total += perda.item()
            if passo % aviso == 0 and passo < passos_por_epoca:
                print(f"    {passo / passos_por_epoca:.0%} da época | perda {perda_total / passo:.4f}", flush=True)
        return perda_total / passos_por_epoca

    def perda_bce(probs, y):
        p = np.clip(probs, 1e-7, 1 - 1e-7)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    print(f"\n=== INÍCIO DO TREINO (época {epoca_inicial} até {EPOCAS}) ===")
    print(f"Early stopping pela AP macro de validação: paciência de {PACIENCIA} | mínimo de {MIN_EPOCAS}\n")

    for epoca in range(epoca_inicial, EPOCAS + 1):
        inicio = time.time()
        perda_tr  = epoca_de_treino()
        probs_val = prever(modelo, carregador_validacao)
        resumo_val, _ = avaliar(probs_val, rotulos_val, escolher_limiares(probs_val, rotulos_val))
        duracao = time.time() - inicio

        historico['perda_treino'].append(perda_tr)
        historico['perda_val'].append(perda_bce(probs_val, rotulos_val))
        historico['ap_val'].append(resumo_val['ap_macro'])
        historico['auc_val'].append(resumo_val['auc_macro'])
        historico['tempo_epoca'].append(duracao)

        melhorou = resumo_val['ap_macro'] > melhor_ap
        if melhorou:
            melhor_ap          = resumo_val['ap_macro']
            contador_paciencia = 0
            torch.save(modelo.state_dict(), arquivo_melhor)
        else:
            contador_paciencia += 1
            if contador_paciencia % REDUCAO_LR == 0:
                escala['lr'] *= 0.1
                for grupo in otimizador.param_groups:          # aplica já (o LambdaLR segue a escala)
                    grupo['lr'] = taxa_aprendizado * calcular_taxa_aprendizado(agendador.last_epoch)

        print(f"Época {epoca:03d} | Perda tr: {perda_tr:.4f} | Perda val: {historico['perda_val'][-1]:.4f} "
              f"AP val: {resumo_val['ap_macro']:.4f} AUC val: {resumo_val['auc_macro']:.4f} "
              f"F1 val: {resumo_val['f1_macro']:.4f} | LR: {otimizador.param_groups[0]['lr']:.2e} | {duracao:.0f}s"
              + (" ✓" if melhorou else f" (paciência {contador_paciencia}/{PACIENCIA})"),
              flush=True)

        parar = epoca >= MIN_EPOCAS and contador_paciencia >= PACIENCIA
        salvar_checkpoint(epoca)                       # épocas longas: salva todas
        if parar:
            print(f"\n⏹ Early stopping na época {epoca}. Melhor AP de validação: {melhor_ap:.4f}")
            break

    modelo.load_state_dict(torch.load(arquivo_melhor, map_location=DISPOSITIVO))
    return avaliacao_final(nome, titulo, modelo, taxa_aprendizado, dados, semente, configuracoes,
                           historico, melhor_ap, pasta_saida)


def salvar_curvas(historico, melhor_epoca, titulo, pasta_saida):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epocas_executadas = range(1, len(historico['perda_treino']) + 1)
    ax1.plot(epocas_executadas, historico['perda_treino'], label='Treino',    color='#1565C0')
    ax1.plot(epocas_executadas, historico['perda_val'],    label='Validação', color='#C62828')
    ax1.set_title('Curva de Perda (BCE)', fontweight='bold'); ax1.set_ylabel('Perda')
    ax2.plot(epocas_executadas, historico['ap_val'],  label='AP macro (val)',  color='#C62828')
    ax2.plot(epocas_executadas, historico['auc_val'], label='AUC macro (val)', color='#2E7D32')
    ax2.set_title('Métricas de validação', fontweight='bold'); ax2.set_ylabel('Valor')
    for ax in (ax1, ax2):
        ax.axvline(melhor_epoca, color='gray', ls='--', lw=1, label='Melhor época')
        ax.set_xlabel('Época'); ax.legend(); ax.grid(alpha=0.3)
    plt.suptitle(f'Histórico de Treinamento — {titulo}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(pasta_saida, 'curvas_aprendizado.png'), dpi=200)
    plt.close(fig)


def avaliacao_final(nome, titulo, modelo, taxa_aprendizado, dados, semente, configuracoes,
                    historico, melhor_ap, pasta_saida):
    """
    Etapas 9 a 12 com os melhores pesos já carregados no modelo. Separada do
    treino para reavaliar_modelo.py poder refazê-la sem treinar de novo.
    """
    print("\n=== AVALIAÇÃO FINAL (melhores pesos) ===")
    modelo = modelo.to(DISPOSITIVO)
    total_parametros = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    sinais, rotulos = dados['sinais'], dados['rotulos']
    carregador_validacao = CarregadorLotes(sinais, dados['validacao'], rotulos)
    carregador_teste     = CarregadorLotes(sinais, dados['teste'], rotulos)
    rotulos_val, rotulos_teste = rotulos[dados['validacao']], rotulos[dados['teste']]

    # ----- Etapa 9: limiares na validação, avaliação no teste e no CODE-test -----

    probs_val = prever(modelo, carregador_validacao)
    limiares  = escolher_limiares(probs_val, rotulos_val)

    inicio = time.time()
    probs_teste = prever(modelo, carregador_teste)
    tempo_inferencia_ms = 1000 * (time.time() - inicio) / len(probs_teste)
    resumo_teste, por_classe_teste = avaliar(probs_teste, rotulos_teste, limiares)

    print(f"{'Classe':25s} {'AUC':>6s} {'AP':>6s} {'Sens':>6s} {'Espec':>6s} {'Prec':>6s} {'F1':>6s}")
    for sigla, c in por_classe_teste.items():
        e = c['equilibrado']
        print(f"{c['nome']:25s} {c['auc']:6.3f} {c['ap']:6.3f} {e['sensibilidade']:6.3f} "
              f"{e['especificidade']:6.3f} {e['precisao']:6.3f} {e['f1']:6.3f}")
    print(f"Macro (6 doenças): AUC {resumo_teste['auc_macro']:.4f} | AP {resumo_teste['ap_macro']:.4f} "
          f"| F1 {resumo_teste['f1_macro']:.4f}")

    code_test = None
    externo = carregar_code_test()
    if externo is not None:
        sinais_ct, rotulos_ct = externo
        probs_ct = prever(modelo, lotes_em_memoria(sinais_ct))[:, :len(DOENCAS)]
        limiares_ct = {m: v[:len(DOENCAS)] for m, v in limiares.items()}
        resumo_ct, por_classe_ct = avaliar(probs_ct, rotulos_ct, limiares_ct, nomes=DOENCAS)
        code_test = {'resumo': resumo_ct, 'por_classe': por_classe_ct,
                     'comparacao_f1': comparar_com_especialistas(probs_ct, rotulos_ct, limiares)}
        print(f"\nCODE-test (externo, {len(rotulos_ct)} exames): AUC {resumo_ct['auc_macro']:.4f} "
              f"| AP {resumo_ct['ap_macro']:.4f} | F1 {resumo_ct['f1_macro']:.4f}")
        for leitor, f1 in code_test['comparacao_f1'].items():
            print(f"  F1 macro — {leitor:28s}: {f1['macro']:.3f}")

    ptbxl = None
    externo = carregar_ptbxl()
    if externo is not None:
        sinais_px, rotulos_px = externo
        probs_px = prever(modelo, lotes_em_memoria(sinais_px))
        resumo_px, por_classe_px = avaliar(probs_px, rotulos_px, limiares)
        ptbxl = {'resumo': resumo_px, 'por_classe': por_classe_px}
        print(f"\nPTB-XL (externo, Alemanha, {len(rotulos_px):,} exames): AUC {resumo_px['auc_macro']:.4f} "
              f"| AP {resumo_px['ap_macro']:.4f} | F1 {resumo_px['f1_macro']:.4f}")

    df = dados['df']
    metricas = {
        'modelo': titulo, 'nome': nome, 'semente': semente, 'protocolo': PROTOCOLO,
        **resumo_teste,
        'por_classe':        por_classe_teste,
        'code_test':         code_test,
        'ptbxl':             ptbxl,
        'limiares':          limiares,
        'melhor_ap_val':     melhor_ap,
        'epocas_treinadas':  len(historico['perda_treino']) if historico else None,   # None = ensemble
        'melhor_epoca':      int(np.argmax(historico['ap_val'])) + 1 if historico else None,
        'tempo_medio_epoca_s': float(np.mean(historico['tempo_epoca'])) if historico else None,
        'tempo_inferencia_ms_por_exame': tempo_inferencia_ms,
        'parametros':        total_parametros,
        'taxa_aprendizado':  taxa_aprendizado,
        'n_treino': len(dados['treino']), 'n_validacao': len(dados['validacao']),
        'n_teste':  len(dados['teste']),
        'assinatura_dados':  dados['assinatura'],
        'positivos': {p: {s: int(rotulos[dados[p], j].sum()) for j, s in enumerate(SAIDAS)}
                      for p in ['treino', 'validacao', 'teste']},
        'pacientes': int(df['patient_id'].nunique()),
        'configuracoes':     configuracoes,
    }
    with open(os.path.join(pasta_saida, 'metricas.json'), 'w', encoding='utf-8') as f:
        json.dump(metricas, f, ensure_ascii=False, indent=2, default=float)

    # ----- Etapa 10: matrizes de confusão 2×2 por classe (teste interno) -----
    fig, eixos = plt.subplots(2, 4, figsize=(16, 8))
    for ax, (sigla, c) in zip(eixos.flat, por_classe_teste.items()):
        m = np.array(c['equilibrado']['matriz'])
        porcentagem = m / m.sum(axis=1, keepdims=True)
        anotacoes = np.array([[f"{m[i, j]:,}\n{porcentagem[i, j]:.1%}" for j in range(2)] for i in range(2)])
        sns.heatmap(porcentagem, annot=anotacoes, fmt='', cmap='Blues', vmin=0, vmax=1, cbar=False, ax=ax,
                    xticklabels=['Não', 'Sim'], yticklabels=['Não', 'Sim'], linewidths=0.5, linecolor='gray')
        ax.set_title(f"{c['nome']}\nSens {c['equilibrado']['sensibilidade']:.1%} · "
                     f"Espec {c['equilibrado']['especificidade']:.1%}", fontsize=10, fontweight='bold')
        ax.set_xlabel('Previsto'); ax.set_ylabel('Real')
    eixos.flat[-1].axis('off')
    plt.suptitle(f'Matrizes de confusão por classe — {titulo} (teste, prevalência real)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(pasta_saida, 'matrizes_confusao.png'), dpi=200)
    plt.close(fig)

    # ----- Etapa 11: curvas de aprendizado (o ensemble não tem treino próprio) -----
    if historico:
        salvar_curvas(historico, metricas['melhor_epoca'], titulo, pasta_saida)

    # ----- Etapa 11b: F1 no CODE-test contra especialistas -----
    if code_test:
        comparacao = code_test['comparacao_f1']
        fig, ax = plt.subplots(figsize=(13, 5))
        largura = 0.8 / len(comparacao)
        for k, (leitor, f1) in enumerate(comparacao.items()):
            ax.bar(np.arange(len(DOENCAS)) + k * largura, [f1[d] for d in DOENCAS], largura,
                   label=f"{leitor} (macro {f1['macro']:.3f})")
        ax.set_xticks(np.arange(len(DOENCAS)) + 0.4 - largura / 2)
        ax.set_xticklabels([NOMES_COMPLETOS[d] for d in DOENCAS], rotation=15)
        ax.set_ylim(0, 1); ax.set_ylabel('F1-score'); ax.grid(axis='y', alpha=0.3)
        ax.set_title(f'CODE-test (827 exames, laudo de cardiologistas) — {titulo}', fontweight='bold')
        ax.legend(fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False)
        plt.tight_layout()
        plt.savefig(os.path.join(pasta_saida, 'comparacao_code_test.png'), dpi=200)
        plt.close(fig)

    # ----- Etapa 12: modelo final com tudo que a inferência precisa -----
    torch.save({
        'protocolo':      PROTOCOLO,
        'nome':           nome,
        'arquitetura':    titulo,
        'pesos':          modelo.state_dict(),
        'configuracoes':  configuracoes,
        'saidas':         SAIDAS,
        'nomes_saidas':   NOMES_SAIDAS,
        'limiares':       limiares,
        'pre_processamento': {'frequencia_hz': FREQUENCIA_HZ, 'num_amostras': NUM_AMOSTRAS,
                              'normalizacao': 'zscore_por_derivacao'},
        'metricas_teste': {k: metricas[k] for k in ['auc_macro', 'ap_macro', 'f1_macro']},
        'metricas_code_test': code_test['resumo'] if code_test else None,
        'metricas_ptbxl':     ptbxl['resumo'] if ptbxl else None,
    }, os.path.join(pasta_saida, 'modelo_final.pt'))

    print(f"\nArquivos gerados em: {pasta_saida}")
    return metricas
