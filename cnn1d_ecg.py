# =============================================================================
# ECG CNN 1D (ResNet) — Classificação de Arritmias Cardíacas
# Dataset: CODE-15% (Rede de Telessaúde de Minas Gerais — TNMG)
# Modelo: ResNet 1D de Ribeiro et al. (2020), Nature Communications 11:1760
# Tarefa: Multiclass — Normal + 6 arritmias (7 classes)
#
# CONTEXTO DO TCC
#   O TCC desenvolve uma IA que lê ECGs de 12 derivações e classifica o
#   traçado em 7 classes. O modelo principal é um Transformer (transformer_ecg.py).
#   Este script treina a CNN 1D usada como BASELINE: é a mesma arquitetura
#   publicada pelos autores do dataset CODE, então serve de referência para
#   responder "o Transformer é melhor que o estado da arte convolucional?".
#
#   Para a comparação ser justa, TUDO fora da Etapa 5 é idêntico ao
#   transformer_ecg.py: mesmos exames, mesmo balanceamento, mesma divisão
#   treino/val/teste (seed 42), mesmo augmentation, mesma loss, mesmo
#   agendador e mesmo early stopping. A única diferença de treino é a taxa
#   de aprendizado (1e-3, a usada por Ribeiro et al.).
# =============================================================================

import os
import math
import time
from glob import glob

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# =============================================================================
# CONFIGURAÇÕES GERAIS
# =============================================================================

PASTA_DADOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dados')
ARQUIVO_CSV = os.path.join(PASTA_DADOS, 'exams.csv')

# Pasta própria para não sobrescrever checkpoint/resultados do Transformer
PASTA_SAIDA        = os.path.join(PASTA_DADOS, 'resultados_cnn1d')
ARQUIVO_CHECKPOINT = os.path.join(PASTA_SAIDA, 'checkpoint_cnn1d.pt')
ARQUIVO_MELHOR     = os.path.join(PASTA_SAIDA, 'melhor_modelo_cnn1d.pt')
os.makedirs(PASTA_SAIDA, exist_ok=True)

DOENCAS = ['1dAVb', 'RBBB', 'LBBB', 'SB', 'ST', 'AF']

NOMES_COMPLETOS = {
    'Normal': 'Normal',
    '1dAVb':  'Bloqueio AV 1º Grau',
    'RBBB':   'Bloqueio Ramo Direito',
    'LBBB':   'Bloqueio Ramo Esquerdo',
    'SB':     'Bradicardia Sinusal',
    'ST':     'Taquicardia Sinusal',
    'AF':     'Fibrilação Atrial',
}

CLASSES_ORDENADAS = ['Normal'] + DOENCAS
NOMES_ORDENADOS   = [NOMES_COMPLETOS[c] for c in CLASSES_ORDENADAS]
NUM_CLASSES       = len(CLASSES_ORDENADAS)   # 7

# Hiperparâmetros do modelo — ResNet 1D (Ribeiro et al., 2020)
NUM_DERIVACOES = 12
TAMANHO_KERNEL = 17                          # artigo usa 16; ímpar permite padding simétrico
FILTROS        = [64, 128, 196, 256, 320]    # camada de entrada + 4 blocos residuais
COMPRIMENTOS   = [4096, 1024, 256, 64, 16]   # amostras após cada etapa (reduz 4× por bloco)
DROPOUT        = 0.2                         # artigo: keep_prob 0.8 → dropout 0.2

# Hiperparâmetros de treino (iguais ao Transformer, exceto a LR)
AMOSTRAS_POR_CLASSE = 2000
TAMANHO_BATCH       = 32
EPOCAS              = 200    # alto — early stopping controla quando parar
PACIENCIA           = 20
MIN_EPOCAS          = 30
TAXA_APRENDIZADO    = 1e-3   # LR usada por Ribeiro et al.; CNN com BatchNorm tolera LR maior
SEMENTE             = 42

DISPOSITIVO = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Dispositivo: {DISPOSITIVO}")

torch.manual_seed(SEMENTE)
np.random.seed(SEMENTE)
torch.backends.cudnn.benchmark = True        # entrada de tamanho fixo → cuDNN escolhe a conv mais rápida

# =============================================================================
# ETAPA 1 — CARREGAMENTO E PREPARAÇÃO DOS DADOS
# =============================================================================

def carregar_metadados(arquivo_csv, pasta_dados):
    df = pd.read_csv(arquivo_csv)
    arquivos_disponiveis = [
        os.path.basename(p) for p in glob(os.path.join(pasta_dados, '*.hdf5'))
    ]
    print(f"Arquivos HDF5 encontrados: {arquivos_disponiveis}")
    df = df[df['trace_file'].isin(arquivos_disponiveis)].reset_index(drop=True)
    print(f"Total de exames disponíveis: {len(df):,}")
    return df


def atribuir_classe(linha):
    doencas_presentes = [d for d in DOENCAS if linha[d]]
    if len(doencas_presentes) == 0:
        return 0
    if len(doencas_presentes) == 1:
        return DOENCAS.index(doencas_presentes[0]) + 1
    return -1


def balancear_dataset(df, amostras_por_classe):
    disponiveis = [len(df[df['classe'] == i]) for i in range(NUM_CLASSES)]
    limite      = min(min(disponiveis), amostras_por_classe)
    print(f"Menor classe: {min(disponiveis)} exames → usando {limite} por classe")

    grupos = []
    for indice_classe in range(NUM_CLASSES):
        subset = df[df['classe'] == indice_classe]
        n      = min(len(subset), limite)
        grupos.append(subset.sample(n, random_state=SEMENTE))
        print(f"  {NOMES_ORDENADOS[indice_classe]:30s}: {n:,} exames")

    return (pd.concat(grupos)
              .sample(frac=1, random_state=SEMENTE)
              .reset_index(drop=True))


df_exames            = carregar_metadados(ARQUIVO_CSV, PASTA_DADOS)
df_exames['classe']  = df_exames.apply(atribuir_classe, axis=1)
df_exames            = df_exames[df_exames['classe'] >= 0].reset_index(drop=True)

print("\nDistribuição antes do balanceamento:")
for i, nome in enumerate(NOMES_ORDENADOS):
    print(f"  {nome:30s}: {(df_exames['classe'] == i).sum():,}")

print("\nBalanceando dataset:")
df_balanceado = balancear_dataset(df_exames, AMOSTRAS_POR_CLASSE)
print(f"\nTotal após balanceamento: {len(df_balanceado):,} exames")

# =============================================================================
# ETAPA 2 — CARREGAMENTO DOS SINAIS EM LOTE
# =============================================================================

def normalizar_sinal(sinal):
    return StandardScaler().fit_transform(sinal).astype(np.float32)


def carregar_sinais_em_lote(df_balanceado, pasta_dados):
    sinais_dict = {}
    exames_por_arquivo = (df_balanceado
                          .groupby('trace_file')['exam_id']
                          .apply(list)
                          .to_dict())

    for nome_arquivo, lista_ids in exames_por_arquivo.items():
        caminho = os.path.join(pasta_dados, nome_arquivo)
        print(f"  Lendo {nome_arquivo} ({len(lista_ids):,} exames)...")
        with h5py.File(caminho, 'r') as arquivo:
            ids_hdf5 = arquivo['exam_id'][:]
            for exam_id in lista_ids:
                posicao = np.where(ids_hdf5 == exam_id)[0][0]
                sinal   = arquivo['tracings'][posicao].astype(np.float32)
                sinais_dict[exam_id] = normalizar_sinal(sinal)

    sinais  = np.array([sinais_dict[eid] for eid in df_balanceado['exam_id']])
    rotulos = np.array(df_balanceado['classe'].tolist())
    return sinais, rotulos


print("\nCarregando sinais do ECG...")
sinais, rotulos = carregar_sinais_em_lote(df_balanceado, PASTA_DADOS)
print(f"Sinais carregados: {sinais.shape} | Rótulos: {rotulos.shape}")

# =============================================================================
# ETAPA 3 — DIVISÃO TREINO / VALIDAÇÃO / TESTE
# =============================================================================

sinais_treino, sinais_temp, rotulos_treino, rotulos_temp = train_test_split(
    sinais, rotulos, test_size=0.3, stratify=rotulos, random_state=SEMENTE)

sinais_val, sinais_teste, rotulos_val, rotulos_teste = train_test_split(
    sinais_temp, rotulos_temp, test_size=0.5, stratify=rotulos_temp, random_state=SEMENTE)

print(f"\nTreino: {len(sinais_treino):,} | Validação: {len(sinais_val):,} | Teste: {len(sinais_teste):,}")

# =============================================================================
# ETAPA 4 — DATASET E DATALOADER
# =============================================================================

class ConjuntoDadosECG(Dataset):
    def __init__(self, sinais, rotulos, aumentar=False):
        self.sinais   = torch.tensor(sinais)
        self.rotulos  = torch.tensor(rotulos, dtype=torch.long)
        self.aumentar = aumentar

    def __len__(self):
        return len(self.sinais)

    def __getitem__(self, idx):
        sinal  = self.sinais[idx].clone()
        rotulo = self.rotulos[idx]

        if self.aumentar:
            sinal += torch.randn_like(sinal) * 0.02
            sinal *= 0.9 + torch.rand(1).item() * 0.2
            if torch.rand(1).item() < 0.5:
                sinal = -sinal

        return sinal, rotulo


def criar_dataloader(sinais, rotulos, aumentar=False, embaralhar=False):
    dataset = ConjuntoDadosECG(sinais, rotulos, aumentar=aumentar)
    return DataLoader(dataset, batch_size=TAMANHO_BATCH,
                      shuffle=embaralhar, pin_memory=torch.cuda.is_available())


carregador_treino    = criar_dataloader(sinais_treino, rotulos_treino, aumentar=True,  embaralhar=True)
carregador_validacao = criar_dataloader(sinais_val,    rotulos_val,    aumentar=False, embaralhar=False)
carregador_teste     = criar_dataloader(sinais_teste,  rotulos_teste,  aumentar=False, embaralhar=False)

PASSOS_POR_EPOCA = len(carregador_treino)
PASSOS_WARMUP    = max(50, min(200, PASSOS_POR_EPOCA // 2))
TOTAL_PASSOS     = EPOCAS * PASSOS_POR_EPOCA

print(f"\nPassos por época:  {PASSOS_POR_EPOCA}")
print(f"Warmup configurado: {PASSOS_WARMUP} passos ({PASSOS_WARMUP/PASSOS_POR_EPOCA:.1f} épocas)")
print(f"Total de passos:   {TOTAL_PASSOS}")

# =============================================================================
# ETAPA 5 — ARQUITETURA DO MODELO (ResNet 1D — Ribeiro et al., 2020)
# =============================================================================

def calcular_padding(kernel, reducao):
    """Padding que faz a conv reduzir o comprimento exatamente pelo fator `reducao`."""
    return max(0, (kernel - reducao + 1) // 2)


class BlocoResidual1D(nn.Module):
    """
    Bloco residual com pré-ativação (He et al., 2016), como em Ribeiro et al.:

      caminho principal: Conv → BN → ReLU → Dropout → Conv (stride = reducao)
      atalho:            MaxPool(reducao) → Conv 1×1 (só se o nº de filtros muda)
      saída:             soma → BN → ReLU → Dropout

    O bloco recebe e devolve dois tensores:
      x → sinal já ativado (segue pelo caminho principal)
      y → soma "crua", antes do BN/ReLU (segue pelo atalho)
    """
    def __init__(self, filtros_in, filtros_out, reducao, kernel, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(filtros_in, filtros_out, kernel,
                               padding=calcular_padding(kernel, 1), bias=False)
        self.bn1   = nn.BatchNorm1d(filtros_out)
        self.conv2 = nn.Conv1d(filtros_out, filtros_out, kernel, stride=reducao,
                               padding=calcular_padding(kernel, reducao), bias=False)
        self.bn2   = nn.BatchNorm1d(filtros_out)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        camadas_atalho = []
        if reducao > 1:
            camadas_atalho.append(nn.MaxPool1d(reducao, stride=reducao))
        if filtros_in != filtros_out:
            camadas_atalho.append(nn.Conv1d(filtros_in, filtros_out, 1, bias=False))
        self.atalho = nn.Sequential(*camadas_atalho)   # vazio = identidade

    def forward(self, x, y):
        y = self.atalho(y)
        x = self.dropout(self.relu(self.bn1(self.conv1(x))))
        x = self.conv2(x) + y
        y = x
        x = self.dropout(self.relu(self.bn2(x)))
        return x, y


class ResNet1DECG(nn.Module):
    """
    CNN 1D para classificação multiclass de ECG (7 classes).
    Entrada: (batch, 4096, 12) — mesmo formato do Transformer.

    Fluxo:
      Conv(12→64)       → 4096 amostras
      Bloco 1 (64→128)  → 1024
      Bloco 2 (128→196) →  256
      Bloco 3 (196→256) →   64
      Bloco 4 (256→320) →   16
      Flatten (320×16 = 5120) → Linear → 7 logits

    Única mudança em relação ao artigo: a saída é softmax de 7 classes
    exclusivas (aqui) em vez de 6 sigmoides independentes (multi-label).
    """
    def __init__(self):
        super().__init__()
        self.conv_entrada = nn.Conv1d(NUM_DERIVACOES, FILTROS[0], TAMANHO_KERNEL,
                                      padding=calcular_padding(TAMANHO_KERNEL, 1), bias=False)
        self.bn_entrada   = nn.BatchNorm1d(FILTROS[0])
        self.relu         = nn.ReLU()

        self.blocos = nn.ModuleList()
        for i in range(1, len(FILTROS)):
            reducao = COMPRIMENTOS[i - 1] // COMPRIMENTOS[i]
            self.blocos.append(
                BlocoResidual1D(FILTROS[i - 1], FILTROS[i], reducao, TAMANHO_KERNEL, DROPOUT))

        self.classificador = nn.Linear(FILTROS[-1] * COMPRIMENTOS[-1], NUM_CLASSES)

    def forward(self, sinal):
        x = sinal.permute(0, 2, 1)                       # (B, 4096, 12) → (B, 12, 4096)
        x = self.relu(self.bn_entrada(self.conv_entrada(x)))
        y = x
        for bloco in self.blocos:
            x, y = bloco(x, y)
        x = x.flatten(1)                                 # (B, 320, 16) → (B, 5120)
        return self.classificador(x)


modelo = ResNet1DECG().to(DISPOSITIVO)
total_parametros = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
print(f"\nModelo criado | Parâmetros treináveis: {total_parametros:,}")

# =============================================================================
# ETAPA 6 — CONFIGURAÇÃO DO TREINO
# =============================================================================

funcao_perda = nn.CrossEntropyLoss(label_smoothing=0.1)
otimizador   = torch.optim.AdamW(
    modelo.parameters(), lr=TAXA_APRENDIZADO, weight_decay=1e-3)

def calcular_taxa_aprendizado(passo_atual):
    """Warmup linear + decaimento cosseno proporcional ao treino real."""
    if passo_atual < PASSOS_WARMUP:
        return passo_atual / max(1, PASSOS_WARMUP)
    progresso = (passo_atual - PASSOS_WARMUP) / max(1, TOTAL_PASSOS - PASSOS_WARMUP)
    return max(0.05, 0.5 * (1 + math.cos(math.pi * progresso)))

agendador = torch.optim.lr_scheduler.LambdaLR(otimizador, calcular_taxa_aprendizado)

# =============================================================================
# ETAPA 7 — CHECKPOINT (retoma de onde parou)
# =============================================================================

def salvar_checkpoint(epoca, modelo, otimizador, agendador,
                      melhor_acuracia, contador_paciencia, historico, caminho):
    """Salva o estado completo do treino (inclusive a paciência do early stopping)."""
    torch.save({
        'epoca':              epoca,
        'modelo':             modelo.state_dict(),
        'otimizador':         otimizador.state_dict(),
        'agendador':          agendador.state_dict(),
        'melhor_acuracia':    melhor_acuracia,
        'contador_paciencia': contador_paciencia,
        'historico':          historico,
    }, caminho)


def carregar_checkpoint(caminho, modelo, otimizador, agendador):
    """Retorna (época de onde retomar, melhor acc, paciência, histórico)."""
    historico_vazio = {'perda_treino': [], 'perda_val': [],
                       'acc_treino':   [], 'acc_val':   [], 'tempo_epoca': []}

    if not os.path.exists(caminho):
        print("Nenhum checkpoint encontrado — iniciando do zero.")
        return 1, 0, 0, historico_vazio

    print(f"Checkpoint encontrado! Carregando {caminho}...")
    estado = torch.load(caminho, map_location=DISPOSITIVO)

    modelo.load_state_dict(estado['modelo'])
    otimizador.load_state_dict(estado['otimizador'])
    agendador.load_state_dict(estado['agendador'])

    historico = estado['historico']
    historico.setdefault('tempo_epoca', [])

    epoca_retomar      = estado['epoca'] + 1
    melhor_acuracia    = estado['melhor_acuracia']
    contador_paciencia = estado.get('contador_paciencia', 0)

    print(f"Retomando da época {epoca_retomar} | Melhor acc até agora: {melhor_acuracia:.4f} "
          f"| Paciência: {contador_paciencia}/{PACIENCIA}")
    return epoca_retomar, melhor_acuracia, contador_paciencia, historico

# =============================================================================
# ETAPA 8 — LOOP DE TREINO
# =============================================================================

def executar_epoca(carregador, treino=True):
    modelo.train() if treino else modelo.eval()

    perda_total, acertos, total = 0, 0, 0
    todas_predicoes, todos_rotulos = [], []

    contexto = torch.enable_grad() if treino else torch.no_grad()
    with contexto:
        for sinais_batch, rotulos_batch in carregador:
            sinais_batch  = sinais_batch.to(DISPOSITIVO)
            rotulos_batch = rotulos_batch.to(DISPOSITIVO)

            logits = modelo(sinais_batch)
            perda  = funcao_perda(logits, rotulos_batch)

            if treino:
                otimizador.zero_grad()
                perda.backward()
                nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
                otimizador.step()
                agendador.step()

            predicoes   = torch.argmax(logits, dim=1)
            acertos     += (predicoes == rotulos_batch).sum().item()
            total       += rotulos_batch.size(0)
            perda_total += perda.item()

            todas_predicoes.extend(predicoes.cpu().numpy())
            todos_rotulos.extend(rotulos_batch.cpu().numpy())

    acuracia = acertos / total
    return perda_total / len(carregador), acuracia, todas_predicoes, todos_rotulos


epoca_inicial, melhor_acuracia, contador_paciencia, historico = carregar_checkpoint(
    ARQUIVO_CHECKPOINT, modelo, otimizador, agendador)

melhores_pesos = None

print(f"\n=== INÍCIO DO TREINO (época {epoca_inicial} até {EPOCAS}) ===")
print(f"Early stopping: paciência de {PACIENCIA} épocas | Mínimo de {MIN_EPOCAS} épocas\n")

for epoca in range(epoca_inicial, EPOCAS + 1):
    inicio = time.time()
    perda_tr,  acc_tr,  _, _ = executar_epoca(carregador_treino,    treino=True)
    perda_val, acc_val, _, _ = executar_epoca(carregador_validacao, treino=False)
    duracao = time.time() - inicio

    historico['perda_treino'].append(perda_tr)
    historico['perda_val'].append(perda_val)
    historico['acc_treino'].append(acc_tr)
    historico['acc_val'].append(acc_val)
    historico['tempo_epoca'].append(duracao)

    melhorou = acc_val > melhor_acuracia
    if melhorou:
        melhor_acuracia    = acc_val
        melhores_pesos     = {k: v.clone() for k, v in modelo.state_dict().items()}
        contador_paciencia = 0
        torch.save(melhores_pesos, ARQUIVO_MELHOR)
    else:
        contador_paciencia += 1

    lr_atual = otimizador.param_groups[0]['lr']
    print(f"Época {epoca:03d} | "
          f"Perda tr: {perda_tr:.4f} Acc tr: {acc_tr:.4f} | "
          f"Perda val: {perda_val:.4f} Acc val: {acc_val:.4f} | "
          f"LR: {lr_atual:.2e} | {duracao:.0f}s"
          + (" ✓" if melhorou else f" (paciência {contador_paciencia}/{PACIENCIA})"))

    if epoca % 5 == 0:
        salvar_checkpoint(epoca, modelo, otimizador, agendador,
                          melhor_acuracia, contador_paciencia, historico, ARQUIVO_CHECKPOINT)
        print(f"  → Checkpoint salvo na época {epoca}")

    if epoca >= MIN_EPOCAS and contador_paciencia >= PACIENCIA:
        print(f"\n⏹ Early stopping na época {epoca}.")
        print(f"   Modelo não melhorou por {PACIENCIA} épocas consecutivas.")
        print(f"   Melhor acurácia de validação: {melhor_acuracia:.4f}")
        salvar_checkpoint(epoca, modelo, otimizador, agendador,
                          melhor_acuracia, contador_paciencia, historico, ARQUIVO_CHECKPOINT)
        break

# =============================================================================
# ETAPA 9 — AVALIAÇÃO FINAL NO TESTE
# =============================================================================

print("\n=== AVALIAÇÃO FINAL NO CONJUNTO DE TESTE ===")
# Se o treino foi retomado e não melhorou nesta execução, os melhores pesos
# só existem em disco — carrega de lá em vez de quebrar com None.
if melhores_pesos is None:
    melhores_pesos = torch.load(ARQUIVO_MELHOR, map_location=DISPOSITIVO)
modelo.load_state_dict(melhores_pesos)
_, acuracia_teste, predicoes_teste, rotulos_teste_lista = executar_epoca(
    carregador_teste, treino=False)

predicoes_teste     = np.array(predicoes_teste)
rotulos_teste_lista = np.array(rotulos_teste_lista)

print(f"Acurácia geral: {acuracia_teste:.4f}\n")
print(classification_report(
    rotulos_teste_lista, predicoes_teste,
    labels=list(range(NUM_CLASSES)), target_names=NOMES_ORDENADOS,
    zero_division=0, digits=4))

relatorio = classification_report(
    rotulos_teste_lista, predicoes_teste,
    labels=list(range(NUM_CLASSES)), target_names=NOMES_ORDENADOS,
    zero_division=0, output_dict=True)

# =============================================================================
# ETAPA 10 — MATRIZ DE CONFUSÃO 7x7
# =============================================================================

matriz = confusion_matrix(rotulos_teste_lista, predicoes_teste,
                          labels=list(range(NUM_CLASSES)))

fig, ax = plt.subplots(figsize=(13, 11))
sns.heatmap(
    matriz,
    annot=True, fmt='d', cmap='Blues', ax=ax,
    xticklabels=NOMES_ORDENADOS,
    yticklabels=NOMES_ORDENADOS,
    linewidths=0.5, linecolor='gray'
)
ax.set_xlabel('Diagnóstico previsto pelo modelo', fontsize=12, labelpad=10)
ax.set_ylabel('Diagnóstico real (gabarito)', fontsize=12, labelpad=10)
ax.set_title('Matriz de Confusão 7×7 — CNN 1D (ResNet)\nNormal + 6 Arritmias Cardíacas',
             fontsize=14, fontweight='bold', pad=15)
plt.xticks(rotation=35, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(PASTA_SAIDA, 'matriz_confusao_7x7_cnn1d.png'), dpi=150)
plt.show()

# =============================================================================
# ETAPA 11 — CURVAS DE APRENDIZADO
# =============================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
epocas_executadas = range(1, len(historico['perda_treino']) + 1)

ax1.plot(epocas_executadas, historico['perda_treino'], label='Treino',    color='#1565C0')
ax1.plot(epocas_executadas, historico['perda_val'],    label='Validação', color='#C62828')
ax1.set_title('Curva de Perda (Loss)', fontweight='bold')
ax1.set_xlabel('Época')
ax1.set_ylabel('Perda')
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot(epocas_executadas, historico['acc_treino'], label='Treino',    color='#1565C0')
ax2.plot(epocas_executadas, historico['acc_val'],    label='Validação', color='#C62828')
ax2.set_title('Curva de Acurácia', fontweight='bold')
ax2.set_xlabel('Época')
ax2.set_ylabel('Acurácia')
ax2.legend()
ax2.grid(alpha=0.3)

plt.suptitle('Histórico de Treinamento — CNN 1D (ResNet)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PASTA_SAIDA, 'curvas_aprendizado_cnn1d.png'), dpi=150)
plt.show()

# =============================================================================
# ETAPA 12 — SALVAR MODELO FINAL COM CONFIGURAÇÕES E MÉTRICAS
# =============================================================================

tempo_medio_epoca = float(np.mean(historico['tempo_epoca'])) if historico['tempo_epoca'] else None

torch.save({
    'pesos':          melhores_pesos,
    'arquitetura':    'ResNet1D (Ribeiro et al., 2020)',
    'configuracoes': {
        'num_classes':    NUM_CLASSES,
        'num_derivacoes': NUM_DERIVACOES,
        'tamanho_kernel': TAMANHO_KERNEL,
        'filtros':        FILTROS,
        'comprimentos':   COMPRIMENTOS,
        'dropout':        DROPOUT,
        'taxa_aprendizado': TAXA_APRENDIZADO,
    },
    'classes':           CLASSES_ORDENADAS,
    'nomes_classes':     NOMES_ORDENADOS,
    'melhor_acc_val':    melhor_acuracia,
    'acc_teste':         acuracia_teste,
    'relatorio_teste':   relatorio,
    'parametros':        total_parametros,
    'tempo_medio_epoca': tempo_medio_epoca,
}, os.path.join(PASTA_SAIDA, 'ecg_cnn1d_final.pt'))

print(f"\nParâmetros treináveis: {total_parametros:,}")
if tempo_medio_epoca is not None:
    print(f"Tempo médio por época: {tempo_medio_epoca:.1f}s")
print(f"\nArquivos gerados em: {PASTA_SAIDA}")
print("  - matriz_confusao_7x7_cnn1d.png")
print("  - curvas_aprendizado_cnn1d.png")
print("  - melhor_modelo_cnn1d.pt   ← melhor época salva automaticamente")
print("  - checkpoint_cnn1d.pt      ← estado completo para retomar treino")
print("  - ecg_cnn1d_final.pt       ← modelo final + configurações + métricas de teste")
