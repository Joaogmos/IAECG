# =============================================================================
# ECG TRANSFORMER — Classificação de Arritmias Cardíacas
# Dataset: CODE (Telehealth Network of Minas Gerais)
# Modelo: Vision Transformer adaptado para sinais 1D (ECG)
# Tarefa: Multiclass — Normal + 6 arritmias (7 classes)
# =============================================================================

import os
import math
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

PASTA_DADOS  = r'C:\Users\Thiago Ruiz\Documents\Dev'
ARQUIVO_CSV  = f'{PASTA_DADOS}\\exams.csv'
PASTA_SAIDA  = f'{PASTA_DADOS}\\resultados'
ARQUIVO_CHECKPOINT = f'{PASTA_SAIDA}\\checkpoint.pt'
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

# Hiperparâmetros do modelo
TAMANHO_PATCH    = 64
DIMENSAO_MODELO  = 128
NUM_CABECAS      = 4
NUM_CAMADAS      = 4
DROPOUT          = 0.1

# Hiperparâmetros de treino
AMOSTRAS_POR_CLASSE = 2000
TAMANHO_BATCH       = 32
EPOCAS              = 200    # alto — early stopping controla quando parar
PACIENCIA           = 20     # aumentado — dá mais chance ao modelo de sair de platôs
MIN_EPOCAS          = 30     # modelo treina no mínimo 30 épocas antes do early stopping
TAXA_APRENDIZADO    = 3e-4
SEMENTE             = 42

DISPOSITIVO = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Dispositivo: {DISPOSITIVO}")

torch.manual_seed(SEMENTE)
np.random.seed(SEMENTE)

# =============================================================================
# ETAPA 1 — CARREGAMENTO E PREPARAÇÃO DOS DADOS
# =============================================================================

def carregar_metadados(arquivo_csv, pasta_dados):
    df = pd.read_csv(arquivo_csv)
    arquivos_disponiveis = [
        os.path.basename(p) for p in glob(f'{pasta_dados}\\*.hdf5')
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
        caminho = f'{pasta_dados}\\{nome_arquivo}'
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
                      shuffle=embaralhar, pin_memory=True)


carregador_treino    = criar_dataloader(sinais_treino, rotulos_treino, aumentar=True,  embaralhar=True)
carregador_validacao = criar_dataloader(sinais_val,    rotulos_val,    aumentar=False, embaralhar=False)
carregador_teste     = criar_dataloader(sinais_teste,  rotulos_teste,  aumentar=False, embaralhar=False)

# Warmup proporcional ao tamanho real dos dados — professor pediu
# Aqui calculamos DEPOIS de criar o dataloader pois dependemos do tamanho
PASSOS_POR_EPOCA = len(carregador_treino)
PASSOS_WARMUP    = max(50, min(200, PASSOS_POR_EPOCA // 2))
TOTAL_PASSOS     = EPOCAS * PASSOS_POR_EPOCA

print(f"\nPassos por época:  {PASSOS_POR_EPOCA}")
print(f"Warmup configurado: {PASSOS_WARMUP} passos ({PASSOS_WARMUP/PASSOS_POR_EPOCA:.1f} épocas)")
print(f"Total de passos:   {TOTAL_PASSOS}")

# =============================================================================
# ETAPA 5 — ARQUITETURA DO MODELO
# =============================================================================

class EmbeddingPatch(nn.Module):
    """
    Divide o sinal ECG em patches e projeta para o espaço do modelo.
    Entrada:  (batch, 4096, 12)
    Saída:    (batch, 64 patches, d_model)
    """
    def __init__(self):
        super().__init__()
        self.projecao = nn.Conv1d(
            in_channels=12,
            out_channels=DIMENSAO_MODELO,
            kernel_size=TAMANHO_PATCH,
            stride=TAMANHO_PATCH
        )

    def forward(self, sinal):
        sinal = sinal.permute(0, 2, 1)
        sinal = self.projecao(sinal)
        return sinal.permute(0, 2, 1)


class TransformerECG(nn.Module):
    """
    Transformer para classificação multiclass de ECG (7 classes).
    Fluxo: Patch Embedding → CLS Token → Positional Emb →
           Transformer Encoder × 4 → CLS → Cabeça Linear → 7 logits
    """
    def __init__(self):
        super().__init__()
        num_patches = 4096 // TAMANHO_PATCH

        self.embedding_patch      = EmbeddingPatch()
        self.token_cls            = nn.Parameter(torch.zeros(1, 1, DIMENSAO_MODELO))
        self.embedding_posicional = nn.Parameter(
            torch.zeros(1, num_patches + 1, DIMENSAO_MODELO))
        self.dropout_entrada      = nn.Dropout(DROPOUT)

        camada_encoder = nn.TransformerEncoderLayer(
            d_model=DIMENSAO_MODELO,
            nhead=NUM_CABECAS,
            dim_feedforward=DIMENSAO_MODELO * 4,
            dropout=DROPOUT,
            batch_first=True,
            norm_first=True
        )
        self.transformer  = nn.TransformerEncoder(camada_encoder, num_layers=NUM_CAMADAS)
        self.normalizacao = nn.LayerNorm(DIMENSAO_MODELO)

        self.cabeca_classificacao = nn.Sequential(
            nn.Linear(DIMENSAO_MODELO, 64),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, NUM_CLASSES)
        )

        self._inicializar_pesos()

    def _inicializar_pesos(self):
        nn.init.trunc_normal_(self.token_cls, std=0.02)
        nn.init.trunc_normal_(self.embedding_posicional, std=0.02)

    def forward(self, sinal):
        tamanho_batch = sinal.size(0)
        tokens        = self.embedding_patch(sinal)
        token_cls     = self.token_cls.expand(tamanho_batch, -1, -1)
        tokens        = torch.cat([token_cls, tokens], dim=1)
        tokens        = self.dropout_entrada(tokens + self.embedding_posicional)
        tokens        = self.transformer(tokens)
        representacao = self.normalizacao(tokens[:, 0])
        return self.cabeca_classificacao(representacao)


modelo = TransformerECG().to(DISPOSITIVO)
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
                      melhor_acuracia, historico, caminho):
    """
    Salva o estado completo do treino.
    Se o processo for interrompido, retoma desta época.
    """
    torch.save({
        'epoca':           epoca,
        'modelo':          modelo.state_dict(),
        'otimizador':      otimizador.state_dict(),
        'agendador':       agendador.state_dict(),
        'melhor_acuracia': melhor_acuracia,
        'historico':       historico,
    }, caminho)


def carregar_checkpoint(caminho, modelo, otimizador, agendador):
    """
    Carrega o estado salvo e retorna a época de onde retomar.
    Retorna época 1 e acurácia 0 se não houver checkpoint.
    """
    if not os.path.exists(caminho):
        print("Nenhum checkpoint encontrado — iniciando do zero.")
        return 1, 0, {'perda_treino': [], 'perda_val': [],
                      'acc_treino':   [], 'acc_val':   []}

    print(f"Checkpoint encontrado! Carregando {caminho}...")
    estado = torch.load(caminho, map_location=DISPOSITIVO)

    modelo.load_state_dict(estado['modelo'])
    otimizador.load_state_dict(estado['otimizador'])
    agendador.load_state_dict(estado['agendador'])

    epoca_retomar    = estado['epoca'] + 1
    melhor_acuracia  = estado['melhor_acuracia']
    historico        = estado['historico']

    print(f"Retomando da época {epoca_retomar} | Melhor acc até agora: {melhor_acuracia:.4f}")
    return epoca_retomar, melhor_acuracia, historico

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


# Carrega checkpoint se existir — caso contrário começa do zero
epoca_inicial, melhor_acuracia, historico = carregar_checkpoint(
    ARQUIVO_CHECKPOINT, modelo, otimizador, agendador)

melhores_pesos     = None
contador_paciencia = 0

print(f"\n=== INÍCIO DO TREINO (época {epoca_inicial} até {EPOCAS}) ===")
print(f"Early stopping: paciência de {PACIENCIA} épocas | Mínimo de {MIN_EPOCAS} épocas\n")

for epoca in range(epoca_inicial, EPOCAS + 1):
    perda_tr,  acc_tr,  _, _ = executar_epoca(carregador_treino,    treino=True)
    perda_val, acc_val, _, _ = executar_epoca(carregador_validacao, treino=False)

    historico['perda_treino'].append(perda_tr)
    historico['perda_val'].append(perda_val)
    historico['acc_treino'].append(acc_tr)
    historico['acc_val'].append(acc_val)

    melhorou = acc_val > melhor_acuracia
    if melhorou:
        melhor_acuracia    = acc_val
        melhores_pesos     = {k: v.clone() for k, v in modelo.state_dict().items()}
        contador_paciencia = 0
        # salva o melhor modelo separado do checkpoint
        torch.save(melhores_pesos, f'{PASTA_SAIDA}\\melhor_modelo.pt')
    else:
        contador_paciencia += 1

    lr_atual = otimizador.param_groups[0]['lr']
    print(f"Época {epoca:03d} | "
          f"Perda tr: {perda_tr:.4f} Acc tr: {acc_tr:.4f} | "
          f"Perda val: {perda_val:.4f} Acc val: {acc_val:.4f} | "
          f"LR: {lr_atual:.2e}"
          + (" ✓" if melhorou else f" (paciência {contador_paciencia}/{PACIENCIA})"))

    # Salva checkpoint a cada 5 épocas para não perder progresso
    if epoca % 5 == 0:
        salvar_checkpoint(epoca, modelo, otimizador, agendador,
                          melhor_acuracia, historico, ARQUIVO_CHECKPOINT)
        print(f"  → Checkpoint salvo na época {epoca}")

    # Early stopping só ativa após MIN_EPOCAS
    if epoca >= MIN_EPOCAS and contador_paciencia >= PACIENCIA:
        print(f"\n⏹ Early stopping na época {epoca}.")
        print(f"   Modelo não melhorou por {PACIENCIA} épocas consecutivas.")
        print(f"   Melhor acurácia de validação: {melhor_acuracia:.4f}")
        salvar_checkpoint(epoca, modelo, otimizador, agendador,
                          melhor_acuracia, historico, ARQUIVO_CHECKPOINT)
        break

# =============================================================================
# ETAPA 9 — AVALIAÇÃO FINAL NO TESTE
# =============================================================================

print("\n=== AVALIAÇÃO FINAL NO CONJUNTO DE TESTE ===")
modelo.load_state_dict(melhores_pesos)
_, acuracia_teste, predicoes_teste, rotulos_teste_lista = executar_epoca(
    carregador_teste, treino=False)

predicoes_teste     = np.array(predicoes_teste)
rotulos_teste_lista = np.array(rotulos_teste_lista)

print(f"Acurácia geral: {acuracia_teste:.4f}\n")
print(classification_report(
    rotulos_teste_lista, predicoes_teste,
    target_names=NOMES_ORDENADOS,
    zero_division=0, digits=4))

# =============================================================================
# ETAPA 10 — MATRIZ DE CONFUSÃO 7x7
# =============================================================================

matriz = confusion_matrix(rotulos_teste_lista, predicoes_teste)

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
ax.set_title('Matriz de Confusão 7×7\nNormal + 6 Arritmias Cardíacas',
             fontsize=14, fontweight='bold', pad=15)
plt.xticks(rotation=35, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig(f'{PASTA_SAIDA}\\matriz_confusao_7x7.png', dpi=150)
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

plt.suptitle('Histórico de Treinamento', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{PASTA_SAIDA}\\curvas_aprendizado.png', dpi=150)
plt.show()

# =============================================================================
# ETAPA 12 — SALVAR MODELO FINAL COM CONFIGURAÇÕES
# =============================================================================

torch.save({
    'pesos':           melhores_pesos,
    'configuracoes': {
        'num_classes':      NUM_CLASSES,
        'tamanho_patch':    TAMANHO_PATCH,
        'dimensao_modelo':  DIMENSAO_MODELO,
        'num_cabecas':      NUM_CABECAS,
        'num_camadas':      NUM_CAMADAS,
        'dropout':          DROPOUT,
    },
    'classes':          CLASSES_ORDENADAS,
    'nomes_classes':    NOMES_ORDENADOS,
    'melhor_acc_val':   melhor_acuracia,
}, f'{PASTA_SAIDA}\\ecg_transformer_final.pt')

print(f"\nArquivos gerados em: {PASTA_SAIDA}")
print("  - matriz_confusao_7x7.png")
print("  - curvas_aprendizado.png")
print("  - melhor_modelo.pt       ← melhor epoch salva automaticamente")
print("  - checkpoint.pt          ← estado completo para retomar treino")
print("  - ecg_transformer_final.pt ← modelo final com configurações")