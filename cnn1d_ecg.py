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
#   Para a comparação ser justa, tudo fora da Etapa 5 vem de comum_ecg.py,
#   o mesmo módulo usado pelo Transformer. A única diferença de treino é a
#   taxa de aprendizado (1e-3, a usada por Ribeiro et al.).
#
# Uso:  python cnn1d_ecg.py              (semente 42)
#       python cnn1d_ecg.py --semente 7  (repetir com outra semente)
# =============================================================================

import argparse

import numpy as np
import torch
import torch.nn as nn

from comum_ecg import NUM_CLASSES, preparar_dados, treinar_e_avaliar

# Hiperparâmetros do modelo — ResNet 1D (Ribeiro et al., 2020)
NUM_DERIVACOES   = 12
TAMANHO_KERNEL   = 17                          # artigo usa 16; ímpar permite padding simétrico
FILTROS          = [64, 128, 196, 256, 320]    # camada de entrada + 4 blocos residuais
COMPRIMENTOS     = [4096, 1024, 256, 64, 16]   # amostras após cada etapa (reduz 4× por bloco)
DROPOUT          = 0.2                         # artigo: keep_prob 0.8 → dropout 0.2
TAXA_APRENDIZADO = 1e-3                        # LR de Ribeiro et al.; BatchNorm tolera LR maior

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--semente', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.semente)
    np.random.seed(args.semente)
    torch.backends.cudnn.benchmark = True        # entrada de tamanho fixo → cuDNN escolhe a conv mais rápida

    dados = preparar_dados()
    treinar_e_avaliar(
        nome='cnn1d', titulo='CNN 1D (ResNet)',
        modelo=ResNet1DECG(), taxa_aprendizado=TAXA_APRENDIZADO,
        dados=dados, semente=args.semente,
        configuracoes={
            'arquitetura':    'ResNet1D (Ribeiro et al., 2020)',
            'num_derivacoes': NUM_DERIVACOES,
            'tamanho_kernel': TAMANHO_KERNEL,
            'filtros':        FILTROS,
            'comprimentos':   COMPRIMENTOS,
            'dropout':        DROPOUT,
        })
