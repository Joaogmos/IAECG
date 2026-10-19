# =============================================================================
# ECG TRANSFORMER — Classificação de Arritmias Cardíacas
# Dataset: CODE-15% (Rede de Telessaúde de Minas Gerais — TNMG)
# Modelo: Vision Transformer adaptado para sinais 1D (ECG) — MODELO PRINCIPAL
# Tarefa: Multiclass — Normal + 6 arritmias (7 classes)
#
# Dados, divisão, treino e avaliação ficam em comum_ecg.py (idênticos aos da
# CNN 1D). Este arquivo só define a arquitetura (Etapa 5) e a taxa de aprendizado.
#
# Uso:  python transformer_ecg.py              (semente 42)
#       python transformer_ecg.py --semente 7  (repetir com outra semente)
# =============================================================================

import argparse

import numpy as np
import torch
import torch.nn as nn

from comum_ecg import NUM_CLASSES, preparar_dados, treinar_e_avaliar

# Hiperparâmetros do modelo
TAMANHO_PATCH    = 64       # 4096 amostras / 64 = 64 patches de 160 ms
DIMENSAO_MODELO  = 128
NUM_CABECAS      = 4
NUM_CAMADAS      = 4
DROPOUT          = 0.1
TAXA_APRENDIZADO = 3e-4

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


class EmbeddingConvolucional(nn.Module):
    """
    Variante híbrida: 3 convoluções extraem a morfologia local (ondas P, QRS, T)
    antes da atenção, em vez de projetar cada patch de 160 ms com uma única
    camada linear. Cada token passa a representar 40 ms (redução total de 16×).
    Entrada:  (batch, 4096, 12)
    Saída:    (batch, 256 tokens, d_model)
    """
    def __init__(self):
        super().__init__()
        self.convolucoes = nn.Sequential(
            nn.Conv1d(12, 64, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, DIMENSAO_MODELO, kernel_size=9, stride=4, padding=4),
        )

    def forward(self, sinal):
        return self.convolucoes(sinal.permute(0, 2, 1)).permute(0, 2, 1)


class TransformerECG(nn.Module):
    """
    Transformer para classificação multiclass de ECG (7 classes).
    Fluxo: Patch Embedding → CLS Token → Positional Emb →
           Transformer Encoder × 4 → CLS → Cabeça Linear → 7 logits
    hibrido=True troca o Patch Embedding pelo EmbeddingConvolucional.
    """
    def __init__(self, hibrido=False):
        super().__init__()
        num_patches = 4096 // 16 if hibrido else 4096 // TAMANHO_PATCH

        self.embedding_patch      = EmbeddingConvolucional() if hibrido else EmbeddingPatch()
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
        self.transformer  = nn.TransformerEncoder(camada_encoder, num_layers=NUM_CAMADAS,
                                                  enable_nested_tensor=False)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--semente', type=int, default=42)
    parser.add_argument('--hibrido', action='store_true',
                        help='usa 3 convoluções antes da atenção (tokens de 40 ms)')
    args = parser.parse_args()

    torch.manual_seed(args.semente)
    np.random.seed(args.semente)

    dados = preparar_dados()
    treinar_e_avaliar(
        nome='transformer_hibrido' if args.hibrido else 'transformer',
        titulo='Transformer híbrido (Conv + Transformer)' if args.hibrido else 'Transformer 1D',
        modelo=TransformerECG(hibrido=args.hibrido), taxa_aprendizado=TAXA_APRENDIZADO,
        dados=dados, semente=args.semente,
        configuracoes={
            'hibrido':         args.hibrido,
            'tamanho_patch':   16 if args.hibrido else TAMANHO_PATCH,
            'dimensao_modelo': DIMENSAO_MODELO,
            'num_cabecas':     NUM_CABECAS,
            'num_camadas':     NUM_CAMADAS,
            'dropout':         DROPOUT,
        })
