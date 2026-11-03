# =============================================================================
# ECG TRANSFORMER — Classificação de Arritmias Cardíacas
# Dataset: CODE-15% (Rede de Telessaúde de Minas Gerais — TNMG)
# Modelo: Vision Transformer adaptado para sinais 1D (ECG) — MODELO PRINCIPAL
# Tarefa: Multirrótulo — 6 alterações + ECG normal (7 sigmoides, protocolo v3)
#
# Dados, divisão, treino e avaliação ficam em comum_ecg.py (idênticos aos da
# CNN 1D). Este arquivo só define a arquitetura (Etapa 5) e a taxa de aprendizado.
#
# Uso:  python transformer_ecg.py              (semente 42)
#       python transformer_ecg.py --semente 7  (repetir com outra semente)
#       python transformer_ecg.py --patch 32 --dimensao 256 --camadas 6 --cabecas 8
#           (configuração fora do padrão: a pasta de saída ganha um sufixo; ver busca_transformer.py)
# =============================================================================

import argparse

import numpy as np
import torch
import torch.nn as nn

from comum_ecg import NUM_SAIDAS, NUM_AMOSTRAS, preparar_dados, treinar_e_avaliar

# Hiperparâmetros padrão do modelo (os do TCC 1)
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
    Saída:    (batch, 4096 / patch, d_model)
    """
    def __init__(self, tamanho_patch, dimensao):
        super().__init__()
        self.projecao = nn.Conv1d(
            in_channels=12,
            out_channels=dimensao,
            kernel_size=tamanho_patch,
            stride=tamanho_patch
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
    def __init__(self, dimensao):
        super().__init__()
        self.convolucoes = nn.Sequential(
            nn.Conv1d(12, 64, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, dimensao, kernel_size=9, stride=4, padding=4),
        )

    def forward(self, sinal):
        return self.convolucoes(sinal.permute(0, 2, 1)).permute(0, 2, 1)


class TransformerECG(nn.Module):
    """
    Transformer para classificação multirrótulo de ECG (7 saídas).
    Fluxo: Patch Embedding → CLS Token → Positional Emb →
           Transformer Encoder × N → CLS → Cabeça Linear → 7 logits
    hibrido=True troca o Patch Embedding pelo EmbeddingConvolucional.
    """
    def __init__(self, hibrido=False, tamanho_patch=TAMANHO_PATCH, dimensao_modelo=DIMENSAO_MODELO,
                 num_cabecas=NUM_CABECAS, num_camadas=NUM_CAMADAS, dropout=DROPOUT):
        super().__init__()
        num_patches = NUM_AMOSTRAS // 16 if hibrido else NUM_AMOSTRAS // tamanho_patch

        self.embedding_patch      = (EmbeddingConvolucional(dimensao_modelo) if hibrido
                                     else EmbeddingPatch(tamanho_patch, dimensao_modelo))
        self.token_cls            = nn.Parameter(torch.zeros(1, 1, dimensao_modelo))
        self.embedding_posicional = nn.Parameter(
            torch.zeros(1, num_patches + 1, dimensao_modelo))
        self.dropout_entrada      = nn.Dropout(dropout)

        camada_encoder = nn.TransformerEncoderLayer(
            d_model=dimensao_modelo,
            nhead=num_cabecas,
            dim_feedforward=dimensao_modelo * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer  = nn.TransformerEncoder(camada_encoder, num_layers=num_camadas,
                                                  enable_nested_tensor=False)
        self.normalizacao = nn.LayerNorm(dimensao_modelo)

        self.cabeca_classificacao = nn.Sequential(
            nn.Linear(dimensao_modelo, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, NUM_SAIDAS)
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


def modelo_das_configuracoes(c):
    """Recria o Transformer a partir do dict 'configuracoes' salvo no treino."""
    return TransformerECG(hibrido=c['hibrido'], tamanho_patch=c.get('patch', TAMANHO_PATCH),
                          dimensao_modelo=c.get('dimensao_modelo', DIMENSAO_MODELO),
                          num_cabecas=c.get('num_cabecas', NUM_CABECAS),
                          num_camadas=c.get('num_camadas', NUM_CAMADAS),
                          dropout=c.get('dropout', DROPOUT))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--semente', type=int, default=42)
    parser.add_argument('--hibrido', action='store_true',
                        help='usa 3 convoluções antes da atenção (tokens de 40 ms)')
    parser.add_argument('--patch', type=int, default=TAMANHO_PATCH, help='amostras por patch (só o puro)')
    parser.add_argument('--dimensao', type=int, default=DIMENSAO_MODELO)
    parser.add_argument('--camadas', type=int, default=NUM_CAMADAS)
    parser.add_argument('--cabecas', type=int, default=NUM_CABECAS)
    parser.add_argument('--lr', type=float, default=TAXA_APRENDIZADO)
    parser.add_argument('--limite', type=int, default=None,
                        help='usa só N exames (teste rápido do pipeline)')
    parser.add_argument('--partes', type=int, nargs='*', default=None,
                        help='só estas partes do CODE-15%% (padrão: todas em dados/)')
    args = parser.parse_args()

    torch.manual_seed(args.semente)
    np.random.seed(args.semente)

    nome = 'transformer_hibrido' if args.hibrido else 'transformer'
    padrao = (args.patch == TAMANHO_PATCH and args.dimensao == DIMENSAO_MODELO and args.camadas == NUM_CAMADAS
              and args.cabecas == NUM_CABECAS and args.lr == TAXA_APRENDIZADO)
    if not padrao:                      # não mistura com os resultados da configuração padrão
        nome += (f"_p{args.patch}" if not args.hibrido else '') + \
                f"_d{args.dimensao}_l{args.camadas}_h{args.cabecas}_lr{args.lr:g}"

    configuracoes = {
        'hibrido':         args.hibrido,
        'patch':           args.patch,
        'tamanho_token_amostras': 16 if args.hibrido else args.patch,
        'dimensao_modelo': args.dimensao,
        'num_cabecas':     args.cabecas,
        'num_camadas':     args.camadas,
        'dropout':         DROPOUT,
    }
    dados = preparar_dados(args.limite, args.partes)
    treinar_e_avaliar(
        nome=nome,
        titulo='Transformer híbrido (Conv + Transformer)' if args.hibrido else 'Transformer 1D',
        modelo=modelo_das_configuracoes(configuracoes), taxa_aprendizado=args.lr,
        dados=dados, semente=args.semente, configuracoes=configuracoes)
