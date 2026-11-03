# =============================================================================
# REAVALIAR — refaz a avaliação final (limiares, teste, CODE-test, figuras e
# modelo_final.pt) de um treino já feito, sem treinar de novo.
#
# Útil quando a avaliação muda (ex.: correção da ordem das derivações do
# CODE-test) e o treino, que leva horas, já está pronto.
#
# Uso:  python reavaliar_modelo.py resultados/v3/cnn1d_s42
#       python reavaliar_modelo.py resultados/v3/cnn1d_s42 --partes 0 1 2 3 4 5 6
# =============================================================================

import os
import json
import argparse

import torch

from comum_ecg import DISPOSITIVO, preparar_dados, avaliacao_final
from inferencia_ecg import construir_modelo

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pasta', help='pasta do treino (com checkpoint.pt e melhor_modelo.pt)')
    parser.add_argument('--limite', type=int, default=None, help='o mesmo --limite usado no treino')
    parser.add_argument('--partes', type=int, nargs='*', default=None,
                        help='só estas partes do CODE-15%% (ex.: 0 1 2 3 4 5 6), se o treino usou menos')
    args = parser.parse_args()

    with open(os.path.join(args.pasta, 'metricas.json'), encoding='utf-8') as f:
        anterior = json.load(f)
    estado = torch.load(os.path.join(args.pasta, 'checkpoint.pt'), map_location='cpu')

    dados = preparar_dados(args.limite, args.partes)
    if estado['assinatura'] != dados['assinatura']:
        raise SystemExit(f"O treino usou outros dados (assinatura {estado['assinatura']}, "
                         f"agora {dados['assinatura']}). Rode com os mesmos HDF5 em dados/.")

    modelo = construir_modelo(anterior['nome'], anterior['configuracoes'])
    modelo.load_state_dict(torch.load(os.path.join(args.pasta, 'melhor_modelo.pt'), map_location=DISPOSITIVO))
    avaliacao_final(anterior['nome'], anterior['modelo'], modelo, anterior['taxa_aprendizado'], dados,
                    anterior['semente'], anterior['configuracoes'], estado['historico'],
                    estado['melhor_ap'], os.path.abspath(args.pasta))
