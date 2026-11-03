# =============================================================================
# ENSEMBLE — junta modelos já treinados (média das probabilidades)
#
# Os limiares do ensemble são escolhidos de novo na validação, e a avaliação
# (teste interno + CODE-test) é a mesma dos modelos individuais. Sai em
# resultados/v3/ensemble_<nome>/, com um modelo_final.pt que a API carrega
# como qualquer outro.
#
# Uso:  python ensemble_ecg.py cnn1d                       (todas as sementes da ResNet)
#       python ensemble_ecg.py cnn1d transformer_hibrido   (mistura arquiteturas)
# =============================================================================

import os
import json
import argparse
from glob import glob

import torch

from comum_ecg import PASTA_RESULTADOS, PROTOCOLO, DISPOSITIVO, preparar_dados, avaliacao_final
from inferencia_ecg import Ensemble, construir_modelo

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('modelos', nargs='+', help="nomes das pastas sem a semente (ex.: cnn1d)")
    args = parser.parse_args()

    dados = preparar_dados()
    membros, configuracoes, titulos = [], [], []
    for nome in args.modelos:
        for pasta in sorted(glob(os.path.join(PASTA_RESULTADOS, PROTOCOLO, f'{nome}_s*'))):
            with open(os.path.join(pasta, 'metricas.json'), encoding='utf-8') as f:
                metricas = json.load(f)
            if metricas['assinatura_dados'] != dados['assinatura']:
                print(f"  {pasta}: treinado com outros dados — fora do ensemble")
                continue
            modelo = construir_modelo(metricas['nome'], metricas['configuracoes'])
            modelo.load_state_dict(torch.load(os.path.join(pasta, 'melhor_modelo.pt'), map_location=DISPOSITIVO))
            membros.append(modelo)
            configuracoes.append({'nome': metricas['nome'], 'configuracoes': metricas['configuracoes'],
                                  'semente': metricas['semente']})
            titulos.append(f"{metricas['modelo']} s{metricas['semente']}")
            print(f"  + {titulos[-1]}")

    if len(membros) < 2:
        raise SystemExit("O ensemble precisa de pelo menos 2 modelos treinados com os dados atuais.")

    nome_saida = 'ensemble_' + '_'.join(args.modelos)
    pasta_saida = os.path.join(PASTA_RESULTADOS, PROTOCOLO, nome_saida)
    os.makedirs(pasta_saida, exist_ok=True)
    avaliacao_final('ensemble', f'Ensemble ({len(membros)} modelos)', Ensemble(membros), None, dados, 0,
                    {'membros': configuracoes, 'titulos': titulos}, None, None, pasta_saida)
