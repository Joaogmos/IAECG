# =============================================================================
# API — serve o modelo para a interface web (frontend/)
#
#   GET  /api/modelo           arquitetura, métricas, limiares e saídas
#   GET  /api/exemplos         exames do CODE-test para demonstração (com laudo)
#   GET  /api/exemplos/{i}     analisa o exame i do CODE-test
#   POST /api/analisar         analisa um arquivo enviado (multipart)
#   GET  /                     interface (arquivos de frontend/)
#
# Nenhum exame enviado é gravado em disco: tudo é processado em memória.
#
# Uso:  python api_ecg.py                   (http://127.0.0.1:8000)
#       IAECG_MODELO=resultados/v3/cnn1d_s42/modelo_final.pt python api_ecg.py
# =============================================================================

import os
import argparse

import numpy as np
import pandas as pd
import h5py
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

from comum_ecg import PASTA_RAIZ, PASTA_CODE_TEST, DOENCAS, NOMES_COMPLETOS, ORDEM_CODE_TEST
from inferencia_ecg import carregar_modelo, ler_arquivos, analisar

TAMANHO_MAXIMO_MB = 50
EXEMPLOS_POR_CLASSE = 4

DISPOSITIVO = os.environ.get('IAECG_DISPOSITIVO', 'cpu')   # CPU: é o cenário de uso real
modelo, info = carregar_modelo(dispositivo=DISPOSITIVO)
print(f"Modelo carregado: {info['arquitetura']} ({info['caminho']}) em {DISPOSITIVO}")

app = FastAPI(title='IAECG', description='Apoio à triagem de ECG de 12 derivações (pesquisa — TCC Mackenzie)')

# ----- exemplos do CODE-test (pacientes que o modelo nunca viu) -----
def montar_exemplos():
    caminho = os.path.join(PASTA_CODE_TEST, 'ecg_tracings.hdf5')
    if not os.path.exists(caminho):
        return [], None
    ouro = pd.read_csv(os.path.join(PASTA_CODE_TEST, 'annotations', 'gold_standard.csv'))[DOENCAS]
    atributos = pd.read_csv(os.path.join(PASTA_CODE_TEST, 'attributes.csv'))

    escolhidos = list(ouro.index[ouro.sum(axis=1) == 0][:EXEMPLOS_POR_CLASSE])
    for d in DOENCAS:
        escolhidos += list(ouro.index[ouro[d] == 1][:EXEMPLOS_POR_CLASSE])
    exemplos = []
    for i in dict.fromkeys(escolhidos):             # sem repetir, na ordem
        laudo = [d for d in DOENCAS if ouro.loc[i, d] == 1]
        exemplos.append({'indice': int(i), 'idade': int(atributos.loc[i, 'age']),
                         'sexo': atributos.loc[i, 'sex'], 'laudo': laudo,
                         'laudo_nomes': [NOMES_COMPLETOS[d] for d in laudo] or ['Sem as 6 alterações']})
    return exemplos, caminho


EXEMPLOS, ARQUIVO_CODE_TEST = montar_exemplos()


@app.get('/api/modelo')
def obter_modelo():
    return {
        'arquitetura': info['arquitetura'], 'protocolo': info['protocolo'],
        'saidas': [{'sigla': s, 'nome': NOMES_COMPLETOS[s]} for s in info['saidas']],
        'limiares': info['limiares'],
        'metricas_teste': info['metricas_teste'],
        'metricas_code_test': info.get('metricas_code_test'),
        'formatos': ['.hdf5 (formato CODE)', '.npy', '.csv / .txt (12 colunas)', 'WFDB (.hea + .dat)'],
    }


@app.get('/api/exemplos')
def listar_exemplos():
    return EXEMPLOS


@app.get('/api/exemplos/{indice}')
def analisar_exemplo(indice: int, modo: str = 'equilibrado'):
    exemplo = next((e for e in EXEMPLOS if e['indice'] == indice), None)
    if exemplo is None:
        raise HTTPException(404, 'Exemplo não encontrado')
    with h5py.File(ARQUIVO_CODE_TEST, 'r') as f:
        sinal = f['tracings'][indice][:, ORDEM_CODE_TEST]
    resultado = analisar(sinal, 400, modelo, info, validar_modo(modo))
    resultado['exemplo'] = exemplo
    return resultado


@app.post('/api/analisar')
async def analisar_arquivo(arquivos: list[UploadFile] = File(...),
                           frequencia: float = Form(400),
                           indice: int = Form(0),
                           modo: str = Form('equilibrado')):
    conteudos = {}
    for arquivo in arquivos:
        conteudo = await arquivo.read()
        if len(conteudo) > TAMANHO_MAXIMO_MB * 2**20:
            raise HTTPException(413, f'Arquivo maior que {TAMANHO_MAXIMO_MB} MB')
        conteudos[arquivo.filename] = conteudo
    if not 50 <= frequencia <= 5000:
        raise HTTPException(422, 'Frequência de amostragem fora do esperado (50–5000 Hz)')
    try:
        sinal, freq = ler_arquivos(conteudos, frequencia, indice)
        return analisar(sinal, freq, modelo, info, validar_modo(modo))
    except (ValueError, KeyError, OSError) as erro:
        raise HTTPException(422, f'Não foi possível ler o ECG: {erro}')


def validar_modo(modo):
    if modo not in ('equilibrado', 'triagem'):
        raise HTTPException(422, "modo deve ser 'equilibrado' ou 'triagem'")
    return modo


app.mount('/', StaticFiles(directory=os.path.join(PASTA_RAIZ, 'frontend'), html=True), name='frontend')

if __name__ == '__main__':
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--porta', type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.porta)
