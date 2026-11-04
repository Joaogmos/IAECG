# Testes da API com um modelo treinado de verdade. Pulam se não houver nenhum
# modelo_final.pt v3 na máquina (rode antes, por exemplo, cnn1d_ecg.py --limite 3000).
#
# Uso:  python -m pytest testes

import io
import os
import sys
from glob import glob

import numpy as np
import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

modelos = sorted(glob(os.path.join(RAIZ, 'resultados', '*', '*', 'modelo_final.pt')))
if not modelos:
    pytest.skip('nenhum modelo treinado em resultados/', allow_module_level=True)
os.environ.setdefault('IAECG_MODELO', modelos[0])

from fastapi.testclient import TestClient   # noqa: E402
from api_ecg import app                     # noqa: E402
from test_pipeline import ecg_sintetico     # noqa: E402

cliente = TestClient(app)


def test_info_do_modelo():
    resposta = cliente.get('/api/modelo')
    assert resposta.status_code == 200
    info = resposta.json()
    assert info['protocolo'] == 'v3'
    assert len(info['limiares']['equilibrado']) == 7


def test_analisa_csv_a_500hz():
    texto = '\n'.join(','.join(f'{v:.4f}' for v in linha) for linha in ecg_sintetico(fs=500))
    resposta = cliente.post('/api/analisar', data={'frequencia': '500', 'modo': 'triagem'},
                            files={'arquivos': ('exame.csv', texto.encode(), 'text/csv')})
    assert resposta.status_code == 200, resposta.text
    r = resposta.json()
    assert r['veredito'] in ('alterado', 'normal', 'outra_alteracao')
    assert len(r['sinal']['valores']) == 12 and len(r['sinal']['valores'][0]) == 2048
    assert abs(r['ritmo']['frequencia_bpm'] - 75) < 3
    assert r['saliencia']


def test_npy_e_erros():
    buffer = io.BytesIO()
    np.save(buffer, ecg_sintetico().T)                      # (12, amostras): a API vira sozinha
    ok = cliente.post('/api/analisar', files={'arquivos': ('exame.npy', buffer.getvalue())})
    assert ok.status_code == 200, ok.text

    formato = cliente.post('/api/analisar', files={'arquivos': ('exame.pdf', b'%PDF')})
    assert formato.status_code == 422 and 'não suportado' in formato.json()['detail']

    poucas = cliente.post('/api/analisar', files={'arquivos': ('x.csv', b'1,2,3\n4,5,6\n')})
    assert poucas.status_code == 422


def test_exemplos_do_code_test():
    exemplos = cliente.get('/api/exemplos').json()
    if not exemplos:
        pytest.skip('CODE-test não baixado')
    r = cliente.get(f"/api/exemplos/{exemplos[0]['indice']}").json()
    assert r['exemplo']['indice'] == exemplos[0]['indice']
    assert cliente.get('/api/exemplos/99999').status_code == 404


def test_triagem_em_lote(tmp_path):
    from triagem_lote import listar_exames, triar, gravar
    from inferencia_ecg import carregar_modelo
    np.save(tmp_path / 'bom.npy', ecg_sintetico(fs=400))
    np.save(tmp_path / 'curto.npy', ecg_sintetico(fs=400)[:800])   # 2 s: tem que pedir repetição
    (tmp_path / 'quebrado.csv').write_bytes(b'1,2,3\n')
    modelo, info = carregar_modelo()
    linhas = triar(listar_exames([str(tmp_path)]), modelo, info, 400, 'equilibrado')
    vereditos = {l['exame']: l['veredito'] for l in linhas}
    assert vereditos['curto.npy'] == 'repetir' and vereditos['quebrado.csv'] == 'erro'
    assert linhas[0]['exame'] == 'curto.npy' and linhas[-1]['exame'] == 'quebrado.csv'
    gravar(linhas, tmp_path / 'fila.csv')
    assert (tmp_path / 'fila.csv').read_text(encoding='utf-8-sig').startswith('posicao;exame')


def test_lote_pela_api():
    buffer = io.BytesIO()
    np.save(buffer, ecg_sintetico())
    arquivos = [('arquivos', ('a.npy', buffer.getvalue())), ('arquivos', ('b.csv', b'1,2\n'))]
    r = cliente.post('/api/lote', data={'modo': 'triagem'}, files=arquivos)
    assert r.status_code == 200, r.text
    corpo = r.json()
    assert corpo['total'] == 2 and corpo['fila'][-1]['veredito'] == 'erro'
    csv = cliente.post('/api/lote', data={'formato': 'csv'}, files=arquivos)
    assert csv.headers['content-type'].startswith('text/csv') and 'posicao;exame' in csv.text
