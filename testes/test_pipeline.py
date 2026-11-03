# Testes do pipeline (rotulação, divisão, métricas) e da inferência.
# Usam sinais sintéticos, então rodam sem os dados do CODE.
#
# Uso:  python -m pytest testes

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comum_ecg import (DOENCAS, NUM_AMOSTRAS, chave_estratificacao, dividir_por_paciente,
                       escolher_limiares, metricas_por_limiar, montar_rotulos, normalizar_sinais)
from inferencia_ecg import (DERIVACOES, ajustar_para_modelo, checar_qualidade, corrigir_aumentadas,
                            ler_arquivos, medir_ritmo, reordenar_derivacoes)


def ecg_sintetico(fs=400, segundos=10, bpm=75, semente=0):
    """Picos gaussianos estreitos (QRS) em DI e DII; DIII e aumentadas pela lei de Einthoven."""
    gerador = np.random.default_rng(semente)
    t = np.arange(int(fs * segundos)) / fs
    batidas = np.arange(0.3, segundos, 60 / bpm)
    qrs = sum(np.exp(-((t - b) / 0.012) ** 2) for b in batidas)
    onda_t = sum(0.25 * np.exp(-((t - b - 0.25) / 0.05) ** 2) for b in batidas)
    di  = 0.8 * qrs + onda_t + 0.01 * gerador.standard_normal(len(t))
    dii = 1.2 * qrs + 1.2 * onda_t + 0.01 * gerador.standard_normal(len(t))
    diii = dii - di
    precordiais = [(k / 3 - 1) * qrs + 0.02 * gerador.standard_normal(len(t)) for k in range(6)]
    return np.stack([di, dii, diii, -(di + dii) / 2, (di - diii) / 2, (dii + diii) / 2,
                     *precordiais], axis=1).astype(np.float32)


# ----- comum_ecg -----

def test_rotulos_e_estratificacao():
    df = pd.DataFrame({d: [False] * 4 for d in DOENCAS})
    df['normal_ecg'] = [True, False, False, False]
    df.loc[2, 'RBBB'] = True
    df.loc[3, ['RBBB', '1dAVb']] = True                     # duas doenças
    rotulos = montar_rotulos(df)
    assert rotulos.shape == (4, 7)
    chave = chave_estratificacao(rotulos)
    assert chave[0] == len(DOENCAS)                          # normal
    assert chave[1] == len(DOENCAS) + 1                      # outra alteração
    assert chave[2] == DOENCAS.index('RBBB')


def test_divisao_nao_repete_paciente():
    gerador = np.random.default_rng(1)
    n = 3000
    df = pd.DataFrame({d: gerador.random(n) < 0.05 for d in DOENCAS})
    df['normal_ecg'] = gerador.random(n) < 0.4
    df['patient_id'] = gerador.integers(0, 1200, n)          # pacientes com vários exames
    particao = dividir_por_paciente(df, montar_rotulos(df))
    conjuntos = {p: set(df.loc[particao == p, 'patient_id']) for p in ['treino', 'validacao', 'teste']}
    assert not conjuntos['treino'] & conjuntos['teste']
    assert not conjuntos['treino'] & conjuntos['validacao']
    assert not conjuntos['validacao'] & conjuntos['teste']
    assert 0.6 < np.mean(particao == 'treino') < 0.8


def test_limiares_e_metricas():
    gerador = np.random.default_rng(2)
    y = (gerador.random((2000, 2)) < 0.1).astype(np.float32)
    p = np.clip(y * 0.6 + gerador.random((2000, 2)) * 0.5, 0, 1)
    limiares = escolher_limiares(p, y)
    for j in range(2):
        triagem = metricas_por_limiar(y[:, j], p[:, j], limiares['triagem'][j])
        assert triagem['sensibilidade'] >= 0.95
        assert limiares['triagem'][j] <= limiares['equilibrado'][j]


def test_normalizacao_zscore():
    x = normalizar_sinais(np.random.default_rng(3).normal(5, 3, (2, NUM_AMOSTRAS, 12)))
    assert np.allclose(x.mean(axis=1), 0, atol=1e-4)
    assert np.allclose(x.std(axis=1), 1, atol=1e-3)


# ----- inferencia_ecg -----

def test_csv_com_tempo_e_nomes_e_500hz():
    sinal = ecg_sintetico(fs=500)
    nomes = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    tempo = np.arange(len(sinal))[:, None] / 500
    texto = 'tempo,' + ','.join(nomes) + '\n' + '\n'.join(
        ','.join(f'{v:.5f}' for v in linha) for linha in np.hstack([tempo, sinal]))
    lido, fs = ler_arquivos({'exame.csv': texto.encode()}, frequencia=500)
    assert lido.shape == sinal.shape and fs == 500
    ajustado, duracao = ajustar_para_modelo(lido, fs)
    assert ajustado.shape == (NUM_AMOSTRAS, 12)
    assert abs(duracao - 10) < 0.05


def test_reordena_derivacoes_pelo_nome():
    sinal = ecg_sintetico()
    ordem = list(reversed(range(12)))
    nomes = [DERIVACOES[i] for i in ordem]
    assert np.allclose(reordenar_derivacoes(sinal[:, ordem], nomes), sinal)
    with pytest.raises(ValueError):
        reordenar_derivacoes(sinal[:, :11], nomes[:11])


def test_einthoven_corrige_aumentadas_trocadas():
    sinal, _ = ajustar_para_modelo(ecg_sintetico(), 400)
    trocado = sinal[:, [0, 1, 2, 4, 5, 3, 6, 7, 8, 9, 10, 11]]       # como no CODE-test
    corrigido, alerta = corrigir_aumentadas(trocado)
    assert np.allclose(corrigido, sinal)
    assert alerta['gravidade'] == 'baixa'
    assert corrigir_aumentadas(sinal)[1] is None


def test_qualidade_derivacao_solta_e_registro_curto():
    sinal = ecg_sintetico(segundos=5)
    sinal[:, 6:9] = 0                                              # V1–V3 soltas
    ajustado, duracao = ajustar_para_modelo(sinal, 400)
    qualidade = checar_qualidade(ajustado, duracao)
    assert not qualidade['adequado']
    texto = ' '.join(a['mensagem'] for a in qualidade['alertas'])
    assert 'V1' in texto and 'curto' in texto


def test_frequencia_cardiaca():
    sinal, _ = ajustar_para_modelo(ecg_sintetico(bpm=90), 400)
    ritmo = medir_ritmo(sinal)
    assert abs(ritmo['frequencia_bpm'] - 90) < 3
    assert ritmo['irregularidade_rr'] < 0.05


def test_ensemble_salva_e_recarrega():
    import torch
    from inferencia_ecg import Ensemble, construir_modelo
    membros = [construir_modelo('cnn1d', {}), construir_modelo('transformer', {'hibrido': True})]
    configuracoes = {'membros': [{'nome': 'cnn1d', 'configuracoes': {}},
                                 {'nome': 'transformer', 'configuracoes': {'hibrido': True}}]}
    ensemble = Ensemble(membros).eval()
    copia = construir_modelo('ensemble', configuracoes).eval()
    copia.load_state_dict(ensemble.state_dict())
    x = torch.randn(2, NUM_AMOSTRAS, 12)
    with torch.no_grad():
        media = torch.stack([torch.sigmoid(m(x)) for m in membros]).mean(0)
        assert torch.allclose(torch.sigmoid(copia(x)), media, atol=1e-5)
