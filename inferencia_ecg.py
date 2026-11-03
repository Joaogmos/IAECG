# =============================================================================
# INFERÊNCIA — usa um modelo treinado (modelo_final.pt, protocolo v3) num ECG
# vindo de fora do CODE-15%
#
# Etapas, na ordem em que o exame passa:
#   1. leitura do arquivo (HDF5 do CODE, .npy, .csv/.txt, WFDB .hea + .dat)
#   2. reordenação das derivações para o padrão do treino
#   3. reamostragem para 400 Hz e ajuste para 4.096 amostras
#   4. checagem de qualidade (derivação solta, sinal curto, saturação)
#   5. medidas clássicas (FC e irregularidade do RR) para conferência
#   6. probabilidades das 7 saídas + limiares escolhidos na validação
#   7. saliência (gradiente × entrada): onde no traçado o modelo se apoiou
#
# Uso pela linha de comando:
#   python inferencia_ecg.py exame.csv --frequencia 500
#   python inferencia_ecg.py dados/code_test/data/ecg_tracings.hdf5 --indice 3
# A API (api_ecg.py) usa as mesmas funções.
# =============================================================================

import os
import io
import json
import argparse
import tempfile
from glob import glob

import numpy as np
import h5py
import torch
from scipy.signal import resample_poly, butter, sosfiltfilt, find_peaks
from fractions import Fraction

from comum_ecg import (PASTA_RESULTADOS, DOENCAS, NOMES_COMPLETOS, FREQUENCIA_HZ,
                       NUM_AMOSTRAS, NUM_DERIVACOES, normalizar_sinais)

# Ordem das derivações no CODE-15% / CODE-test (a mesma do treino)
DERIVACOES = ['DI', 'DII', 'DIII', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
APELIDOS = {
    'I': 'DI', 'D1': 'DI', 'LEADI': 'DI', 'II': 'DII', 'D2': 'DII', 'LEADII': 'DII',
    'III': 'DIII', 'D3': 'DIII', 'LEADIII': 'DIII',
}
DURACAO_MINIMA_S = 7          # o CODE tem exames de 7 e 10 s

# =============================================================================
# MODELO
# =============================================================================

def encontrar_melhor_modelo():
    """O modelo_final.pt v3 com maior AP macro no teste (resultados/v3/*)."""
    candidatos = []
    for caminho in glob(os.path.join(PASTA_RESULTADOS, 'v3', '*', 'modelo_final.pt')):
        info = torch.load(caminho, map_location='cpu', weights_only=False)
        candidatos.append((info['metricas_teste']['ap_macro'], caminho))
    if not candidatos:
        raise FileNotFoundError("Nenhum modelo v3 em resultados/v3/. Treine com cnn1d_ecg.py primeiro.")
    return max(candidatos)[1]


def construir_modelo(nome, configuracoes):
    """Arquitetura vazia a partir do 'nome' salvo no treino."""
    if nome == 'cnn1d':
        from cnn1d_ecg import ResNet1DECG
        return ResNet1DECG()
    from transformer_ecg import TransformerECG
    return TransformerECG(hibrido=configuracoes['hibrido'])


def carregar_modelo(caminho=None, dispositivo='cpu'):
    caminho = caminho or os.environ.get('IAECG_MODELO') or encontrar_melhor_modelo()
    info = torch.load(caminho, map_location='cpu', weights_only=False)
    if info.get('protocolo') != 'v3':
        raise ValueError(f"{caminho} é do protocolo v2 (softmax); a inferência usa modelos v3.")

    modelo = construir_modelo(info['nome'], info['configuracoes'])
    modelo.load_state_dict(info['pesos'])
    modelo.to(dispositivo).eval()
    info['caminho'] = caminho
    return modelo, info

# =============================================================================
# 1–2. LEITURA DE ARQUIVOS E ORDEM DAS DERIVAÇÕES
# =============================================================================

def padronizar_nome(nome):
    n = str(nome).strip().upper().replace(' ', '').replace('_', '')
    n = APELIDOS.get(n, n)
    for d in DERIVACOES:
        if n == d.upper():
            return d
    return None


def reordenar_derivacoes(sinal, nomes):
    """sinal (amostras, k). Coloca as colunas na ordem DERIVACOES; aceita só as 12."""
    if nomes is None:
        return sinal
    padronizados = [padronizar_nome(n) for n in nomes]
    faltando = [d for d in DERIVACOES if d not in padronizados]
    if faltando:
        raise ValueError(f"Derivações ausentes no arquivo: {', '.join(faltando)}")
    return sinal[:, [padronizados.index(d) for d in DERIVACOES]]


def orientar(sinal):
    """Garante o formato (amostras, 12)."""
    sinal = np.asarray(sinal, dtype=np.float32).squeeze()
    if sinal.ndim != 2 or NUM_DERIVACOES not in sinal.shape:
        raise ValueError(f"Esperado um ECG de 12 derivações; recebido formato {sinal.shape}")
    return sinal if sinal.shape[1] == NUM_DERIVACOES else sinal.T


def ler_csv(conteudo):
    texto = conteudo.decode('utf-8-sig', errors='replace')
    primeira = texto.splitlines()[0]
    separador = ';' if primeira.count(';') > primeira.count(',') else (',' if ',' in primeira else None)
    tem_cabecalho = any(c.isalpha() for c in primeira)
    dados = np.loadtxt(io.StringIO(texto), delimiter=separador, skiprows=int(tem_cabecalho), dtype=np.float32)
    nomes = [c.strip() for c in primeira.split(separador)] if tem_cabecalho else None
    if nomes and dados.shape[1] == NUM_DERIVACOES + 1:       # 1ª coluna = tempo
        dados, nomes = dados[:, 1:], nomes[1:]
    return dados, nomes


def ler_arquivos(arquivos, frequencia=FREQUENCIA_HZ, indice=0):
    """
    arquivos: dict {nome_do_arquivo: bytes}. Retorna (sinal (amostras, 12), frequência).
    WFDB precisa do .hea e do .dat juntos.
    """
    extensoes = {os.path.splitext(n)[1].lower(): n for n in arquivos}

    if '.hea' in extensoes:
        import wfdb
        with tempfile.TemporaryDirectory() as pasta:
            for nome, conteudo in arquivos.items():
                with open(os.path.join(pasta, os.path.basename(nome)), 'wb') as f:
                    f.write(conteudo)
            registro = wfdb.rdrecord(os.path.join(pasta, os.path.splitext(os.path.basename(extensoes['.hea']))[0]))
        return reordenar_derivacoes(orientar(registro.p_signal), registro.sig_name), float(registro.fs)

    if len(arquivos) != 1:
        raise ValueError("Envie um único arquivo (ou o par .hea + .dat do WFDB).")
    nome, conteudo = next(iter(arquivos.items()))
    extensao = os.path.splitext(nome)[1].lower()

    if extensao in ('.hdf5', '.h5'):
        with h5py.File(io.BytesIO(conteudo), 'r') as f:
            sinais = f['tracings']
            if not 0 <= indice < len(sinais):
                raise ValueError(f"Índice {indice} fora do arquivo ({len(sinais)} exames)")
            return orientar(sinais[indice]), float(FREQUENCIA_HZ)     # formato CODE: 400 Hz
    if extensao == '.npy':
        return orientar(np.load(io.BytesIO(conteudo))), float(frequencia)
    if extensao in ('.csv', '.txt'):
        dados, nomes = ler_csv(conteudo)
        return reordenar_derivacoes(orientar(dados), nomes), float(frequencia)
    raise ValueError(f"Formato {extensao} não suportado (use .hdf5, .npy, .csv, .txt ou WFDB .hea + .dat)")

# =============================================================================
# 3–5. PRÉ-PROCESSAMENTO, QUALIDADE E MEDIDAS
# =============================================================================

def trecho_util(sinal):
    """Remove os zeros de preenchimento das pontas (o CODE completa até 4.096)."""
    ativo = np.flatnonzero(np.abs(sinal).sum(axis=1) > 0)
    return sinal[ativo[0]:ativo[-1] + 1] if len(ativo) else sinal


def ajustar_para_modelo(sinal, frequencia):
    """Reamostra para 400 Hz e centraliza em 4.096 amostras (corta ou completa com zeros)."""
    sinal = np.nan_to_num(sinal)
    if abs(frequencia - FREQUENCIA_HZ) > 1e-6:
        razao = Fraction(FREQUENCIA_HZ / frequencia).limit_denominator(1000)
        sinal = resample_poly(sinal, razao.numerator, razao.denominator, axis=0).astype(np.float32)
    duracao_s = len(trecho_util(sinal)) / FREQUENCIA_HZ
    if len(sinal) >= NUM_AMOSTRAS:
        inicio = (len(sinal) - NUM_AMOSTRAS) // 2
        return sinal[inicio:inicio + NUM_AMOSTRAS], duracao_s
    falta = NUM_AMOSTRAS - len(sinal)
    return np.pad(sinal, ((falta // 2, falta - falta // 2), (0, 0))), duracao_s


def corrigir_aumentadas(sinal):
    """
    Pela lei de Einthoven, aVR = -(DI+DII)/2, aVL = (DI-DIII)/2 e aVF = (DII+DIII)/2.
    Se as colunas 4–6 estiverem em outra ordem (acontece entre equipamentos e
    até entre bases: o CODE-test usa aVL, aVF, aVR), reordena pela correlação.
    Devolve (sinal, alerta ou None).
    """
    util = trecho_util(sinal).astype(np.float64)
    if len(util) < FREQUENCIA_HZ:
        return sinal, None
    di, dii, diii = util[:, 0], util[:, 1], util[:, 2]
    esperadas = [-(di + dii) / 2, (di - diii) / 2, (dii + diii) / 2]
    corr = np.array([[np.corrcoef(e, util[:, 3 + c])[0, 1] for c in range(3)] for e in esperadas])
    corr = np.nan_to_num(corr)
    escolha = [int(np.argmax(linha)) for linha in corr]
    if sorted(escolha) != [0, 1, 2] or min(corr[i, escolha[i]] for i in range(3)) < 0.9:
        return sinal, {'gravidade': 'media',
                       'mensagem': 'DI, DII, DIII e as aumentadas não seguem a lei de Einthoven: '
                                   'confira a ordem e a colocação dos eletrodos dos membros'}
    if escolha == [0, 1, 2]:
        return sinal, None
    sinal = sinal[:, [0, 1, 2] + [3 + c for c in escolha] + list(range(6, 12))]
    return sinal, {'gravidade': 'baixa',
                   'mensagem': 'aVR, aVL e aVF estavam em outra ordem no arquivo e foram reordenadas'}


def checar_qualidade(sinal, duracao_s):
    """
    Alertas que tornam a leitura do modelo pouco confiável. Os limites são
    relativos (o arquivo pode estar em mV, µV ou unidade arbitrária).
    """
    alertas = []
    util = trecho_util(sinal)
    amplitude = np.percentile(np.abs(util), 99, axis=0)
    referencia = np.median(amplitude) + 1e-12

    planas = [DERIVACOES[i] for i in range(NUM_DERIVACOES)
              if amplitude[i] < 0.05 * referencia or np.std(util[:, i]) < 1e-6]
    if planas:
        alertas.append({'gravidade': 'alta' if len(planas) >= 3 else 'media',
                        'mensagem': f"Derivação(ões) sem sinal ou soltas: {', '.join(planas)}"})

    saturadas = [DERIVACOES[i] for i in range(NUM_DERIVACOES)
                 if np.mean(np.abs(util[:, i]) >= 0.999 * np.abs(util[:, i]).max()) > 0.02]
    if saturadas:
        alertas.append({'gravidade': 'media',
                        'mensagem': f"Possível saturação (sinal cortado) em: {', '.join(saturadas)}"})

    if duracao_s < DURACAO_MINIMA_S:
        alertas.append({'gravidade': 'alta',
                        'mensagem': f"Registro curto ({duracao_s:.1f} s); o modelo foi treinado com 7–10 s"})

    return {'adequado': not any(a['gravidade'] == 'alta' for a in alertas),
            'alertas': alertas, 'duracao_s': round(duracao_s, 2)}


def medir_ritmo(sinal):
    """
    Frequência cardíaca e irregularidade do RR a partir dos picos R em DII
    (filtro 5–15 Hz, realça o QRS). Serve para o médico conferir a saída do
    modelo — não entra na classificação.
    """
    util = trecho_util(sinal)
    if len(util) < 2 * FREQUENCIA_HZ:
        return None
    sos = butter(2, [5, 15], btype='bandpass', fs=FREQUENCIA_HZ, output='sos')
    energia = np.abs(sosfiltfilt(sos, util[:, DERIVACOES.index('DII')]))
    picos, _ = find_peaks(energia, distance=int(0.25 * FREQUENCIA_HZ),   # até 240 bpm
                          height=0.35 * np.percentile(energia, 99))
    if len(picos) < 3:
        return None
    rr = np.diff(picos) / FREQUENCIA_HZ
    return {
        'frequencia_bpm':  round(float(60 / np.median(rr)), 1),
        'rr_medio_ms':     round(float(1000 * rr.mean()), 1),
        'irregularidade_rr': round(float(rr.std() / rr.mean()), 3),   # coef. de variação
        'batimentos':      int(len(picos)),
        'picos_r':         (picos / FREQUENCIA_HZ).round(3).tolist(),
    }

# =============================================================================
# 6–7. PREVISÃO E SALIÊNCIA
# =============================================================================

def saliencia(modelo, x, saida, janela=40):
    """|gradiente × entrada| somado nas derivações e suavizado (~100 ms), de 0 a 1."""
    x = x.clone().requires_grad_(True)
    modelo.zero_grad(set_to_none=True)
    modelo(x)[0, saida].backward()
    mapa = (x.grad * x).abs().sum(dim=2)[0].detach().cpu().numpy()
    mapa = np.convolve(mapa, np.ones(janela) / janela, mode='same')
    return mapa / (mapa.max() + 1e-12)


def analisar(sinal, frequencia, modelo, info, modo='equilibrado', fator_reducao=2):
    """
    sinal: (amostras, 12) em qualquer frequência. Devolve um dict pronto para JSON.
    fator_reducao: decima o traçado devolvido para o gráfico (400 Hz → 200 Hz).
    """
    sinal, duracao_s = ajustar_para_modelo(orientar(sinal), frequencia)
    sinal, alerta_ordem = corrigir_aumentadas(sinal)
    qualidade = checar_qualidade(sinal, duracao_s)
    if alerta_ordem:
        qualidade['alertas'].append(alerta_ordem)
    ritmo     = medir_ritmo(sinal)

    dispositivo = next(modelo.parameters()).device
    x = torch.from_numpy(normalizar_sinais(sinal)[None]).to(dispositivo)
    with torch.no_grad():
        probs = torch.sigmoid(modelo(x).float())[0].cpu().numpy()

    limiares = info['limiares'][modo]
    saidas   = info['saidas']
    resultados = [{'sigla': s, 'nome': NOMES_COMPLETOS[s], 'probabilidade': float(probs[j]),
                   'limiar': float(limiares[j]), 'positivo': bool(probs[j] >= limiares[j])}
                  for j, s in enumerate(saidas)]
    achados = [r for r in resultados if r['sigla'] in DOENCAS and r['positivo']]
    normal  = next(r for r in resultados if r['sigla'] == 'Normal')

    if achados:
        veredito = 'alterado'
    elif normal['positivo']:
        veredito = 'normal'
    else:
        veredito = 'outra_alteracao'     # nenhuma das 6, mas também não parece normal

    mapas = {}
    for r in (achados or [max((r for r in resultados if r['sigla'] in DOENCAS),
                              key=lambda r: r['probabilidade'])]):
        mapas[r['sigla']] = saliencia(modelo, x, saidas.index(r['sigla']))[::fator_reducao].round(3).tolist()

    return {
        'veredito':   veredito,
        'modo':       modo,
        'resultados': resultados,
        'achados':    [r['sigla'] for r in achados],
        'qualidade':  qualidade,
        'ritmo':      ritmo,
        'saliencia':  mapas,
        'sinal': {
            'frequencia_hz': FREQUENCIA_HZ / fator_reducao,
            'derivacoes':    DERIVACOES,
            'valores':       sinal[::fator_reducao].T.round(4).tolist(),
        },
        'modelo': {'arquitetura': info['arquitetura'], 'protocolo': info['protocolo']},
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('arquivos', nargs='+', help='arquivo do ECG (ou .hea e .dat do WFDB)')
    parser.add_argument('--frequencia', type=float, default=FREQUENCIA_HZ, help='Hz (.csv/.npy)')
    parser.add_argument('--indice', type=int, default=0, help='exame dentro do HDF5')
    parser.add_argument('--modo', choices=['equilibrado', 'triagem'], default='equilibrado')
    parser.add_argument('--modelo', default=None, help='caminho do modelo_final.pt')
    args = parser.parse_args()

    modelo, info = carregar_modelo(args.modelo)
    conteudos = {}
    for caminho in args.arquivos:
        with open(caminho, 'rb') as f:
            conteudos[caminho] = f.read()
    sinal, frequencia = ler_arquivos(conteudos, args.frequencia, args.indice)
    resultado = analisar(sinal, frequencia, modelo, info, args.modo)

    print(f"Modelo: {info['arquitetura']} ({info['caminho']})")
    print(f"Veredito: {resultado['veredito']} | Achados: {resultado['achados'] or 'nenhum'}")
    for r in resultado['resultados']:
        print(f"  {r['nome']:25s} {r['probabilidade']:6.1%}  (limiar {r['limiar']:.2f})"
              + ("  ◀" if r['positivo'] else ""))
    print("Qualidade:", json.dumps(resultado['qualidade'], ensure_ascii=False))
    if resultado['ritmo']:
        print(f"FC {resultado['ritmo']['frequencia_bpm']} bpm | "
              f"irregularidade RR {resultado['ritmo']['irregularidade_rr']}")
