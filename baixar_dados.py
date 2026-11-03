# =============================================================================
# BAIXAR DADOS — CODE-15% e CODE-test direto do Zenodo para dados/
#
# Baixa só o que falta, confere o MD5 informado pelo Zenodo, extrai o .zip
# e apaga o .zip para economizar disco. Pode ser interrompido e rodado de
# novo: retoma o download do ponto em que parou.
#
# Uso:  python baixar_dados.py                  (CODE-15% completo + CODE-test)
#       python baixar_dados.py --partes 7 8 9   (só algumas partes)
#       python baixar_dados.py --so-code-test
#       python baixar_dados.py --so-ptbxl        (PTB-XL, validação externa em outro país)
# =============================================================================

import os
import json
import hashlib
import shutil
import zipfile
import argparse
import urllib.request
from glob import glob

PASTA_RAIZ  = os.path.dirname(os.path.abspath(__file__))
PASTA_DADOS = os.path.join(PASTA_RAIZ, 'dados')
PASTA_CODE_TEST = os.path.join(PASTA_DADOS, 'code_test')   # o zip extrai em code_test/data/

REGISTRO_CODE15   = '4916206'   # CODE-15% (Ribeiro et al., Lima et al. 2021)
REGISTRO_CODETEST = '3765780'   # CODE-test: 827 exames laudados por cardiologistas
PASTA_PTBXL = os.path.join(PASTA_DADOS, 'ptbxl')
URL_PTBXL   = 'https://physionet-open.s3.amazonaws.com/ptb-xl/ptb-xl-1.0.3.zip'


def listar_arquivos(registro):
    with urllib.request.urlopen(f'https://zenodo.org/api/records/{registro}') as resposta:
        dados = json.load(resposta)
    return {f['key']: f for f in dados['files']}


def md5_do_arquivo(caminho):
    h = hashlib.md5()
    with open(caminho, 'rb') as f:
        for bloco in iter(lambda: f.read(1 << 22), b''):
            h.update(bloco)
    return h.hexdigest()


def baixar(info, destino):
    """Download com retomada (HTTP Range) e conferência de MD5."""
    tamanho  = info['size']
    parcial  = destino + '.parcial'
    ja_baixado = os.path.getsize(parcial) if os.path.exists(parcial) else 0

    if ja_baixado < tamanho:
        pedido = urllib.request.Request(info['links']['self'],
                                        headers={'Range': f'bytes={ja_baixado}-'})
        with urllib.request.urlopen(pedido) as resposta, open(parcial, 'ab') as f:
            ultimo_aviso = 0
            while True:
                bloco = resposta.read(1 << 20)
                if not bloco:
                    break
                f.write(bloco)
                ja_baixado += len(bloco)
                if ja_baixado - ultimo_aviso > 250 * 2**20:
                    print(f"    {info['key']}: {ja_baixado / tamanho:.0%}", flush=True)
                    ultimo_aviso = ja_baixado

    if info.get('checksum'):
        esperado = info['checksum'].split(':')[-1]
        if md5_do_arquivo(parcial) != esperado:
            os.remove(parcial)
            raise RuntimeError(f"MD5 de {info['key']} não confere — arquivo apagado, rode de novo.")
    elif os.path.getsize(parcial) != tamanho:
        raise RuntimeError(f"{info['key']} veio incompleto — rode de novo para retomar.")
    os.replace(parcial, destino)


def extrair(arquivo_zip, pasta):
    """Cada arquivo é extraído com nome temporário e renomeado no fim: um .hdf5
    pela metade (extração interrompida) nunca aparece com o nome final."""
    with zipfile.ZipFile(arquivo_zip) as z:
        for membro in z.infolist():
            alvo = os.path.join(pasta, membro.filename)
            if membro.is_dir():
                os.makedirs(alvo, exist_ok=True)
                continue
            os.makedirs(os.path.dirname(alvo), exist_ok=True)
            with z.open(membro) as origem, open(alvo + '.extraindo', 'wb') as saida:
                shutil.copyfileobj(origem, saida, 1 << 22)
            os.replace(alvo + '.extraindo', alvo)


def baixar_e_extrair(info, pasta):
    os.makedirs(pasta, exist_ok=True)
    destino = os.path.join(pasta, info['key'])
    print(f"  Baixando {info['key']} ({info['size'] / 2**30:.1f} GB)...", flush=True)
    baixar(info, destino)
    if destino.endswith('.zip'):
        print(f"  Extraindo {info['key']}...", flush=True)
        extrair(destino, pasta)
        os.remove(destino)


def info_ptbxl():
    """PTB-XL 1.0.3 (PhysioNet, ODC-BY): o PhysioNet não publica MD5 do zip, confere só o tamanho."""
    pedido = urllib.request.Request(URL_PTBXL, method='HEAD')
    with urllib.request.urlopen(pedido) as resposta:
        tamanho = int(resposta.headers['Content-Length'])
    return {'key': 'ptb-xl-1.0.3.zip', 'size': tamanho, 'links': {'self': URL_PTBXL}, 'checksum': None}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--partes', type=int, nargs='*', default=list(range(18)))
    parser.add_argument('--so-code-test', action='store_true')
    parser.add_argument('--ptbxl', action='store_true', help='também baixa o PTB-XL (1,8 GB; validação externa)')
    parser.add_argument('--so-ptbxl', action='store_true')
    args = parser.parse_args()
    os.makedirs(PASTA_DADOS, exist_ok=True)

    if args.ptbxl or args.so_ptbxl:
        if not os.path.exists(os.path.join(PASTA_PTBXL, 'ptbxl_database.csv')):
            temporaria = os.path.join(PASTA_DADOS, 'ptbxl_zip')
            baixar_e_extrair(info_ptbxl(), temporaria)
            raiz = os.path.dirname(glob(os.path.join(temporaria, '**', 'ptbxl_database.csv'), recursive=True)[0])
            os.replace(raiz, PASTA_PTBXL)
            shutil.rmtree(temporaria, ignore_errors=True)
        if args.so_ptbxl:
            print("Pronto."); raise SystemExit(0)

    if not args.so_code_test:
        arquivos = listar_arquivos(REGISTRO_CODE15)
        if not os.path.exists(os.path.join(PASTA_DADOS, 'exams.csv')):
            baixar_e_extrair(arquivos['exams.csv'], PASTA_DADOS)
        for n in args.partes:
            if os.path.exists(os.path.join(PASTA_DADOS, f'exams_part{n}.hdf5')):
                print(f"  exams_part{n}.hdf5 já existe — pulando")
                continue
            baixar_e_extrair(arquivos[f'exams_part{n}.zip'], PASTA_DADOS)

    if not os.path.exists(os.path.join(PASTA_CODE_TEST, 'data', 'ecg_tracings.hdf5')):
        baixar_e_extrair(listar_arquivos(REGISTRO_CODETEST)['data.zip'], PASTA_CODE_TEST)
    print("Pronto.")
