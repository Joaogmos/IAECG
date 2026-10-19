# =============================================================================
# COMUM — dados, treino e avaliação compartilhados pelos dois modelos
#
# transformer_ecg.py e cnn1d_ecg.py só definem a arquitetura (Etapa 5) e
# chamam treinar_e_avaliar(). Todo o resto mora aqui, então a comparação
# Transformer x CNN usa garantidamente os mesmos exames, a mesma divisão,
# o mesmo augmentation, a mesma loss, o mesmo agendador e o mesmo early stopping.
#
# Saídas de cada execução em resultados/<nome>_s<semente>/:
#   metricas.json, matriz_confusao.png, curvas_aprendizado.png,
#   melhor_modelo.pt, checkpoint.pt, modelo_final.pt
# =============================================================================

import os
import json
import math
import time
import hashlib
from glob import glob

import numpy as np
import pandas as pd
import h5py
import matplotlib
matplotlib.use('Agg')                        # salva figuras sem abrir janela
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, f1_score, balanced_accuracy_score)

# =============================================================================
# CONFIGURAÇÕES GERAIS
# =============================================================================

PASTA_RAIZ       = os.path.dirname(os.path.abspath(__file__))
PASTA_DADOS      = os.path.join(PASTA_RAIZ, 'dados')
ARQUIVO_CSV      = os.path.join(PASTA_DADOS, 'exams.csv')
PASTA_RESULTADOS = os.path.join(PASTA_RAIZ, 'resultados')

DOENCAS = ['1dAVb', 'RBBB', 'LBBB', 'SB', 'ST', 'AF']

NOMES_COMPLETOS = {
    'Normal': 'Normal',
    '1dAVb':  'Bloqueio AV 1º Grau',
    'RBBB':   'Bloqueio Ramo Direito',
    'LBBB':   'Bloqueio Ramo Esquerdo',
    'SB':     'Bradicardia Sinusal',
    'ST':     'Taquicardia Sinusal',
    'AF':     'Fibrilação Atrial',
}

CLASSES_ORDENADAS = ['Normal'] + DOENCAS
NOMES_ORDENADOS   = [NOMES_COMPLETOS[c] for c in CLASSES_ORDENADAS]
NUM_CLASSES       = len(CLASSES_ORDENADAS)   # 7

# Hiperparâmetros de treino (iguais para os dois modelos, exceto a LR)
AMOSTRAS_POR_CLASSE = 2000   # teto por classe; classes menores entram com tudo que tiverem
TAMANHO_BATCH       = 32
EPOCAS              = 200    # alto — early stopping controla quando parar
PACIENCIA           = 20
MIN_EPOCAS          = 30
SEMENTE_DIVISAO     = 42     # fixa: os dois modelos (e todas as sementes) usam o mesmo split

DISPOSITIVO = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USA_AMP     = DISPOSITIVO.type == 'cuda'     # bfloat16 na GPU: ~2× mais rápido, mesma acurácia

# =============================================================================
# ETAPA 1 — CARREGAMENTO E PREPARAÇÃO DOS DADOS
# =============================================================================

def carregar_metadados(arquivo_csv, pasta_dados):
    df = pd.read_csv(arquivo_csv)
    arquivos_disponiveis = sorted(
        os.path.basename(p) for p in glob(os.path.join(pasta_dados, '*.hdf5')))
    print(f"Arquivos HDF5 encontrados ({len(arquivos_disponiveis)}): {arquivos_disponiveis}")
    df = df[df['trace_file'].isin(arquivos_disponiveis)].reset_index(drop=True)
    print(f"Total de exames disponíveis: {len(df):,}")
    return df


def atribuir_classes(df):
    """
    Classe 0 (Normal): ECG laudado como normal (normal_ecg) e sem nenhuma das 6 doenças.
    Classes 1–6: exatamente uma das 6 doenças.
    -1 (descartado): mais de uma doença, ou "sem as 6 doenças" mas com outra
    alteração no laudo. Antes esses ECGs alterados entravam como Normal
    (~57% do grupo "sem doença" não tem normal_ecg), o que sujava a classe.
    """
    n_doencas = df[DOENCAS].sum(axis=1)
    classe    = np.full(len(df), -1)
    classe[(n_doencas == 0) & df['normal_ecg']] = 0
    for i, doenca in enumerate(DOENCAS, start=1):
        classe[(n_doencas == 1) & df[doenca]] = i
    return classe


def balancear_dataset(df, amostras_por_classe):
    grupos = []
    for indice_classe in range(NUM_CLASSES):
        subset = df[df['classe'] == indice_classe]
        n      = min(len(subset), amostras_por_classe)
        grupos.append(subset.sample(n, random_state=SEMENTE_DIVISAO))
        print(f"  {NOMES_ORDENADOS[indice_classe]:30s}: {n:,} de {len(subset):,} exames")

    return (pd.concat(grupos)
              .sample(frac=1, random_state=SEMENTE_DIVISAO)
              .reset_index(drop=True))

# =============================================================================
# ETAPA 2 — CARREGAMENTO DOS SINAIS EM LOTE (com cache em .npy)
# =============================================================================

def normalizar_sinais(sinais):
    """Z-score por derivação, por exame (mesmo efeito do StandardScaler antigo)."""
    media  = sinais.mean(axis=1, keepdims=True)
    desvio = sinais.std(axis=1, keepdims=True)
    return ((sinais - media) / (desvio + 1e-8)).astype(np.float32)


def carregar_sinais_em_lote(df_balanceado, pasta_dados):
    sinais = np.empty((len(df_balanceado), 4096, 12), dtype=np.float32)

    for nome_arquivo, grupo in df_balanceado.groupby('trace_file'):
        caminho = os.path.join(pasta_dados, nome_arquivo)
        print(f"  Lendo {nome_arquivo} ({len(grupo):,} exames)...")
        with h5py.File(caminho, 'r') as arquivo:
            posicao_por_id = {eid: i for i, eid in enumerate(arquivo['exam_id'][:])}
            posicoes = np.array([posicao_por_id[eid] for eid in grupo['exam_id']])
            ordem    = np.argsort(posicoes)          # h5py exige índices crescentes
            lidos    = arquivo['tracings'][posicoes[ordem]]
            sinais[grupo.index.to_numpy()[ordem]] = normalizar_sinais(lidos)

    return sinais


def preparar_dados():
    """
    Retorna dict com sinais/rótulos de treino, validação e teste.
    A divisão é POR PACIENTE: o mesmo paciente nunca aparece em dois
    conjuntos (o CODE-15% tem pacientes com vários exames; dividir por
    exame vazaria o paciente do treino para o teste e inflaria a acurácia).
    """
    df = carregar_metadados(ARQUIVO_CSV, PASTA_DADOS)
    df['classe'] = atribuir_classes(df)

    print("\nDistribuição antes do balanceamento:")
    for i, nome in enumerate(NOMES_ORDENADOS):
        print(f"  {nome:30s}: {(df['classe'] == i).sum():,}")
    print(f"  {'(descartados)':30s}: {(df['classe'] < 0).sum():,}")
    df = df[df['classe'] >= 0].reset_index(drop=True)

    print("\nBalanceando dataset:")
    df_bal = balancear_dataset(df, AMOSTRAS_POR_CLASSE)
    print(f"Total após balanceamento: {len(df_bal):,} exames")

    # Divisão 70/15/15 estratificada por classe e agrupada por paciente
    divisor = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=SEMENTE_DIVISAO)
    dobra   = np.empty(len(df_bal), dtype=int)
    for i, (_, indices) in enumerate(divisor.split(df_bal, df_bal['classe'], df_bal['patient_id'])):
        dobra[indices] = i
    df_bal['particao'] = np.select([dobra < 3, dobra < 6], ['teste', 'validacao'], 'treino')

    # Cache: a leitura dos HDF5 é a parte lenta; a 2ª execução lê o .npy direto
    assinatura = hashlib.md5(df_bal['exam_id'].to_numpy().tobytes()).hexdigest()[:10]
    arquivo_cache = os.path.join(PASTA_DADOS, f'cache_sinais_{assinatura}.npy')
    if os.path.exists(arquivo_cache):
        print(f"\nUsando cache de sinais: {arquivo_cache}")
        sinais = np.load(arquivo_cache, mmap_mode='r')
    else:
        print("\nCarregando sinais do ECG...")
        sinais = carregar_sinais_em_lote(df_bal, PASTA_DADOS)
        np.save(arquivo_cache, sinais)
        print(f"Cache salvo em {arquivo_cache}")

    dados = {'assinatura': assinatura, 'df': df_bal}
    for particao in ['treino', 'validacao', 'teste']:
        mascara = (df_bal['particao'] == particao).to_numpy()
        dados[particao] = (np.ascontiguousarray(sinais[mascara]),
                           df_bal.loc[mascara, 'classe'].to_numpy())

    print(f"\nTreino: {len(dados['treino'][1]):,} | Validação: {len(dados['validacao'][1]):,} "
          f"| Teste: {len(dados['teste'][1]):,}  (sem paciente repetido entre conjuntos)")
    return dados

# =============================================================================
# ETAPA 4 — DATALOADER E DATA AUGMENTATION
# =============================================================================

def criar_dataloader(sinais, rotulos, embaralhar=False):
    dataset = TensorDataset(torch.from_numpy(sinais), torch.from_numpy(rotulos).long())
    return DataLoader(dataset, batch_size=TAMANHO_BATCH, shuffle=embaralhar,
                      pin_memory=torch.cuda.is_available())


def aumentar_batch(sinais):
    """
    Augmentation aplicado direto na GPU, por exame:
      - ruído gaussiano leve (σ = 0.02)
      - escala de amplitude entre 0.9 e 1.1
      - deslocamento temporal de até ±0.5 s (200 amostras a 400 Hz)
    A inversão de polaridade (sinal = -sinal) da versão antiga foi removida:
    ela inverte a morfologia do QRS, que é justamente o que diferencia
    bloqueio de ramo direito de esquerdo.
    """
    b = sinais.size(0)
    sinais = sinais + torch.randn_like(sinais) * 0.02
    sinais = sinais * (0.9 + 0.2 * torch.rand(b, 1, 1, device=sinais.device))
    deslocamento = int(torch.randint(-200, 201, (1,)))
    return torch.roll(sinais, shifts=deslocamento, dims=1)

# =============================================================================
# ETAPA 6 a 12 — TREINO, CHECKPOINT, AVALIAÇÃO E FIGURAS
# =============================================================================

def treinar_e_avaliar(nome, titulo, modelo, taxa_aprendizado, dados, semente, configuracoes):
    """
    nome:   identificador curto ('transformer', 'cnn1d') — vira a pasta de saída
    titulo: nome legível para os gráficos ('Transformer 1D', 'CNN 1D (ResNet)')
    """
    pasta_saida = os.path.join(PASTA_RESULTADOS, f'{nome}_s{semente}')
    os.makedirs(pasta_saida, exist_ok=True)
    arquivo_checkpoint = os.path.join(pasta_saida, 'checkpoint.pt')
    arquivo_melhor     = os.path.join(pasta_saida, 'melhor_modelo.pt')

    modelo = modelo.to(DISPOSITIVO)
    total_parametros = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    print(f"\n{titulo} | Parâmetros treináveis: {total_parametros:,} | Dispositivo: {DISPOSITIVO}")

    carregador_treino    = criar_dataloader(*dados['treino'],    embaralhar=True)
    carregador_validacao = criar_dataloader(*dados['validacao'])
    carregador_teste     = criar_dataloader(*dados['teste'])

    # ----- Etapa 6: loss, otimizador e agendador -----
    passos_por_epoca = len(carregador_treino)
    passos_warmup    = max(50, min(200, passos_por_epoca // 2))
    total_passos     = EPOCAS * passos_por_epoca

    funcao_perda = nn.CrossEntropyLoss(label_smoothing=0.1)
    otimizador   = torch.optim.AdamW(modelo.parameters(), lr=taxa_aprendizado, weight_decay=1e-3)

    def calcular_taxa_aprendizado(passo_atual):
        """Warmup linear + decaimento cosseno proporcional ao treino real."""
        if passo_atual < passos_warmup:
            return passo_atual / max(1, passos_warmup)
        progresso = (passo_atual - passos_warmup) / max(1, total_passos - passos_warmup)
        return max(0.05, 0.5 * (1 + math.cos(math.pi * progresso)))

    agendador = torch.optim.lr_scheduler.LambdaLR(otimizador, calcular_taxa_aprendizado)

    # ----- Etapa 7: checkpoint (retoma de onde parou) -----
    historico = {'perda_treino': [], 'perda_val': [], 'acc_treino': [], 'acc_val': [],
                 'tempo_epoca': []}
    epoca_inicial, melhor_acuracia, contador_paciencia = 1, 0.0, 0

    if os.path.exists(arquivo_checkpoint):
        estado = torch.load(arquivo_checkpoint, map_location=DISPOSITIVO)
        if estado.get('assinatura') == dados['assinatura']:
            modelo.load_state_dict(estado['modelo'])
            otimizador.load_state_dict(estado['otimizador'])
            agendador.load_state_dict(estado['agendador'])
            historico          = estado['historico']
            epoca_inicial      = estado['epoca'] + 1
            melhor_acuracia    = estado['melhor_acuracia']
            contador_paciencia = estado['contador_paciencia']
            print(f"Retomando da época {epoca_inicial} | Melhor acc: {melhor_acuracia:.4f}")
        else:
            print("Checkpoint é de outro conjunto de dados — ignorando e começando do zero.")

    def salvar_checkpoint(epoca):
        torch.save({
            'epoca': epoca, 'assinatura': dados['assinatura'],
            'modelo': modelo.state_dict(), 'otimizador': otimizador.state_dict(),
            'agendador': agendador.state_dict(), 'melhor_acuracia': melhor_acuracia,
            'contador_paciencia': contador_paciencia, 'historico': historico,
        }, arquivo_checkpoint)

    # ----- Etapa 8: loop de treino -----
    def executar_epoca(carregador, treino=True):
        modelo.train() if treino else modelo.eval()
        perda_total, acertos, total = 0.0, 0, 0
        todas_probs, todos_rotulos = [], []

        with torch.set_grad_enabled(treino):
            for sinais_batch, rotulos_batch in carregador:
                sinais_batch  = sinais_batch.to(DISPOSITIVO, non_blocking=True)
                rotulos_batch = rotulos_batch.to(DISPOSITIVO, non_blocking=True)
                if treino:
                    sinais_batch = aumentar_batch(sinais_batch)

                with torch.autocast(DISPOSITIVO.type, dtype=torch.bfloat16, enabled=USA_AMP):
                    logits = modelo(sinais_batch)
                    perda  = funcao_perda(logits.float(), rotulos_batch)

                if treino:
                    otimizador.zero_grad(set_to_none=True)
                    perda.backward()
                    nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
                    otimizador.step()
                    agendador.step()

                acertos     += (logits.argmax(dim=1) == rotulos_batch).sum().item()
                total       += rotulos_batch.size(0)
                perda_total += perda.item()
                if not treino:
                    todas_probs.append(torch.softmax(logits.float(), dim=1).cpu())
                    todos_rotulos.append(rotulos_batch.cpu())

        probs   = torch.cat(todas_probs).numpy() if todas_probs else None
        rotulos = torch.cat(todos_rotulos).numpy() if todos_rotulos else None
        return perda_total / len(carregador), acertos / total, probs, rotulos

    print(f"\n=== INÍCIO DO TREINO (época {epoca_inicial} até {EPOCAS}) ===")
    print(f"Early stopping: paciência de {PACIENCIA} épocas | Mínimo de {MIN_EPOCAS} épocas\n")

    for epoca in range(epoca_inicial, EPOCAS + 1):
        inicio = time.time()
        perda_tr,  acc_tr,  _, _ = executar_epoca(carregador_treino,    treino=True)
        perda_val, acc_val, _, _ = executar_epoca(carregador_validacao, treino=False)
        duracao = time.time() - inicio

        historico['perda_treino'].append(perda_tr)
        historico['perda_val'].append(perda_val)
        historico['acc_treino'].append(acc_tr)
        historico['acc_val'].append(acc_val)
        historico['tempo_epoca'].append(duracao)

        melhorou = acc_val > melhor_acuracia
        if melhorou:
            melhor_acuracia    = acc_val
            contador_paciencia = 0
            torch.save(modelo.state_dict(), arquivo_melhor)
        else:
            contador_paciencia += 1

        print(f"Época {epoca:03d} | "
              f"Perda tr: {perda_tr:.4f} Acc tr: {acc_tr:.4f} | "
              f"Perda val: {perda_val:.4f} Acc val: {acc_val:.4f} | "
              f"LR: {otimizador.param_groups[0]['lr']:.2e} | {duracao:.0f}s"
              + (" ✓" if melhorou else f" (paciência {contador_paciencia}/{PACIENCIA})"),
              flush=True)

        parar = epoca >= MIN_EPOCAS and contador_paciencia >= PACIENCIA
        if epoca % 5 == 0 or parar:
            salvar_checkpoint(epoca)
        if parar:
            print(f"\n⏹ Early stopping na época {epoca}. Melhor acc de validação: {melhor_acuracia:.4f}")
            break

    # ----- Etapa 9: avaliação final no teste (com os melhores pesos) -----
    print("\n=== AVALIAÇÃO FINAL NO CONJUNTO DE TESTE ===")
    modelo.load_state_dict(torch.load(arquivo_melhor, map_location=DISPOSITIVO))

    inicio = time.time()
    _, acuracia_teste, probs_teste, rotulos_teste = executar_epoca(carregador_teste, treino=False)
    tempo_inferencia_ms = 1000 * (time.time() - inicio) / len(rotulos_teste)
    predicoes_teste = probs_teste.argmax(axis=1)

    rotulos_idx = list(range(NUM_CLASSES))
    print(f"Acurácia geral: {acuracia_teste:.4f}\n")
    print(classification_report(rotulos_teste, predicoes_teste, labels=rotulos_idx,
                                target_names=NOMES_ORDENADOS, zero_division=0, digits=4))

    relatorio = classification_report(rotulos_teste, predicoes_teste, labels=rotulos_idx,
                                      target_names=NOMES_ORDENADOS, zero_division=0,
                                      output_dict=True)
    matriz = confusion_matrix(rotulos_teste, predicoes_teste, labels=rotulos_idx)

    # Métricas por classe no formato usado em artigos médicos (um-contra-todos)
    por_classe = {}
    for i, sigla in enumerate(CLASSES_ORDENADAS):
        vp = matriz[i, i]
        fn = matriz[i].sum() - vp
        fp = matriz[:, i].sum() - vp
        vn = matriz.sum() - vp - fn - fp
        por_classe[sigla] = {
            'nome':          NOMES_ORDENADOS[i],
            'precisao':      relatorio[NOMES_ORDENADOS[i]]['precision'],
            'sensibilidade': relatorio[NOMES_ORDENADOS[i]]['recall'],
            'especificidade': vn / (vn + fp),
            'f1':            relatorio[NOMES_ORDENADOS[i]]['f1-score'],
            'auc':           roc_auc_score(rotulos_teste == i, probs_teste[:, i]),
            'suporte':       int(matriz[i].sum()),
        }

    metricas = {
        'modelo': titulo, 'nome': nome, 'semente': semente,
        'acuracia':          acuracia_teste,
        'acuracia_balanceada': balanced_accuracy_score(rotulos_teste, predicoes_teste),
        'f1_macro':          f1_score(rotulos_teste, predicoes_teste, average='macro'),
        'auc_macro':         roc_auc_score(rotulos_teste, probs_teste, multi_class='ovr'),
        'melhor_acc_val':    melhor_acuracia,
        'acc_treino_final':  historico['acc_treino'][-1],
        'epocas_treinadas':  len(historico['perda_treino']),
        'melhor_epoca':      int(np.argmax(historico['acc_val'])) + 1,
        'tempo_medio_epoca_s': float(np.mean(historico['tempo_epoca'])),
        'tempo_inferencia_ms_por_exame': tempo_inferencia_ms,
        'parametros':        total_parametros,
        'taxa_aprendizado':  taxa_aprendizado,
        'n_treino': len(dados['treino'][1]), 'n_validacao': len(dados['validacao'][1]),
        'n_teste':  len(rotulos_teste),
        'assinatura_dados':  dados['assinatura'],
        'distribuicao': {p: {c: int(((dados['df']['particao'] == p) & (dados['df']['classe'] == i)).sum())
                             for i, c in enumerate(CLASSES_ORDENADAS)}
                         for p in ['treino', 'validacao', 'teste']},
        'pacientes': int(dados['df']['patient_id'].nunique()),
        'por_classe':        por_classe,
        'matriz_confusao':   matriz.tolist(),
        'configuracoes':     configuracoes,
    }
    with open(os.path.join(pasta_saida, 'metricas.json'), 'w', encoding='utf-8') as f:
        json.dump(metricas, f, ensure_ascii=False, indent=2, default=float)

    print(f"F1 macro: {metricas['f1_macro']:.4f} | AUC macro: {metricas['auc_macro']:.4f}")

    # ----- Etapa 10: matriz de confusão 7x7 (contagem + % da linha) -----
    porcentagem = matriz / matriz.sum(axis=1, keepdims=True)
    anotacoes   = np.array([[f"{matriz[i, j]}\n{porcentagem[i, j]:.0%}"
                             for j in rotulos_idx] for i in rotulos_idx])

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(porcentagem, annot=anotacoes, fmt='', cmap='Blues', ax=ax, vmin=0, vmax=1,
                xticklabels=NOMES_ORDENADOS, yticklabels=NOMES_ORDENADOS,
                linewidths=0.5, linecolor='gray', cbar_kws={'label': '% da classe real'})
    ax.set_xlabel('Diagnóstico previsto pelo modelo', fontsize=12, labelpad=10)
    ax.set_ylabel('Diagnóstico real (gabarito)', fontsize=12, labelpad=10)
    ax.set_title(f'Matriz de Confusão — {titulo}\nAcurácia {acuracia_teste:.1%} · '
                 f'F1 macro {metricas["f1_macro"]:.3f}', fontsize=14, fontweight='bold', pad=15)
    plt.xticks(rotation=35, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(pasta_saida, 'matriz_confusao.png'), dpi=200)
    plt.close(fig)

    # ----- Etapa 11: curvas de aprendizado -----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epocas_executadas = range(1, len(historico['perda_treino']) + 1)
    for ax, chave, rotulo_y, titulo_ax in [(ax1, 'perda', 'Perda', 'Curva de Perda (Loss)'),
                                           (ax2, 'acc',   'Acurácia', 'Curva de Acurácia')]:
        ax.plot(epocas_executadas, historico[f'{chave}_treino'], label='Treino',    color='#1565C0')
        ax.plot(epocas_executadas, historico[f'{chave}_val'],    label='Validação', color='#C62828')
        ax.axvline(metricas['melhor_epoca'], color='gray', ls='--', lw=1, label='Melhor época')
        ax.set_title(titulo_ax, fontweight='bold')
        ax.set_xlabel('Época')
        ax.set_ylabel(rotulo_y)
        ax.legend()
        ax.grid(alpha=0.3)
    plt.suptitle(f'Histórico de Treinamento — {titulo}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(pasta_saida, 'curvas_aprendizado.png'), dpi=200)
    plt.close(fig)

    # ----- Etapa 12: modelo final com configurações (usado pelo front end) -----
    torch.save({
        'pesos':          modelo.state_dict(),
        'arquitetura':    titulo,
        'configuracoes':  configuracoes,
        'classes':        CLASSES_ORDENADAS,
        'nomes_classes':  NOMES_ORDENADOS,
        'metricas_teste': {k: metricas[k] for k in ['acuracia', 'f1_macro', 'auc_macro']},
    }, os.path.join(pasta_saida, 'modelo_final.pt'))

    print(f"\nArquivos gerados em: {pasta_saida}")
    return metricas
