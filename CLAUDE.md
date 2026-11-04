# CLAUDE.md — IAECG

TCC (Mackenzie). Info pessoal do grupo fica em `CLAUDE.local.md` (fora do Git).

Projeto: IA que lê ECG de 12 derivações e aponta 6 alterações (1dAVb, RBBB, LBBB, SB, ST, AF)
+ "ECG normal". PyTorch. TCC 1 entregue (protocolo v2, na `main`); TCC 2 na branch `tcc2`
(protocolo v3, API e interface).

## Código
- `comum_ecg.py`: tudo que é compartilhado — rótulos, divisão por paciente, cache memmap,
  augmentation, loop de treino, early stopping, limiares, métricas, CODE-test e figuras.
  `avaliacao_final()` é separada do treino (usada por `reavaliar_modelo.py`).
- `transformer_ecg.py`: Transformer 1D (patch 64 + CLS + 4 camadas, 4 cabeças).
  `--hibrido`: stem de 3 convs (tokens de 40 ms) antes da atenção.
- `cnn1d_ecg.py`: ResNet 1D (Ribeiro et al. 2020). Melhor modelo; é o do produto.
- `inferencia_ecg.py`: leitura de formatos (hdf5/npy/csv/WFDB), reamostragem, Einthoven,
  qualidade, FC/RR, saliência. `api_ecg.py` (FastAPI) serve isso e `frontend/`.
  `triagem_lote.py`: pasta ou HDF5 inteiro → fila em CSV (repetir, alterado, outra, normal).
- `baixar_dados.py`, `treinar_todos.py`, `reavaliar_modelo.py`, `gerar_tabelas.py`, `figuras_relatorio.py`.
- `testes/`: pytest com sinais sintéticos.
Os scripts de modelo só definem a arquitetura e a LR; mudanças de treino vão em `comum_ecg.py`
para valer para todos.

## Decisões de método (v3, manter)
- Multirrótulo: 7 sigmoides (6 doenças + `normal_ecg`), BCE. Todos os exames, prevalência real,
  nada descartado. "Outra alteração" = nenhuma das 6 e `normal_ecg` False.
- Divisão 70/15/15 por `patient_id` (StratifiedGroupKFold pela doença mais rara, semente 42 fixa).
  Nunca dividir por exame.
- Sem inversão de polaridade no augmentation (troca a morfologia RBBB/LBBB).
- Seleção da época pela AP macro de validação. Limiares por classe só na validação
  (equilibrado = F1 máx.; triagem = sensibilidade ≥ 95%, nunca acima do equilibrado).
- Métricas finais no teste interno e no CODE-test. CODE-test grava as aumentadas como
  aVL, aVF, aVR: reordenar com `ORDEM_CODE_TEST`.
- v2 (TCC 1): balanceado, 7 classes exclusivas, seleção por acurácia. Só na `main`.

## Resultados
- v2 (19/10/2026, 3 sementes, teste): CNN 93,4 ± 0,4% (F1 0,933) > híbrido 89,6 ± 0,6% > Transformer 87,9 ± 0,8%.
- v3: em andamento; ver `docs/tabelas/v3/tabelas.md` e `docs/experimentos.md`.

## Dados (NUNCA no Git)
- CODE-15% (zenodo 4916206): `exams.csv` + 18 `exams_partN.hdf5` (~3,9 GB cada). `baixar_dados.py` baixa.
- CODE-test (zenodo 3765780) em `dados/code_test/data/`; CODE-II (arXiv 2511.15632) como referência.
- `dados/cache_sinais_v3_*.npy` (float16, ~34 GB na base inteira) é gerado no 1º treino.
- Caminhos sempre relativos ao script (`os.path.join`), nunca `C:\Users\...`.

## Estrutura
```
dados/            (ignorado) exams.csv, *.hdf5, code_test/, cache .npy
resultados/       (ignorado) v3/<modelo>_s<semente>/; v2 em <modelo>_s<semente>/; teste_rapido/
docs/tabelas/v3/  tabelas.md e .csv geradas por gerar_tabelas.py
figuras/          gráficos finais (v3 em figuras/v3/), versionados
docs/             (ignorado) guias da equipe, experimentos, tabelas; compartilhados à parte
frontend/         interface (base feita; André continua). Contrato da API em frontend/README.md
testes/           pytest
```

## Convenções
- Código e comentários em português.
- Não commitar dados, `.pt`, `.zip`, `.npy`, `.venv`.
- Cada experimento: anotar config e métricas em `docs/experimentos.md`.
- Comparar modelos no mesmo split (assinatura de dados igual no `metricas.json`).
