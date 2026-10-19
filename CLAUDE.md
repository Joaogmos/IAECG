# CLAUDE.md — IAECG

TCC (Mackenzie). Info pessoal do grupo fica em `CLAUDE.local.md` (fora do Git).

Projeto: IA que classifica ECG de 12 derivações em 7 classes:
Normal, 1dAVb, RBBB, LBBB, SB, ST, AF. PyTorch.

## Código
- `comum_ecg.py`: tudo que é compartilhado — rótulos, balanceamento, divisão por paciente,
  cache de sinais, augmentation, loop de treino, early stopping, métricas e figuras.
- `transformer_ecg.py`: Transformer 1D (patch 64 + CLS + 4 camadas, 4 cabeças). Modelo principal.
  `--hibrido`: stem de 3 convs (tokens de 40 ms) antes da atenção.
- `cnn1d_ecg.py`: ResNet 1D (Ribeiro et al. 2020). Baseline de comparação.
- `gerar_tabelas.py`: junta `resultados/*/metricas.json` em tabelas (md/csv) e copia figuras para `figuras/`.
- `figuras_relatorio.py`: figuras de ECG, distribuição de classes e fluxo do projeto.
Os scripts de modelo só definem a arquitetura e a LR; mudanças de treino vão em `comum_ecg.py`
para valer para os dois.

## Decisões de método (manter)
- Normal = `normal_ecg` True e nenhuma das 6 doenças. Exames com 2+ doenças ou com outra
  alteração são descartados.
- Divisão 70/15/15 por `patient_id` (StratifiedGroupKFold, semente 42 fixa). Nunca dividir por exame.
- Sem inversão de polaridade no augmentation (troca a morfologia RBBB/LBBB).
- Seleção do modelo pela acurácia de validação; métricas finais só no teste.

## Resultados atuais (v2, 19/10/2026, 3 sementes, teste)
CNN 93,4 ± 0,4% (F1 0,933) > Transformer híbrido 89,6 ± 0,6% > Transformer 87,9 ± 0,8%.
Histórico completo em `docs/experimentos.md`.

## Dados (NUNCA no Git)
- CODE-15% (zenodo 4916206): `exams.csv` + 18 arquivos `exams_partN.hdf5` (~3,9 GB cada).
  Temos part0–part6 no OneDrive do grupo (`IAECG/dados`).
- CODE-test (zenodo 3765780) para teste externo; CODE-II (arXiv 2511.15632) como referência.
- Tudo em `dados/` na raiz do repo. `dados/cache_sinais_*.npy` é gerado no 1º treino.
- Caminhos sempre relativos ao script (`os.path.join`), nunca `C:\Users\...`.

## Estrutura
```
dados/            (ignorado) exams.csv, *.hdf5, cache .npy
resultados/       (ignorado) <modelo>_s<semente>/: metricas.json, .pt, png
docs/tabelas/     tabelas.md e .csv geradas por gerar_tabelas.py
figuras/          gráficos finais para pôster/relatório (versionados)
docs/             (ignorado) guias da equipe, experimentos, tabelas; compartilhados à parte
frontend/         interface (André)
```

## Convenções
- Código e comentários em português.
- Não commitar dados, `.pt`, `.zip`, `.npy`, `.venv`.
- Cada experimento: anotar config e métricas em `docs/experimentos.md`.
- Comparar Transformer x CNN no mesmo split (assinatura de dados igual no `metricas.json`).
