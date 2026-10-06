# CLAUDE.md — IAECG

TCC (Mackenzie). Info pessoal do grupo fica em `CLAUDE.local.md` (fora do Git).

Projeto: IA que classifica ECG de 12 derivações em 7 classes:
Normal, 1dAVb, RBBB, LBBB, SB, ST, AF. PyTorch.

## Modelos
- `transformer_ecg.py`: Transformer 1D (patch embedding + CLS + 4 camadas, 4 cabeças). Modelo principal.
- `cnn1d_ecg.py`: ResNet 1D (Ribeiro et al. 2020). Baseline de comparação.
Os dois: lêem `exams.csv` + `.hdf5`, balanceiam por classe, normalizam, treinam e salvam
matriz de confusão, curvas e checkpoints.

## Dados (NUNCA no Git)
- CODE-15% (zenodo 4916206): `exams.csv` + ~20 arquivos `.hdf5` de 3 GB+.
- CODE-test (zenodo 3765780) para teste; CODE-II (arXiv 2511.15632) como referência.
- Cada membro baixa do Drive do grupo e coloca em `dados/` na raiz do repo.
- Caminhos sempre relativos ao script (`os.path.join`), nunca `C:\Users\...`.

## Estrutura
```
dados/            (ignorado) exams.csv, *.hdf5
resultados/       (ignorado) checkpoints .pt, saídas de treino
figuras/          gráficos finais escolhidos para pôster/relatório (versionados)
docs/             relatório, pôster, artigos
frontend/         interface (André)
```

## Convenções
- Código e comentários em português.
- Não commitar dados, `.pt`, `.zip`, `.venv`.
- Cada experimento: anotar config e métricas (acc, F1 macro por classe) em `docs/experimentos.md`.
- Comparar Transformer x CNN no mesmo split.

## Problemas conhecidos
- `transformer_ecg.py` usa caminho fixo do PC do Thiago e separadores `\\`; alinhar com `cnn1d_ecg.py`.
- CNN com overfitting forte (treino ~100%, validação ~82%): testar dropout, data augmentation, early stopping, mais dados por classe.
