# IAECG: classificação de alterações em ECG de 12 derivações

Código do Trabalho de Conclusão de Curso da Universidade Presbiteriana Mackenzie (FCI). O projeto treina
redes neurais que leem o ECG de 12 derivações e apontam seis alterações: bloqueio AV de 1º grau (1dAVb),
bloqueio de ramo direito (RBBB) e esquerdo (LBBB), bradicardia sinusal (SB), taquicardia sinusal (ST) e
fibrilação atrial (AF). No TCC 2 o modelo ganhou uma API e uma interface web para apoio à triagem.

## Integrantes
* Enzo Ponte Gamberi - RA: 10389931
* João Guilherme Messias de Oliveira Santos - RA: 10426110
* Thiago Ruiz Fernandes Silva - RA: 10426057
* André Ihsan Ward - RA: 10425684

Orientador: Prof. Dr. Gustavo Scalabrini Sampaio.

## Dados
* CODE-15% (https://zenodo.org/records/4916206): 345.779 exames de 233.770 pacientes, usado para treino, validação e teste interno.
* CODE-test (https://zenodo.org/records/3765780): 827 exames de outros pacientes, laudados por cardiologistas. É o teste externo.
* PTB-XL (https://physionet.org/content/ptb-xl/1.0.3/): 21.799 ECGs da Alemanha, outro equipamento e outra
  população. Segundo teste externo (`python baixar_dados.py --so-ptbxl`).
* CODE-II (https://arxiv.org/abs/2511.15632): referência.

`python baixar_dados.py` baixa o que faltar direto do Zenodo para `dados/` (pasta fora do Git), confere o MD5 e
retoma downloads interrompidos. São cerca de 70 GB descompactados. Quem já tem parte dos `.hdf5` (Drive do grupo)
pode copiá-los antes para `dados/`, e o script baixa só o resto.

## Protocolos

**v2 (TCC 1, na branch `main`).** Subconjunto balanceado de 13.613 exames, 7 classes exclusivas (softmax) e
divisão por paciente. Resultado no teste, com 3 sementes: CNN 93,4 ± 0,4% de acurácia (F1 0,933), Transformer
híbrido 89,6 ± 0,6% e Transformer 87,9 ± 0,8%.

**v3 (TCC 2, branch `tcc2`).** Feito para ficar perto do uso real:
* usa todos os exames do CODE-15% com a prevalência natural (cerca de 2% por doença), sem balancear e sem descartar
  exames com outras alterações;
* é multirrótulo: 6 sigmoides para as doenças e uma 7ª para "ECG normal". Um exame pode ter duas doenças, e um
  exame sem nenhuma das seis pode ter outra alteração;
* divide 70/15/15 por paciente, estratificando pela doença mais rara de cada exame;
* escolhe dois limiares por classe na validação: "equilibrado" (F1 máximo, como em Ribeiro et al., 2020) e
  "triagem" (sensibilidade de pelo menos 95%);
* seleciona a época pela AP macro de validação (área sob a curva precisão-sensibilidade, que não depende de limiar);
* avalia no teste interno, no CODE-test (onde compara o F1 com residentes de cardiologia, residentes de
  emergência, estudantes de medicina e a DNN de Ribeiro et al., 2020) e no PTB-XL;
* mede a calibração (ECE), porque a interface mostra a probabilidade ao médico.

No CODE-test as derivações aumentadas estão gravadas como aVL, aVF, aVR, e não aVR, aVL, aVF como no CODE-15%.
Conferimos pela lei de Einthoven (aVR = -(DI+DII)/2) e o código reordena na leitura.

## Modelos
* `cnn1d_ecg.py`: ResNet 1D de Ribeiro et al. (2020), 6,8 M de parâmetros. Melhor modelo no TCC 1.
* `transformer_ecg.py`: Transformer 1D com patches de 64 amostras (160 ms), token CLS, 4 camadas e 4 cabeças.
  Com `--hibrido`, 3 convoluções geram tokens de 40 ms antes da atenção.

Treino, dados e avaliação ficam em `comum_ecg.py`, iguais para todos os modelos.

## Inferência, API e interface
* `inferencia_ecg.py` lê `.hdf5` (formato CODE), `.npy`, `.csv`/`.txt` com 12 colunas e WFDB (`.hea` + `.dat`).
  Reordena as derivações pelo nome, reamostra para 400 Hz, tira a linha de base (passa-alta de 0,5 Hz), confere a
  ordem das aumentadas por Einthoven, avisa
  sobre derivação solta, saturação e registro curto, mede FC e irregularidade do RR e calcula a saliência
  (onde no traçado o modelo se apoiou).
* `api_ecg.py` (FastAPI) serve o modelo em `/api` e a interface de `frontend/` em `/`. O exame é processado em
  memória e não fica gravado. O contrato da API está em `frontend/README.md`.

A ferramenta é de pesquisa. Não é dispositivo médico registrado na ANVISA e não substitui o laudo do cardiologista.

## Como executar
```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # GPU NVIDIA
pip install -r requirements.txt

python baixar_dados.py                  # CODE-15% e CODE-test em dados/
python treinar_todos.py                 # 3 modelos × 3 sementes (v3); retoma se parar
python gerar_tabelas.py                 # docs/tabelas/v3/tabelas.md e figuras/v3/
python api_ecg.py                       # interface em http://127.0.0.1:8000
python -m pytest testes                 # testes do pipeline e da inferência
```

Um treino isolado: `python cnn1d_ecg.py --semente 7`. Ensemble das sementes: `python ensemble_ecg.py cnn1d`.
Busca de hiperparâmetros do Transformer (48 combinações num subconjunto): `python busca_transformer.py`. Teste rápido do pipeline com poucos exames:
`python cnn1d_ecg.py --limite 6000` (sai em `resultados/teste_rapido/`). Para refazer só a avaliação de um treino
pronto: `python reavaliar_modelo.py resultados/v3/cnn1d_s42`. Na linha de comando:
`python inferencia_ecg.py exame.csv --frequencia 500`. Para uma fila de laudos,
`python triagem_lote.py pasta_de_exames/ --saida fila.csv` analisa todos os arquivos da pasta (ou todos os
exames de um HDF5) e grava um CSV com os de sinal ruim primeiro (repetir), depois os alterados pela maior
probabilidade, os de outra alteração e os normais. O CODE-test inteiro (827 exames) leva uns 12 s na CPU.
`python robustez_ecg.py [--filtro]` mede no CODE-test quanto o modelo perde com ruído, linha de base, rede
elétrica, eletrodo solto, amostragem menor e registro curto (`robustez.md` na pasta do modelo).

Na base inteira o cache de sinais (`dados/cache_sinais_v3_*.npy`, float16) ocupa cerca de 34 GB e é lido por
memmap, então não precisa caber na RAM. Numa RTX 4060, cada época da ResNet leva uns 5 minutos.

## Organização
* `comum_ecg.py`: rótulos, divisão por paciente, cache, augmentation, treino, limiares, métricas e figuras.
* `cnn1d_ecg.py`, `transformer_ecg.py`: arquiteturas.
* `inferencia_ecg.py`, `api_ecg.py`, `frontend/`, `triagem_lote.py`: uso do modelo fora do treino.
* `robustez_ecg.py`: teste de robustez com defeitos comuns do traçado.
* `baixar_dados.py`, `treinar_todos.py`, `ensemble_ecg.py`, `busca_transformer.py`, `reavaliar_modelo.py`,
  `gerar_tabelas.py`, `figuras_relatorio.py`: utilitários.
* `testes/`: testes com sinais sintéticos (não precisam dos dados).
* `dados/`, `resultados/`, `docs/`: fora do Git. `figuras/`: gráficos finais, versionados.
