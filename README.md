# ECG Transformer: Classificação Multiclasse de Arritmias Cardíacas

Este repositório contém o código-fonte, a documentação e o artigo de investigação do projeto **ECG Transformer**, desenvolvido como Trabalho de Conclusão de Curso (TCC) e para a disciplina de Inteligência Artificial da Universidade Presbiteriana Mackenzie.

## Integrantes
* Enzo Ponte Gamberi - RA: 10389931
* João Guilherme Messias de Oliveira Santos - RA: 10426110
* Thiago Ruiz Fernandes Silva – RA: 10426057
* André Ihsan Ward - RA: 10425684

## Sobre o Projeto
As doenças cardiovasculares são uma das principais causas de mortalidade no mundo. O Eletrocardiograma (ECG) é o exame padrão para o diagnóstico destas patologias, mas a sua interpretação depende de médicos especialistas, frequentemente escassos em sistemas de telessaúde.

Este projeto propõe uma abordagem inovadora utilizando **Deep Learning**. Como modelo principal, implementamos um modelo baseado no mecanismo de atenção **Vision Transformer (ViT)**, adaptado para o processamento de sinais unidimensionais (1D), comparado com uma CNN 1D (ResNet) como baseline. Os modelos classificam exames de ECG em 7 categorias clínicas:
1. Normal
2. Bloqueio AV de 1º Grau (1dAVb)
3. Bloqueio de Ramo Direito (RBBB)
4. Bloqueio de Ramo Esquerdo (LBBB)
5. Bradicardia Sinusal (SB)
6. Taquicardia Sinusal (ST)
7. Fibrilação Atrial (AF)

## Dataset: CODE (Telehealth Network of Minas Gerais)
Os dados utilizados neste projeto pertencem ao dataset CODE, uma base de dados em larga escala contendo exames de ECG reais oriundos da rede de telessaúde de Minas Gerais. 

* **Formato dos Dados:** Os sinais brutos (12 derivações) estão armazenados em ficheiros `.hdf5`, e os metadados e os rótulos de diagnóstico encontram-se no ficheiro `exams.csv`.
* **Pré-processamento:** O pipeline realiza a extração dos sinais, normalização z-score por derivação, balanceamento (até 2.000 exames por classe) e divisão 70/15/15 **por paciente**, para que nenhum paciente apareça em mais de um conjunto.
* *Nota:* Devido ao limite de armazenamento do GitHub, os ficheiros `.hdf5` originais não estão no repositório.

## Datasets
* CODE-15% (https://zenodo.org/records/4916206): treino e validação.
* CODE-test (https://zenodo.org/records/3765780): teste anotado por cardiologistas.
* CODE-II (https://arxiv.org/abs/2511.15632): referência.

Baixe do Drive do grupo e coloque `exams.csv` e os `.hdf5` em `dados/` (pasta ignorada pelo Git).

## Modelos
### Transformer (`transformer_ecg.py`, principal)
O modelo `TransformerECG` foi construído do zero utilizando **PyTorch**. O fluxo de processamento consiste em:
1. **Patch Embedding:** Divisão do sinal 1D de 4096 amostras em *patches* menores usando convoluções.
2. **Positional Embedding & CLS Token:** Injeção da informação espacial/temporal no sinal.
3. **Transformer Encoder:** 4 camadas com 4 cabeças de atenção (*Multi-Head Attention*) para captar dependências globais e de longo alcance no batimento cardíaco.
4. **Classification Head:** Camada linear final que converte a representação para os logits das 7 classes.

Variante híbrida (`python transformer_ecg.py --hibrido`): 3 convoluções extraem a morfologia local (tokens de 40 ms) antes das camadas de atenção.

### ResNet 1D (`cnn1d_ecg.py`, baseline)
Baseado em Ribeiro et al. (2020). Resultados atualizados em `docs/tabelas/tabelas.md` e gráficos em `figuras/cnn1d/`.

## Resultados (conjunto de teste, média ± desvio de 3 sementes)
| Modelo | Acurácia | F1 macro | AUC macro | Parâmetros |
|---|---|---|---|---|
| Transformer 1D | 87,9 ± 0,8% | 87,7 ± 0,8% | 0,983 | 0,9 M |
| Transformer híbrido | 89,6 ± 0,6% | 89,5 ± 0,6% | 0,986 | 1,1 M |
| CNN 1D ResNet | 93,4 ± 0,4% | 93,3 ± 0,4% | 0,989 | 6,8 M |

## Organização do Repositório
* `comum_ecg.py`: dados, divisão por paciente, treino e avaliação (compartilhado pelos dois modelos).
* `transformer_ecg.py`, `cnn1d_ecg.py`: arquiteturas e scripts de treino.
* `gerar_tabelas.py`: tabelas de resultados (Markdown/CSV) e figuras para o pôster.
* `figuras_relatorio.py`: figuras de ECG, distribuição das classes e fluxo do projeto.
* `dados/`: dados locais (fora do Git). `resultados/`: saídas de treino (fora do Git).
* `figuras/`: gráficos finais para pôster e relatório.
* `docs/`: guias da equipe, registro de experimentos, relatório e pôster.
* `CLAUDE.md`: contexto do projeto para o Claude Code.

## Como Executar
```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # GPU NVIDIA
pip install -r requirements.txt
python cnn1d_ecg.py          # baseline
python transformer_ecg.py    # modelo principal
python gerar_tabelas.py      # tabelas em docs/tabelas/tabelas.md
python figuras_relatorio.py  # figuras em figuras/relatorio/
```
Para repetir com outra semente: `python transformer_ecg.py --semente 7`.
