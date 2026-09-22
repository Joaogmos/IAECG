# ECG Transformer: Classificação Multiclasse de Arritmias Cardíacas 🫀🤖

Este repositório contém o código-fonte, a documentação e o artigo de investigação do projeto **ECG Transformer**, desenvolvido como Trabalho de Conclusão de Curso (TCC) e para a disciplina de Inteligência Artificial da Universidade Presbiteriana Mackenzie.

## 👥 Integrantes
* Enzo Ponte Gamberi - RA: 10389931
* João Guilherme Messias de Oliveira Santos - RA: 10426110
* Thiago Ruiz Fernandes Silva – RA: 10426057
* André Ihsan Ward - RA: 10425684

## 📌 Sobre o Projeto
As doenças cardiovasculares são uma das principais causas de mortalidade no mundo. O Eletrocardiograma (ECG) é o exame padrão para o diagnóstico destas patologias, mas a sua interpretação depende de médicos especialistas, frequentemente escassos em sistemas de telessaúde.

Este projeto propõe uma abordagem inovadora utilizando **Deep Learning**. Em vez das tradicionais Redes Neurais Convolucionais (CNNs), implementamos um modelo baseado no mecanismo de atenção **Vision Transformer (ViT)**, adaptado para o processamento de sinais unidimensionais (1D). O modelo é capaz de classificar exames de ECG em 7 categorias clínicas:
1. Normal
2. Bloqueio AV de 1º Grau (1dAVb)
3. Bloqueio de Ramo Direito (RBBB)
4. Bloqueio de Ramo Esquerdo (LBBB)
5. Bradicardia Sinusal (SB)
6. Taquicardia Sinusal (ST)
7. Fibrilação Atrial (AF)

## 📊 Dataset: CODE (Telehealth Network of Minas Gerais)
Os dados utilizados neste projeto pertencem ao dataset CODE, uma base de dados em larga escala contendo exames de ECG reais oriundos da rede de telessaúde de Minas Gerais. 

* **Formato dos Dados:** Os sinais brutos (12 derivações) estão armazenados em ficheiros `.hdf5`, e os metadados e os rótulos de diagnóstico encontram-se no ficheiro `exams.csv`.
* **Pré-processamento:** O pipeline do projeto realiza a extração dos sinais, normalização (`StandardScaler`) e um balanceamento estratificado rigoroso (2.000 amostras por classe) para lidar com o forte desbalanceamento natural da base de dados.
* *Nota:* Devido ao limite de armazenamento do GitHub, os ficheiros `.hdf5` originais não estão no repositório.

## ⚙️ Arquitetura do Modelo
O modelo `TransformerECG` foi construído do zero utilizando **PyTorch**. O fluxo de processamento consiste em:
1. **Patch Embedding:** Divisão do sinal 1D de 4096 amostras em *patches* menores usando convoluções.
2. **Positional Embedding & CLS Token:** Injeção da informação espacial/temporal no sinal.
3. **Transformer Encoder:** 4 camadas com 4 cabeças de atenção (*Multi-Head Attention*) para captar dependências globais e de longo alcance no batimento cardíaco.
4. **Classification Head:** Camada linear final que converte a representação para os logits das 7 classes.

## 📂 Organização do Repositório
* `main40.py` (ou ficheiro `.ipynb`): Script principal contendo o carregamento de dados, o modelo Transformer, a configuração de treino e a geração de gráficos.
* `Artigo_Parcial_ECG_Transformer.pdf`: Artigo científico documentando a contextualização, a metodologia e os resultados preliminares do projeto.
* `/resultados/`: Pasta onde o código salva automaticamente a matriz de confusão, as curvas de aprendizagem e os *checkpoints* do modelo (`melhor_modelo.pt`).

## 🚀 Como Executar
1. Instale as dependências necessárias:
   ```bash
   pip install torch pandas numpy h5py matplotlib seaborn scikit-learn
