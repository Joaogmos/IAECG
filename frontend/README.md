# Interface IAECG (frontend)

Base da interface. Ficou em HTML, CSS e JavaScript puros, sem build, para dar para mexer direto.
O André assume daqui em diante, e pode trocar por React ou Vite se fizer sentido.

## Como rodar

```bash
python api_ecg.py          # http://127.0.0.1:8000
```

A API serve esta pasta em `/` e o modelo em `/api`. Ela usa o melhor `resultados/v3/*/modelo_final.pt`;
para escolher outro, defina `IAECG_MODELO=caminho/do/modelo_final.pt`.
Os exames de demonstração vêm do CODE-test (`python baixar_dados.py --so-code-test`).

## Arquivos

| Arquivo | O que tem |
|---|---|
| `index.html` | estrutura: topo (modo), coluna de entrada (arquivo, fila de vários exames, exemplos), leitura (fila ou papel + painéis), rodapé |
| `estilo.css` | tokens de cor e tipo em `:root`; layout em grid; quebra para 1 coluna abaixo de 980 px |
| `app.js` | chamadas à API, fila de leitura (tabela, abrir exame, baixar CSV), desenho do papel de ECG em `<canvas>`, painéis de resultado |

## Identidade visual

- O traçado aparece como na impressão do eletrocardiógrafo: papel com grade de 1 mm e 5 mm, a
  25 mm/s e 10 mm/mV, layout 3×4 (2,5 s por derivação) mais a tira de ritmo em DII, e o pulso de
  calibração de 1 mV no começo de cada linha.
- A marcação amarela, como um marca-texto, mostra os ~12% de instantes em que o modelo mais se
  apoiou para a alteração escolhida (gradiente × entrada). Como cada coluna do papel é o mesmo
  instante para todas as derivações, a marcação ocupa a altura inteira.
- Tipos: Barlow Semi Condensed (títulos), IBM Plex Sans (texto), IBM Plex Mono (números e cabeçalho).
- Cores do veredito: vermelho = alteração, verde = normal, âmbar = outra alteração ou sinal ruim.

## Contrato da API

`GET /api/modelo` devolve arquitetura, protocolo, saídas, limiares (`equilibrado` e `triagem`),
métricas do teste interno, do CODE-test e do PTB-XL, e `desempenho_por_classe`
(`{sigla: {modo: {sensibilidade, especificidade, precisao}}}`, do teste interno), que a interface
mostra embaixo de cada barra.

`GET /api/exemplos` devolve `[{indice, idade, sexo, laudo: ['RBBB'], laudo_nomes: [...]}]`.

`GET /api/exemplos/{indice}?modo=equilibrado` e `POST /api/analisar` (multipart: `arquivos` (1 ou o
par .hea/.dat), `frequencia`, `indice`, `modo`) devolvem:

```jsonc
{
  "veredito": "alterado" | "normal" | "outra_alteracao",
  "modo": "equilibrado",
  "achados": ["RBBB"],
  "resultados": [{"sigla": "RBBB", "nome": "...", "probabilidade": 0.97, "limiar": 0.31, "positivo": true}, ...],
  "qualidade": {"adequado": true, "duracao_s": 10.0, "alertas": [{"gravidade": "alta|media|baixa", "mensagem": "..."}]},
  "ritmo": {"frequencia_bpm": 72, "rr_medio_ms": 830, "irregularidade_rr": 0.03, "batimentos": 12, "picos_r": [0.41, ...]},
  "saliencia": {"RBBB": [0..1 a 200 Hz, 2048 valores]},
  "sinal": {"frequencia_hz": 200, "derivacoes": ["DI", ..., "V6"], "valores": [[...2048], ...12]},
  "modelo": {"arquitetura": "CNN 1D (ResNet)", "protocolo": "v3"},
  "exemplo": {...}            // só nos exemplos do CODE-test
}
```

`POST /api/lote` (multipart: `arquivos` (vários; o .hea e o .dat do WFDB juntos, um HDF5 vale por
todos os seus exames), `frequencia`, `modo`, `formato` = `json` ou `csv`) analisa até 2.000 exames
sem saliência e devolve a fila na ordem de leitura:

```jsonc
{
  "modo": "equilibrado", "total": 827,
  "contagem": {"alterado": 132, "outra_alteracao": 240, "normal": 455},
  "fila": [{"posicao": 1, "exame": "ecg_tracings.hdf5#330", "veredito": "repetir" | "alterado" |
            "outra_alteracao" | "normal" | "erro", "achados": "RBBB", "maior_prob": 0.98,
            "p_1dAVb": 0.0001, ..., "p_Normal": 0.0001, "fc_bpm": 86.8, "observacao": "..."}, ...]
}
```

Com `formato=csv` vem o mesmo conteúdo em CSV (`;`, UTF-8 com BOM), pronto para o Excel.

Erros vêm com status 4xx e `{"detail": "mensagem em português"}`, que a interface mostra como está.

## Próximos passos sugeridos

- Tela de "como ler o resultado" para o médico (o que é limiar, o que é o modo triagem).
- Zoom e rolagem no papel; régua para medir intervalos (PR, QRS) clicando em dois pontos.
- Exportar o resultado em PDF para anexar ao prontuário (com o aviso de ferramenta de pesquisa).
- Modo escuro, que hoje não existe: o papel de ECG claro foi escolhido de propósito, mas o resto da tela pode escurecer.
- Testar com leitor de tela; as barras de probabilidade já têm `aria-label`.
