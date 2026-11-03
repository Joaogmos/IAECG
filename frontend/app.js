// IAECG — interface base. Conversa com api_ecg.py (mesma origem).
// Estrutura: estado → chamadas à API → desenho do papel de ECG → painéis.

const estado = {
  modo: 'equilibrado',
  ultimaAnalise: null,       // função que refaz a última análise (ao trocar o modo)
  resultado: null,
  salienciaAtiva: null,      // sigla cuja saliência está desenhada
};

const AJUDA_MODO = {
  equilibrado: 'Limiar com o melhor equilíbrio entre encontrar doentes e evitar alertas falsos.',
  triagem: 'Limiar que encontra pelo menos 95% dos doentes da validação, ao custo de mais alertas falsos.',
};

const $ = (id) => document.getElementById(id);
const pct = (v) => (v * 100).toLocaleString('pt-BR', { minimumFractionDigits: 1, maximumFractionDigits: 1 }) + '%';
const pct0 = (v) => Math.round(v * 100) + '%';
const num = (v, casas = 0) => v.toLocaleString('pt-BR', { minimumFractionDigits: casas, maximumFractionDigits: casas });

// ---------------------------------------------------------------------------
// API
// ---------------------------------------------------------------------------

async function chamar(url, opcoes) {
  const resposta = await fetch(url, opcoes);
  const corpo = await resposta.json().catch(() => ({}));
  if (!resposta.ok) throw new Error(corpo.detail || `Erro ${resposta.status} ao falar com o servidor`);
  return corpo;
}

async function analisar(executar) {
  estado.ultimaAnalise = executar;
  $('erro').hidden = true;
  document.querySelector('.leitura').classList.add('carregando');
  try {
    mostrarResultado(await executar());
  } catch (erro) {
    $('erro').textContent = erro.message;
    $('erro').hidden = false;
  } finally {
    document.querySelector('.leitura').classList.remove('carregando');
  }
}

function analisarArquivos(arquivos) {
  if (!arquivos.length) return;
  marcarExemplo(null);
  analisar(() => {
    const dados = new FormData();
    for (const arquivo of arquivos) dados.append('arquivos', arquivo);
    dados.append('frequencia', $('frequencia').value || '400');
    dados.append('indice', $('indice').value || '0');
    dados.append('modo', estado.modo);
    return chamar('/api/analisar', { method: 'POST', body: dados });
  });
}

function analisarExemplo(indice) {
  marcarExemplo(indice);
  analisar(() => chamar(`/api/exemplos/${indice}?modo=${estado.modo}`));
}

// ---------------------------------------------------------------------------
// Entrada: arquivo, exemplos e modo
// ---------------------------------------------------------------------------

function configurarEntrada() {
  const zona = $('soltar');
  $('arquivo').addEventListener('change', (e) => analisarArquivos(e.target.files));
  zona.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); $('arquivo').click(); }
  });
  zona.addEventListener('dragover', (e) => { e.preventDefault(); zona.classList.add('arrastando'); });
  zona.addEventListener('dragleave', () => zona.classList.remove('arrastando'));
  zona.addEventListener('drop', (e) => {
    e.preventDefault();
    zona.classList.remove('arrastando');
    analisarArquivos(e.dataTransfer.files);
  });

  document.querySelectorAll('.controle-modo button').forEach((botao) => {
    botao.addEventListener('click', () => {
      estado.modo = botao.dataset.modo;
      document.querySelectorAll('.controle-modo button').forEach((b) =>
        b.setAttribute('aria-checked', String(b === botao)));
      $('modo-ajuda').textContent = AJUDA_MODO[estado.modo];
      if (estado.ultimaAnalise) analisar(estado.ultimaAnalise);
    });
  });
}

async function carregarExemplos() {
  const lista = $('exemplos');
  try {
    const exemplos = await chamar('/api/exemplos');
    lista.innerHTML = '';
    if (!exemplos.length) {
      lista.innerHTML = '<li class="nota">CODE-test não encontrado em dados/. Rode baixar_dados.py --so-code-test.</li>';
      return;
    }
    for (const ex of exemplos) {
      const item = document.createElement('li');
      item.innerHTML = `<button type="button" data-indice="${ex.indice}">
          <span class="exemplo-laudo">${ex.laudo_nomes.join(' + ')}</span>
          <span class="exemplo-num">nº ${ex.indice}</span>
          <span class="exemplo-paciente">${ex.sexo === 'M' ? 'Homem' : 'Mulher'}, ${ex.idade} anos</span>
        </button>`;
      item.querySelector('button').addEventListener('click', () => analisarExemplo(ex.indice));
      lista.appendChild(item);
    }
  } catch (erro) {
    lista.innerHTML = `<li class="nota">${erro.message}</li>`;
  }
}

function marcarExemplo(indice) {
  document.querySelectorAll('.exemplos button').forEach((b) =>
    b.setAttribute('aria-current', String(Number(b.dataset.indice) === indice)));
}

async function carregarInfoModelo() {
  try {
    const info = await chamar('/api/modelo');
    estado.desempenho = info.desempenho_por_classe || null;
    if (estado.resultado) mostrarSaidas(estado.resultado);
    const ct = info.metricas_code_test;
    $('modelo-info').textContent = `Modelo: ${info.arquitetura} · protocolo ${info.protocolo}`
      + ` · teste interno: AUC ${num(info.metricas_teste.auc_macro, 3)}, F1 ${num(info.metricas_teste.f1_macro, 3)}`
      + (ct ? ` · CODE-test: AUC ${num(ct.auc_macro, 3)}, F1 ${num(ct.f1_macro, 3)}` : '');
  } catch { /* o rodapé fica sem a linha do modelo */ }
}

// ---------------------------------------------------------------------------
// Resultado
// ---------------------------------------------------------------------------

const VEREDITOS = {
  alterado: (r) => ({
    titulo: `Alteração detectada: ${r.resultados.filter((s) => s.positivo && s.sigla !== 'Normal').map((s) => s.nome).join(', ')}`,
    texto: 'Priorize este exame para o laudo do cardiologista.',
  }),
  normal: () => ({
    titulo: 'Sem as seis alterações e com padrão normal',
    texto: 'O laudo do cardiologista continua necessário; o modelo só não encontrou motivo para prioridade.',
  }),
  outra_alteracao: () => ({
    titulo: 'Nenhuma das seis alterações, mas o traçado não parece normal',
    texto: 'Pode haver outra alteração que este modelo não classifica. Encaminhe para revisão do cardiologista.',
  }),
};

function mostrarResultado(r) {
  estado.resultado = r;
  $('vazio').hidden = true;
  $('resultado').hidden = false;

  mostrarImpressao(r);
  mostrarVeredito(r);
  mostrarSaidas(r);
  mostrarQualidade(r);
  mostrarControleSaliencia(r);
  desenharPapel(true);
}

function mostrarImpressao(r) {
  const ritmo = r.ritmo;
  const partes = [
    ['FC', ritmo ? `${num(ritmo.frequencia_bpm)} bpm` : '—'],
    ['RR médio', ritmo ? `${num(ritmo.rr_medio_ms)} ms` : '—'],
    ['Irregularidade RR', ritmo ? num(ritmo.irregularidade_rr, 3) : '—'],
    ['Duração', `${num(r.qualidade.duracao_s, 1)} s`],
    ['', '25 mm/s · 10 mm/mV'],
  ];
  if (r.exemplo) partes.unshift(['CODE-test', `nº ${r.exemplo.indice} · ${r.exemplo.sexo}, ${r.exemplo.idade} anos`]);
  $('impressao').innerHTML = partes
    .map(([rotulo, valor]) => `<span>${rotulo ? `<span class="rotulo">${rotulo}</span> ` : ''}<b>${valor}</b></span>`)
    .join('');
}

function mostrarVeredito(r) {
  const caixa = $('veredito');
  let { titulo, texto } = VEREDITOS[r.veredito](r);
  let tipo = r.veredito;
  if (!r.qualidade.adequado) {
    tipo = 'sinal';
    texto = 'O sinal tem problemas de qualidade (veja abaixo). Repita o exame antes de confiar nesta leitura. ' + texto;
  }
  caixa.dataset.tipo = tipo;
  caixa.innerHTML = `<p class="veredito-titulo">${titulo}</p><p>${texto}</p>`
    + (r.exemplo ? `<p class="laudo-ref">Laudo dos cardiologistas: <b>${r.exemplo.laudo_nomes.join(' + ')}</b></p>` : '');
}

function mostrarSaidas(r) {
  $('saidas').innerHTML = r.resultados.map((s) => {
    // Desempenho desta classe no teste interno (prevalência real), no modo escolhido
    const d = estado.desempenho && estado.desempenho[s.sigla] && estado.desempenho[s.sigla][r.modo];
    const desempenho = d
      ? `<span class="saida-desempenho">no teste: encontra ${pct0(d.sensibilidade)} dos casos · `
        + `${pct0(d.precisao)} dos alertas confirmados</span>`
      : '';
    return `
    <li class="saida ${s.positivo ? 'positivo' : ''} ${s.sigla === 'Normal' ? 'normal' : ''}">
      <span class="saida-nome">${s.nome}</span>
      <span class="barra" role="img" aria-label="${s.nome}: ${pct(s.probabilidade)}, limiar ${pct(s.limiar)}">
        <span class="barra-valor" style="width:${(s.probabilidade * 100).toFixed(1)}%"></span>
        <span class="barra-limiar" style="left:calc(${(s.limiar * 100).toFixed(1)}% - 1px)"></span>
      </span>
      <span class="saida-pct">${pct(s.probabilidade)}</span>
      ${desempenho}
    </li>`;
  }).join('');
}

function mostrarQualidade(r) {
  const ritmo = r.ritmo;
  const medidas = ritmo ? `<dl class="medidas">
      <dt>Batimentos detectados</dt><dd>${ritmo.batimentos}</dd>
      <dt>Frequência cardíaca</dt><dd>${num(ritmo.frequencia_bpm)} bpm</dd>
      <dt>Irregularidade do RR</dt><dd>${num(ritmo.irregularidade_rr, 3)}</dd>
    </dl>` : '<p class="nota">Não foi possível detectar os batimentos em DII.</p>';
  const alertas = r.qualidade.alertas.length
    ? `<ul>${r.qualidade.alertas.map((a) => `<li data-gravidade="${a.gravidade}">${a.mensagem}</li>`).join('')}</ul>`
    : '<p>Nenhum problema de qualidade encontrado.</p>';
  $('qualidade').innerHTML = `<h3>Medidas e qualidade do sinal</h3>${medidas}${alertas}
    <p class="nota">As medidas vêm de um detector de picos R, não do modelo. Servem para conferir a leitura.</p>`;
}

function mostrarControleSaliencia(r) {
  const siglas = Object.keys(r.saliencia);
  estado.salienciaAtiva = siglas[0] || null;
  const nomes = Object.fromEntries(r.resultados.map((s) => [s.sigla, s.nome]));
  const caixa = $('saliencia-controle');
  caixa.innerHTML = '<span>Marcar onde o modelo se apoiou para:</span>'
    + siglas.map((s) => `<button type="button" data-sigla="${s}" aria-pressed="${s === estado.salienciaAtiva}">${nomes[s]}</button>`).join('')
    + '<button type="button" data-sigla="" aria-pressed="false">Sem marcação</button>';
  caixa.querySelectorAll('button').forEach((botao) => botao.addEventListener('click', () => {
    estado.salienciaAtiva = botao.dataset.sigla || null;
    caixa.querySelectorAll('button').forEach((b) =>
      b.setAttribute('aria-pressed', String((b.dataset.sigla || null) === estado.salienciaAtiva)));
    desenharPapel(false);
  }));
}

// ---------------------------------------------------------------------------
// Papel de ECG: 3×4 derivações (2,5 s cada) + tira de ritmo em DII (10 s)
// 25 mm/s e 10 mm/mV, como na impressão do eletrocardiógrafo.
// ---------------------------------------------------------------------------

const MM_CALIBRACAO = 8;           // margem esquerda para o pulso de 1 mV
const SEGUNDOS = 10;
const MM_LINHA = 30;               // altura de cada linha de derivações
const LINHAS = [
  ['DI', 'aVR', 'V1', 'V4'],
  ['DII', 'aVL', 'V2', 'V5'],
  ['DIII', 'aVF', 'V3', 'V6'],
];
const cor = (nome) => getComputedStyle(document.documentElement).getPropertyValue(nome).trim();

function escalaMv(valores) {
  // O arquivo pode estar em mV (CODE) ou µV; converte se a amplitude for grande demais para mV.
  const amostra = valores.flat().filter((_, i) => i % 7 === 0).map(Math.abs).sort((a, b) => a - b);
  const p99 = amostra[Math.floor(amostra.length * 0.99)] || 1;
  return p99 > 20 ? 0.001 : 1;
}

let animacao = null;

function desenharPapel(animar) {
  const r = estado.resultado;
  if (!r) return;
  const canvas = $('papel');
  const largura = canvas.clientWidth;
  const mm = largura / (MM_CALIBRACAO + SEGUNDOS * 25);
  const altura = Math.round(mm * (MM_LINHA * 4 + 6));
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(largura * dpr);
  canvas.height = Math.round(altura * dpr);
  canvas.style.height = `${altura}px`;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const { valores, derivacoes, frequencia_hz: fs } = r.sinal;
  const escala = escalaMv(valores);
  const inicio = Math.max(0, Math.floor((valores[0].length - SEGUNDOS * fs) / 2));   // centraliza 10 s
  const x0 = MM_CALIBRACAO * mm;
  const pxPorAmostra = (25 * mm) / fs;
  const pxPorMv = 10 * mm;

  const desenharTudo = (ate) => {
    ctx.clearRect(0, 0, largura, altura);
    desenharGrade(ctx, largura, altura, mm);
    desenharSaliencia(ctx, r, inicio, x0, pxPorAmostra, altura);

    ctx.save();
    ctx.beginPath();
    ctx.rect(0, 0, ate, altura);
    ctx.clip();
    ctx.strokeStyle = cor('--tinta');
    ctx.fillStyle = cor('--tinta');
    ctx.lineWidth = Math.max(1, mm * 0.28);
    ctx.lineJoin = 'round';
    ctx.font = `500 ${Math.max(10, mm * 3)}px 'IBM Plex Mono', monospace`;

    const linhas = [...LINHAS, ['DII*']];
    linhas.forEach((grupo, l) => {
      const base = mm * (MM_LINHA * l + MM_LINHA * 0.62 + 2);
      desenharCalibracao(ctx, mm, base, pxPorMv);
      const segmento = (SEGUNDOS * fs) / grupo.length;
      grupo.forEach((nome, c) => {
        const deriv = derivacoes.indexOf(nome.replace('*', ''));
        const de = inicio + Math.round(c * segmento);
        const ate2 = inicio + Math.round((c + 1) * segmento);
        const serie = valores[deriv];
        ctx.beginPath();
        for (let i = de; i < ate2 && i < serie.length; i++) {
          const x = x0 + (i - inicio) * pxPorAmostra;
          const y = base - serie[i] * escala * pxPorMv;
          i === de ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        }
        ctx.stroke();
        const xRotulo = x0 + c * segmento * pxPorAmostra + mm * 1.5;
        ctx.fillText(nome === 'DII*' ? 'DII (ritmo)' : nome, xRotulo, base - MM_LINHA * mm * 0.42);
        if (c > 0) {                                          // marca da troca de derivação
          const xs = x0 + c * segmento * pxPorAmostra;
          ctx.beginPath(); ctx.moveTo(xs, base - mm * 3); ctx.lineTo(xs, base + mm * 3); ctx.stroke();
        }
      });
    });
    ctx.restore();
  };

  cancelAnimationFrame(animacao);
  const semMovimento = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (!animar || semMovimento) { desenharTudo(largura); return; }

  const t0 = performance.now();
  const passo = (agora) => {
    const p = Math.min(1, (agora - t0) / 900);
    desenharTudo(largura * p);                       // o traçado "sai da impressora"
    if (p < 1) animacao = requestAnimationFrame(passo);
  };
  animacao = requestAnimationFrame(passo);
}

function desenharGrade(ctx, largura, altura, mm) {
  ctx.fillStyle = cor('--papel');
  ctx.fillRect(0, 0, largura, altura);
  for (const [passoMm, corGrade, espessura] of [[1, '--grade-fina', 0.5], [5, '--grade-grossa', 1]]) {
    ctx.strokeStyle = cor(corGrade);
    ctx.lineWidth = espessura;
    ctx.beginPath();
    for (let x = 0; x <= largura; x += passoMm * mm) { ctx.moveTo(x, 0); ctx.lineTo(x, altura); }
    for (let y = 0; y <= altura; y += passoMm * mm) { ctx.moveTo(0, y); ctx.lineTo(largura, y); }
    ctx.stroke();
  }
}

function desenharCalibracao(ctx, mm, base, pxPorMv) {
  // pulso de 1 mV por 200 ms
  const x = mm * 1.5;
  ctx.beginPath();
  ctx.moveTo(x, base);
  ctx.lineTo(x + mm, base);
  ctx.lineTo(x + mm, base - pxPorMv);
  ctx.lineTo(x + mm * 6, base - pxPorMv);
  ctx.lineTo(x + mm * 6, base);
  ctx.lineTo(x + mm * 7, base);
  ctx.stroke();
}

function desenharSaliencia(ctx, r, inicio, x0, pxPorAmostra, altura) {
  const mapa = estado.salienciaAtiva && r.saliencia[estado.salienciaAtiva];
  if (!mapa) return;
  // Cada coluna do papel é um instante de tempo para todas as derivações,
  // então a marcação ocupa a altura inteira, como um marca-texto.
  // Só os ~12% de instantes mais relevantes, para a marcação apontar e não cobrir tudo.
  const corte = [...mapa].sort((a, b) => a - b)[Math.floor(mapa.length * 0.88)];
  ctx.fillStyle = cor('--marcador');
  for (let i = inicio; i < mapa.length; i++) {
    const v = mapa[i];
    if (v < corte) continue;
    ctx.globalAlpha = 0.25 + 0.4 * (v - corte) / Math.max(1e-6, 1 - corte);
    ctx.fillRect(x0 + (i - inicio) * pxPorAmostra, 0, pxPorAmostra + 0.6, altura);
  }
  ctx.globalAlpha = 1;
}

new ResizeObserver(() => desenharPapel(false)).observe($('papel'));

configurarEntrada();
carregarExemplos();
carregarInfoModelo();
