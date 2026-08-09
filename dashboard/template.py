"""Render del dashboard: HTML autocontenido con CSS + JS embebidos (sin CDN)."""
from __future__ import annotations

import json

# Paleta de referencia validada del skill dataviz (usada verbatim, sin re-stepping).
CSS = r"""
:root{
  color-scheme: light;
  --plane:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,0.10);
  --s1:#2a78d6; --s2:#008300; --s3:#e87ba4; --s4:#eda100; --s5:#1baf7a;
  --s6:#eb6834; --s7:#4a3aa7; --s8:#e34948;
  --good:#0ca30c; --warn:#fab219; --serious:#ec835a; --crit:#d03b3b;
}
@media (prefers-color-scheme: dark){
  :root:where(:not([data-theme="light"])){
    color-scheme: dark;
    --plane:#0d0d0d; --surface:#1a1a19; --ink:#fff; --ink2:#c3c2b7; --muted:#898781;
    --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,0.10);
    --s1:#3987e5; --s3:#d55181; --s4:#c98500; --s5:#199e70; --s6:#d95926; --s7:#9085e9; --s8:#e66767;
  }
}
:root[data-theme="dark"]{
  color-scheme: dark;
  --plane:#0d0d0d; --surface:#1a1a19; --ink:#fff; --ink2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,0.10);
  --s1:#3987e5; --s3:#d55181; --s4:#c98500; --s5:#199e70; --s6:#d95926; --s7:#9085e9; --s8:#e66767;
}
*{box-sizing:border-box}
body{margin:0;background:var(--plane);color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif;font-size:14px;line-height:1.5}
a{color:var(--s1)}
.app{display:grid;grid-template-columns:230px 1fr;min-height:100vh}
.side{background:var(--surface);border-right:1px solid var(--border);padding:20px 14px;position:sticky;top:0;height:100vh;overflow:auto}
.brand{font-weight:700;font-size:16px;margin:0 0 2px}
.brand small{display:block;color:var(--muted);font-weight:400;font-size:12px}
.nav{list-style:none;padding:0;margin:18px 0 0}
.nav li{margin:2px 0}
.nav button{width:100%;text-align:left;background:none;border:0;color:var(--ink2);
  padding:9px 12px;border-radius:8px;cursor:pointer;font-size:14px}
.nav button:hover{background:var(--plane)}
.nav button.active{background:var(--s1);color:#fff;font-weight:600}
.main{padding:26px 32px;max-width:1180px;overflow-x:hidden}
h1{font-size:22px;margin:0 0 4px}
h2{font-size:17px;margin:30px 0 12px}
h3{font-size:15px;margin:0 0 8px}
.lead{color:var(--ink2);margin:0 0 18px;max-width:78ch}
.section{display:none}
.section.active{display:block}
.grid{display:grid;gap:14px}
.tiles{grid-template-columns:repeat(auto-fill,minmax(150px,1fr))}
.cards{grid-template-columns:repeat(auto-fill,minmax(320px,1fr))}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:16px}
.tile .k{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.04em}
.tile .v{font-size:26px;font-weight:700;margin-top:4px}
.tile .s{color:var(--ink2);font-size:12px;margin-top:2px}
.chip{display:inline-block;padding:2px 9px;border-radius:999px;font-size:12px;font-weight:600}
.chip.tendencia{background:color-mix(in srgb,var(--s1) 18%,transparent);color:var(--s1)}
.chip.reversion{background:color-mix(in srgb,var(--s7) 22%,transparent);color:var(--s7)}
.chip.mixto{background:color-mix(in srgb,var(--muted) 22%,transparent);color:var(--ink2)}
.chip.alto{background:color-mix(in srgb,var(--crit) 20%,transparent);color:var(--crit)}
.chip.medio{background:color-mix(in srgb,var(--warn) 26%,transparent);color:var(--serious)}
.chip.bajo{background:color-mix(in srgb,var(--muted) 22%,transparent);color:var(--ink2)}
.chip.pendiente{background:color-mix(in srgb,var(--warn) 26%,transparent);color:var(--serious)}
.chip.placeholder{background:color-mix(in srgb,var(--muted) 22%,transparent);color:var(--ink2)}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--border)}
th{color:var(--muted);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.03em}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.tblwrap{overflow-x:auto}
.mbar{height:9px;border-radius:4px;background:var(--s1);display:inline-block;vertical-align:middle}
.mbar.v1{background:var(--muted)}
.mono{font-family:ui-monospace,"Cascadia Code",Consolas,monospace;font-size:12.5px}
.btn{background:var(--s1);color:#fff;border:0;border-radius:8px;padding:8px 13px;cursor:pointer;font-size:13px;font-weight:600}
.btn.ghost{background:none;color:var(--ink2);border:1px solid var(--border)}
.btn:hover{filter:brightness(1.06)}
.toggle{position:absolute;top:20px;right:20px}
.tag{color:var(--muted);font-size:12px}
.note{background:color-mix(in srgb,var(--warn) 14%,transparent);border:1px solid color-mix(in srgb,var(--warn) 40%,transparent);
  border-radius:10px;padding:10px 14px;font-size:13px;color:var(--ink2);margin:8px 0 16px}
.legend{display:flex;gap:14px;flex-wrap:wrap;margin:6px 0 0;font-size:12px;color:var(--ink2)}
.legend span{display:inline-flex;align-items:center;gap:6px}
.swatch{width:11px;height:11px;border-radius:3px;display:inline-block}
.prompt{background:var(--plane);border:1px solid var(--border);border-radius:8px;padding:11px;
  font-family:ui-monospace,Consolas,monospace;font-size:12px;white-space:pre-wrap;color:var(--ink2);max-height:150px;overflow:auto;margin:10px 0}
.rowbtns{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.scenlist{display:flex;flex-direction:column;gap:6px;max-height:520px;overflow:auto;padding-right:6px}
.scenitem{text-align:left;background:var(--surface);border:1px solid var(--border);border-radius:9px;padding:10px 12px;cursor:pointer}
.scenitem:hover{border-color:var(--s1)}
.scenitem.active{border-color:var(--s1);box-shadow:0 0 0 1px var(--s1) inset}
.scenitem .t{font-weight:600}
.split{display:grid;grid-template-columns:280px 1fr;gap:18px;align-items:start}
.chart-tip{position:fixed;pointer-events:none;background:var(--surface);border:1px solid var(--border);
  border-radius:7px;padding:6px 9px;font-size:12px;box-shadow:0 4px 14px rgba(0,0,0,.18);z-index:9;display:none}
svg{max-width:100%;display:block}
.pill{font-size:11px;color:var(--muted);border:1px solid var(--border);border-radius:6px;padding:1px 6px;margin-right:4px}
@media(max-width:820px){.app{grid-template-columns:1fr}.side{position:static;height:auto}.split{grid-template-columns:1fr}}
"""

APP_JS = r"""
const D = window.DATA;
const $ = (s,r=document)=>r.querySelector(s);
const $$ = (s,r=document)=>[...r.querySelectorAll(s)];
const NS='http://www.w3.org/2000/svg';
const cssv = n=>getComputedStyle(document.body).getPropertyValue(n).trim();
const fmt = (x,d=2)=> (x===null||x===undefined||Number.isNaN(x))?'-':Number(x).toFixed(d);

// --- theme ---
function initTheme(){
  const t=$('#themeBtn');
  t.onclick=()=>{
    const cur=document.documentElement.getAttribute('data-theme');
    const next=cur==='dark'?'light':(cur==='light'?'dark':(matchMedia('(prefers-color-scheme: dark)').matches?'light':'dark'));
    document.documentElement.setAttribute('data-theme',next);
    t.textContent=next==='dark'?'☀ Claro':'☾ Oscuro';
    rerenderCharts();
  };
}
let RECHART=[];
function rerenderCharts(){ RECHART.forEach(f=>{try{f()}catch(e){}}); }

// --- nav ---
function initNav(){
  $$('.nav button').forEach(b=>b.onclick=()=>{
    $$('.nav button').forEach(x=>x.classList.remove('active'));
    $$('.section').forEach(x=>x.classList.remove('active'));
    b.classList.add('active');
    $('#'+b.dataset.sec).classList.add('active');
    window.scrollTo(0,0);
  });
}

// --- svg chart helpers ---
function el(tag,attrs){const e=document.createElementNS(NS,tag);for(const k in attrs)e.setAttribute(k,attrs[k]);return e;}
function tip(){let t=$('#ctip');if(!t){t=document.createElement('div');t.id='ctip';t.className='chart-tip';document.body.appendChild(t);}return t;}

function lineChart(host, series, opts={}){
  opts=Object.assign({h:230,marksColor:'var(--s6)',pad:{t:12,r:14,b:26,l:44}},opts);
  const draw=()=>{
    host.innerHTML='';
    const W=host.clientWidth||680, H=opts.h, p=opts.pad;
    const n=Math.max(...series.map(s=>s.values.length));
    let lo=Infinity,hi=-Infinity;
    series.forEach(s=>s.values.forEach(v=>{if(v<lo)lo=v;if(v>hi)hi=v;}));
    if(!isFinite(lo)){lo=0;hi=1;}
    const pad=(hi-lo)*0.06||1; lo-=pad; hi+=pad;
    const X=i=>p.l+(W-p.l-p.r)*(n<2?0:i/(n-1));
    const Y=v=>p.t+(H-p.t-p.b)*(1-(v-lo)/(hi-lo||1));
    const svg=el('svg',{viewBox:`0 0 ${W} ${H}`,width:'100%',height:H});
    // gridlines + y ticks
    for(let g=0;g<=4;g++){const val=lo+(hi-lo)*g/4,y=Y(val);
      svg.appendChild(el('line',{x1:p.l,y1:y,x2:W-p.r,y2:y,stroke:cssv('--grid'),'stroke-width':1}));
      const tx=el('text',{x:p.l-6,y:y+3,'text-anchor':'end',fill:cssv('--muted'),'font-size':10});tx.textContent=fmt(val,0);svg.appendChild(tx);}
    svg.appendChild(el('line',{x1:p.l,y1:H-p.b,x2:W-p.r,y2:H-p.b,stroke:cssv('--axis'),'stroke-width':1}));
    // paths
    series.forEach(s=>{
      let d='';s.values.forEach((v,i)=>{d+=(i?'L':'M')+X(i)+' '+Y(v);});
      svg.appendChild(el('path',{d,fill:'none',stroke:s.color,'stroke-width':2,'stroke-linejoin':'round'}));
      (s.marks||[]).forEach(mi=>{if(mi<s.values.length)svg.appendChild(el('circle',{cx:X(mi),cy:Y(s.values[mi]),r:3.4,fill:opts.marksColor,stroke:cssv('--surface'),'stroke-width':1.4}));});
    });
    // hover
    const cross=el('line',{x1:0,y1:p.t,x2:0,y2:H-p.b,stroke:cssv('--axis'),'stroke-width':1,opacity:0});
    svg.appendChild(cross);
    const hit=el('rect',{x:p.l,y:p.t,width:W-p.l-p.r,height:H-p.t-p.b,fill:'transparent'});
    svg.appendChild(hit);
    const tt=tip();
    hit.addEventListener('mousemove',ev=>{
      const rect=svg.getBoundingClientRect();
      const rel=(ev.clientX-rect.left)/rect.width*W;
      const i=Math.max(0,Math.min(n-1,Math.round((rel-p.l)/((W-p.l-p.r)/(n<2?1:n-1)))));
      cross.setAttribute('x1',X(i));cross.setAttribute('x2',X(i));cross.setAttribute('opacity',1);
      tt.style.display='block';tt.style.left=(ev.clientX+12)+'px';tt.style.top=(ev.clientY-10)+'px';
      tt.innerHTML=(opts.xlab?('<b>'+opts.xlab+' '+i+'</b><br>'):'')+series.map(s=>`<span style="color:${s.color}">■</span> ${s.name}: ${fmt(s.values[i],1)}`).join('<br>');
    });
    hit.addEventListener('mouseleave',()=>{cross.setAttribute('opacity',0);tt.style.display='none';});
    host.appendChild(svg);
  };
  draw();RECHART.push(draw);
}

function dotStrip(host, groups, opts={}){
  // groups: [{label, values:[..], color}] -> filas con puntos y mediana
  opts=Object.assign({h:26*groups.length+40,pad:{t:10,r:16,b:24,l:150}},opts);
  const draw=()=>{
    host.innerHTML='';
    const W=host.clientWidth||680,H=opts.h,p=opts.pad;
    let lo=Infinity,hi=-Infinity;groups.forEach(g=>g.values.forEach(v=>{if(v<lo)lo=v;if(v>hi)hi=v;}));
    if(!isFinite(lo)){lo=0;hi=1;} const pad=(hi-lo)*0.08||1;lo-=pad;hi+=pad;
    const X=v=>p.l+(W-p.l-p.r)*(v-lo)/(hi-lo||1);
    const rowH=(H-p.t-p.b)/groups.length;
    const svg=el('svg',{viewBox:`0 0 ${W} ${H}`,width:'100%',height:H});
    // x grid
    for(let g=0;g<=4;g++){const val=lo+(hi-lo)*g/4,x=X(val);
      svg.appendChild(el('line',{x1:x,y1:p.t,x2:x,y2:H-p.b,stroke:cssv('--grid'),'stroke-width':1}));
      const tx=el('text',{x,y:H-p.b+14,'text-anchor':'middle',fill:cssv('--muted'),'font-size':10});tx.textContent=fmt(val,1);svg.appendChild(tx);}
    // zero line
    if(lo<0&&hi>0){svg.appendChild(el('line',{x1:X(0),y1:p.t,x2:X(0),y2:H-p.b,stroke:cssv('--axis'),'stroke-width':1.4}));}
    const tt=tip();
    groups.forEach((g,gi)=>{
      const cy=p.t+rowH*(gi+0.5);
      const lab=el('text',{x:p.l-10,y:cy+4,'text-anchor':'end',fill:cssv('--ink2'),'font-size':12});lab.textContent=g.label;svg.appendChild(lab);
      const sorted=[...g.values].sort((a,b)=>a-b);
      const med=sorted[Math.floor(sorted.length/2)];
      g.values.forEach(v=>{const c=el('circle',{cx:X(v),cy,r:4,fill:g.color,opacity:.75,stroke:cssv('--surface'),'stroke-width':1});
        c.addEventListener('mousemove',ev=>{tt.style.display='block';tt.style.left=(ev.clientX+12)+'px';tt.style.top=(ev.clientY-10)+'px';tt.innerHTML=`${g.label}<br>Calmar OOS: <b>${fmt(v,2)}</b>`;});
        c.addEventListener('mouseleave',()=>tt.style.display='none');svg.appendChild(c);});
      svg.appendChild(el('line',{x1:X(med),y1:cy-9,x2:X(med),y2:cy+9,stroke:cssv('--ink'),'stroke-width':2}));
    });
    host.appendChild(svg);
  };
  draw();RECHART.push(draw);
}

function miniBar(v,max,cls=''){const w=Math.max(2,Math.round(100*Math.abs(v)/(max||1)));return `<span class="mbar ${cls}" style="width:${w}px"></span>`;}
function esc(s){return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}

// ---------------- sections ----------------
function renderOverview(){
  const k=D.kpis, host=$('#overview');
  const tiles=[
    ['Escenarios ai_v2', k.ai_v2?k.ai_v2.scenarios:'-', (k.ai_v2?k.ai_v2.samples:'-')+' muestras'],
    ['Paths por escenario', k.ai_v2?k.ai_v2.paths:'-','Monte Carlo'],
    ['Estrategias', k.n_strategies,'primitivas paramétricas'],
    ['Features de observación', (k.n_own_features+k.n_regime_features),k.n_own_features+' propio + '+k.n_regime_features+' régimen'],
    ['Commits', k.commit_count||'-', k.commit||''],
  ];
  const roadmapStatus=[
    ['B','Generador (colas, clustering, serial)','hecho'],
    ['Wiring','ai_v2 como sustrato por defecto del scoring','hecho'],
    ['A','Métrica y ranking honestos','pendiente'],
    ['C','Costes que muerden','pendiente'],
    ['D','Validación (CPCV/walk-forward)','pendiente'],
    ['E','Limpieza de consistencia','pendiente'],
  ];
  host.innerHTML=`
    <h1>AI-Trader · Dashboard</h1>
    <p class="lead">Herramienta de inversión sobre datos sintéticos deterministas (paper trading).
      Esta v1 reúne los datos sintéticos, las estrategias, el ranking y las evoluciones pendientes.
      Generado desde el commit <span class="mono">${esc(k.commit)}</span>${k.generated_at?(' · '+k.generated_at):''}.</p>
    <div class="grid tiles">${tiles.map(t=>`<div class="card tile"><div class="k">${t[0]}</div><div class="v">${t[1]}</div><div class="s">${t[2]}</div></div>`).join('')}</div>
    <h2>Estado del roadmap</h2>
    <div class="tblwrap"><table><thead><tr><th>Línea</th><th>Trabajo</th><th>Estado</th></tr></thead><tbody>
    ${roadmapStatus.map(r=>`<tr><td><span class="pill">${r[0]}</span></td><td>${r[1]}</td>
      <td>${r[2]==='hecho'?'<span class="chip" style="background:color-mix(in srgb,var(--good) 20%,transparent);color:var(--good)">✓ hecho</span>':'<span class="chip pendiente">pendiente</span>'}</td></tr>`).join('')}
    </tbody></table></div>
    <p class="tag" style="margin-top:14px">Ve a <b>Evoluciones</b> para el detalle y los prompts copiables de cada mejora.</p>`;
}

function renderSynthetic(){
  const host=$('#synthetic'), sc=D.synthetic.scenarios||[];
  const f=D.facts||{};
  const factRows=[
    ['Autocorr lag-1 (spread entre escenarios)','ac_spread',3,'Diversidad de régimen serial'],
    ['Escenarios que revierten','n_revert',0,'Mean-reversion tiene edge'],
    ['Escenarios que tienden','n_trend',0,'Momentum tiene edge'],
    ['Clustering de |retorno|','clustering',3,'Volatilidad se agrupa'],
    ['Exceedances >3σ (%)','exceed_pct',2,'Colas gruesas'],
  ];
  const colors=[cssv('--s1'),cssv('--s2'),cssv('--s4')];
  host.innerHTML=`
    <h1>Datos sintéticos</h1>
    <p class="lead">Librería <b>${esc(D.synthetic.library)}</b>: ${sc.length} escenarios macro × ${D.synthetic.n_paths} paths Monte Carlo,
      horizonte ${D.synthetic.horizon_days} días, 35 activos. Deterministas y reproducibles. Diseñador: <span class="mono">${esc(D.synthetic.designer||'')}</span>.</p>
    <h2>El mundo dejó de mentir: ai_v1 (iid) → ai_v2 (retrofit)</h2>
    <p class="lead">El generador ganó colas gruesas, clustering de volatilidad y estructura serial. Sin esto, la reversión a la media era inganable por construcción.</p>
    <div class="card"><div class="tblwrap"><table><thead><tr><th>Stylized fact</th><th class="num">ai_v1 (iid)</th><th class="num">ai_v2</th><th>Qué significa</th></tr></thead><tbody>
    ${factRows.map(r=>{const v1=f.ai_v1?f.ai_v1[r[1]]:null,v2=f.ai_v2?f.ai_v2[r[1]]:null;const mx=Math.max(Math.abs(v1||0),Math.abs(v2||0));
      return `<tr><td>${r[0]}</td>
      <td class="num">${fmt(v1,r[2])} ${miniBar(v1,mx,'v1')}</td>
      <td class="num">${fmt(v2,r[2])} ${miniBar(v2,mx)}</td>
      <td class="tag">${r[3]}</td></tr>`;}).join('')}
    </tbody></table></div></div>
    <div class="note"><b>Sustrato por defecto.</b> El harness de scoring optimiza sobre <span class="mono">ai_v2</span>
      (<span class="mono">DEFAULT_LIBRARY_ID</span> en <span class="mono">src/ai_trader/scoring/optimize.py</span>);
      <span class="mono">ai_v1</span> se conserva solo como referencia comparativa y hay que pedirla explícitamente.</div>
    <h2>Explorador de escenarios</h2>
    <div class="split">
      <div class="scenlist" id="scenlist"></div>
      <div id="scendetail"></div>
    </div>`;
  const list=$('#scenlist');
  sc.forEach((s,i)=>{const b=document.createElement('button');b.className='scenitem'+(i===0?' active':'');
    b.innerHTML=`<div class="t">${esc(s.name)}</div><span class="chip ${s.regime}">${s.regime}</span> <span class="tag">idio_ar ${fmt(s.idio_ar_avg,2)}</span>`;
    b.onclick=()=>{$$('.scenitem').forEach(x=>x.classList.remove('active'));b.classList.add('active');showScenario(s);};
    list.appendChild(b);});
  if(sc.length)showScenario(sc[0]);

  function showScenario(s){
    const d=$('#scendetail');const syms=Object.keys(s.series||{});
    d.innerHTML=`<div class="card">
      <h3>${esc(s.name)} <span class="chip ${s.regime}">${s.regime}</span></h3>
      <p class="lead" style="margin:6px 0 4px">${esc(s.narrative)}</p>
      <div class="legend">${syms.map((sy,i)=>`<span><span class="swatch" style="background:${colors[i%3]}"></span>${esc(sy)}</span>`).join('')}
        <span class="tag">· precio normalizado a 100 · ${s.n_shocks} shock(s)</span></div>
      <div id="scenchart"></div>
      <h3 style="margin-top:14px">Fases (${s.phases.length})</h3>
      <div class="tblwrap"><table><thead><tr><th>Días</th><th>Deriva top</th><th>Vol top</th><th class="num">idio_ar</th><th class="num">colas</th><th class="num">salto</th><th></th></tr></thead><tbody>
      ${s.phases.map(p=>`<tr><td class="num">${p.length_days}</td><td class="mono">${esc(p.top_drift)}</td><td class="mono">${esc(p.top_vol)}</td>
        <td class="num">${fmt(p.idio_ar,2)}</td><td class="num">${p.tail_dof?('dof '+fmt(p.tail_dof,0)):'-'}</td>
        <td class="num">${p.jump_intensity?fmt(p.jump_intensity,3):'-'}</td>
        <td>${p.crisis?'<span class="chip alto">crisis</span>':''}</td></tr>`).join('')}
      </tbody></table></div></div>`;
    const series=syms.map((sy,i)=>({name:sy,color:colors[i%3],values:s.series[sy]}));
    if(series.length)lineChart($('#scenchart'),series,{xlab:'día',h:240});
  }
}

function renderStrategies(){
  const host=$('#strategies'),S=D.strategies;
  const demo=D.signals||{};
  host.innerHTML=`
    <h1>Estrategias</h1>
    <p class="lead">Dos primitivas paramétricas de regímenes opuestos. El motor de scoring/RL las optimiza sobre los backtests sintéticos.</p>
    <div class="grid cards">
    ${S.strategies.map(st=>`<div class="card">
      <h3>${esc(st.name)} <span class="chip ${st.regime}">${st.regime}</span></h3>
      <p class="lead" style="margin:6px 0 8px">${esc(st.idea)}</p>
      <b class="tag">Cómo decide la entrada</b>
      <ul style="margin:6px 0 10px;padding-left:18px">${st.rules.map(r=>`<li>${esc(r)}</li>`).join('')}</ul>
      <b class="tag">Parámetros</b>
      <div class="tblwrap"><table><tbody>${st.params.map(p=>`<tr><td class="mono">${esc(p.name)}</td><td class="num mono">${p.value}</td></tr>`).join('')}</tbody></table></div>
      <div id="demo_${st.id}" style="margin-top:12px"></div>
    </div>`).join('')}
    </div>
    <h2>Espacio de observación</h2>
    <p class="lead">Lo que la política ve en CADA decisión (anti-look-ahead: solo hasta el cierre de ayer). Vector de orden estable, base del RL futuro.</p>
    <div class="split" style="grid-template-columns:1fr 1fr">
      <div class="card"><h3>Mercado del propio activo (${S.observation.own_asset.length})</h3>
        <div class="tblwrap"><table><tbody>${S.observation.own_asset.map(f=>`<tr><td class="mono">${esc(f.name)}</td><td class="tag">${esc(f.desc)}</td></tr>`).join('')}</tbody></table></div></div>
      <div class="card"><h3>Cross-sectional / régimen (${S.observation.regime.length})</h3>
        <div class="tblwrap"><table><tbody>${S.observation.regime.map(f=>`<tr><td class="mono">${esc(f.name)}</td><td class="tag">${esc(f.desc)}</td></tr>`).join('')}</tbody></table></div>
        <p class="tag" style="margin-top:8px">+ one-hot de clase de activo (crypto/stock/macro).</p></div>
    </div>`;
  S.strategies.forEach(st=>{const dm=demo[st.id];const holder=$('#demo_'+st.id);if(!dm){holder.innerHTML='<span class="tag">Sin demo de señales.</span>';return;}
    holder.innerHTML=`<b class="tag">Entradas sobre una muestra real (${esc(dm.symbol)}, escenario ${esc(dm.scenario)})</b>
      <div class="legend"><span><span class="swatch" style="background:var(--s1)"></span>precio</span><span><span class="swatch" style="background:var(--s6)"></span>entrada (${dm.signals.length})</span></div>
      <div id="sig_${st.id}"></div>`;
    lineChart($('#sig_'+st.id),[{name:'precio',color:cssv('--s1'),values:dm.series,marks:dm.signals}],{h:180,xlab:'día'});
  });
}

function renderRanking(){
  const host=$('#ranking'),R=D.ranking,sc=R.scope||{};
  host.innerHTML=`
    <h1>Ranking de estrategias</h1>
    <p class="lead">La unidad de evaluación es la DISTRIBUCIÓN sobre muestras (no un path). Se rankea por <b>CVaR@25%</b> del Calmar out-of-sample (media del peor cuartil: robusto y consciente de la cola).</p>
    <div class="note"><b>Muestra reducida.</b> ${sc.n_scenarios} escenarios × ${sc.n_paths} paths sobre <b>${esc(sc.library)}</b>,
      universo de ${sc.universe?sc.universe.length:'?'} activos, ventana ${sc.window_days} días. Para ampliar el scope,
      edita las constantes <span class="mono">RANK_*</span> en <span class="mono">dashboard/build_dashboard.py</span> y regenera.
      <br>Nota: el headline actual es Calmar OOS; sustituirlo por Sharpe−turnover−κ·maxDD es la Línea A (pendiente).</div>
    ${(R.rows&&R.rows.length)?`
    <div class="card"><div class="tblwrap"><table><thead><tr><th>#</th><th>Estrategia</th><th class="num">CVaR@25%</th><th class="num">media</th><th class="num">P25</th><th class="num">std</th><th class="num">peor</th><th class="num">mejor</th><th class="num">n</th></tr></thead><tbody>
    ${R.rows.map((r,i)=>`<tr><td class="num">${i+1}</td><td><b>${esc(r.label)}</b> <span class="pill">${esc(r.type)}</span></td>
      <td class="num"><b>${fmt(r.cvar25)}</b></td><td class="num">${fmt(r.mean)}</td><td class="num">${fmt(r.p25)}</td>
      <td class="num">${fmt(r.std)}</td><td class="num">${fmt(r.worst)}</td><td class="num">${fmt(r.best)}</td><td class="num">${r.n}</td></tr>`).join('')}
    </tbody></table></div></div>
    <h2>Distribución detrás del ranking</h2>
    <p class="lead">Cada punto es el Calmar OOS de una muestra; la barra vertical es la mediana. La cola izquierda es lo que penaliza el CVaR.</p>
    <div class="card"><div id="distchart"></div></div>`
    :`<div class="card"><p class="tag">Ranking no disponible (¿falta ai_v2 en disco? corre <span class="mono">build_dashboard.py</span>).</p></div>`}`;
  if(R.rows&&R.rows.length){
    const pal=[cssv('--s1'),cssv('--s2'),cssv('--s4'),cssv('--s7'),cssv('--s6'),cssv('--s5')];
    const groups=R.rows.map((r,i)=>({label:r.label,values:(R.distributions[r.label]||[]),color:pal[i%pal.length]}));
    dotStrip($('#distchart'),groups);
  }
}

function renderPaper(){
  $('#paper').innerHTML=`
    <h1>Paper trading</h1>
    <p class="lead">El runner ya opera en paper con estado persistido, pero esta vista aún no está conectada. Se poblará más adelante.</p>
    <div class="grid cards">
      <div class="card"><h3>Curva de equity <span class="chip placeholder">próximamente</span></h3><p class="tag">Line chart de equity marcado a mercado a lo largo del tiempo.</p></div>
      <div class="card"><h3>Posiciones <span class="chip placeholder">próximamente</span></h3><p class="tag">Tabla de posiciones abiertas y cerradas con PnL neto de comisiones.</p></div>
      <div class="card"><h3>Riesgo <span class="chip placeholder">próximamente</span></h3><p class="tag">Exposición desplegada, nº de posiciones vs máximo, drawdown de cuenta.</p></div>
    </div>
    <div class="note">Para implementarla, usa el prompt "Vista de paper trading" en la sección <b>Evoluciones</b>.</div>`;
}

function renderRoadmap(){
  const host=$('#roadmap');
  host.innerHTML=`
    <h1>Evoluciones pendientes</h1>
    <p class="lead">Cada mejora acordada, con su prompt detallado para Claude Code. Copia el prompt y pégaselo cuando quieras abordarla, en el orden que decidas.</p>
    <div class="grid cards">
    ${D.roadmap.map((r,i)=>`<div class="card">
      <div class="rowbtns" style="justify-content:space-between">
        <h3 style="margin:0">${esc(r.title)}</h3><span class="pill">${esc(r.line)}</span></div>
      <div class="rowbtns" style="margin:8px 0">
        <span class="chip ${r.status}">${esc(r.status)}</span>
        <span class="chip ${r.impact}">impacto ${esc(r.impact)}</span>
        <span class="tag">esfuerzo ${esc(r.effort)}</span></div>
      <p class="lead" style="margin:2px 0 4px">${esc(r.why)}</p>
      <div class="prompt" id="pr_${i}">${esc(r.prompt)}</div>
      <div class="rowbtns"><button class="btn" onclick="copyPrompt(${i},this)">Copiar prompt</button></div>
    </div>`).join('')}
    </div>`;
}
function copyPrompt(i,btn){const t=D.roadmap[i].prompt;navigator.clipboard.writeText(t).then(()=>{const o=btn.textContent;btn.textContent='✓ Copiado';setTimeout(()=>btn.textContent=o,1500);});}
window.copyPrompt=copyPrompt;

function main(){
  initTheme();initNav();
  renderOverview();renderSynthetic();renderStrategies();renderRanking();renderPaper();renderRoadmap();
  addEventListener('resize',()=>{clearTimeout(window._rt);window._rt=setTimeout(rerenderCharts,150);});
}
main();
"""

SHELL = """
<div class="app">
  <aside class="side">
    <div class="brand">AI-Trader <small>dashboard v1</small></div>
    <ul class="nav">
      <li><button class="active" data-sec="overview">Resumen</button></li>
      <li><button data-sec="synthetic">Datos sintéticos</button></li>
      <li><button data-sec="strategies">Estrategias</button></li>
      <li><button data-sec="ranking">Ranking</button></li>
      <li><button data-sec="paper">Paper trading</button></li>
      <li><button data-sec="roadmap">Evoluciones</button></li>
    </ul>
  </aside>
  <main class="main">
    <button id="themeBtn" class="btn ghost toggle">☾ Oscuro</button>
    <section id="overview" class="section active"></section>
    <section id="synthetic" class="section"></section>
    <section id="strategies" class="section"></section>
    <section id="ranking" class="section"></section>
    <section id="paper" class="section"></section>
    <section id="roadmap" class="section"></section>
  </main>
</div>
"""


def render_html(data: dict) -> str:
    blob = json.dumps(data, ensure_ascii=False)
    return (
        "<!doctype html><html lang=\"es\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        "<title>AI-Trader Dashboard</title><style>" + CSS + "</style></head><body>"
        + SHELL
        + "<script>window.DATA=" + blob + ";</script>"
        + "<script>" + APP_JS + "</script>"
        + "</body></html>"
    )
