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
.chip.hecho{background:color-mix(in srgb,var(--good) 20%,transparent);color:var(--good)}
.chip.pend{background:color-mix(in srgb,var(--crit) 18%,transparent);color:var(--crit)}
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
        c.addEventListener('mousemove',ev=>{tt.style.display='block';tt.style.left=(ev.clientX+12)+'px';tt.style.top=(ev.clientY-10)+'px';tt.innerHTML=`${g.label}<br>Headline OOS: <b>${fmt(v,2)}</b>`;});
        c.addEventListener('mouseleave',()=>tt.style.display='none');svg.appendChild(c);});
      svg.appendChild(el('line',{x1:X(med),y1:cy-9,x2:X(med),y2:cy+9,stroke:cssv('--ink'),'stroke-width':2}));
    });
    host.appendChild(svg);
  };
  draw();RECHART.push(draw);
}

function scatterChart(host, points, opts={}){
  // points: [{name,x,y,lo,hi}] -> real (x) vs sintetico (y), con el rango p10-p90 del
  // ensemble como barra vertical. Misma escala en ambos ejes: la diagonal y=x es la
  // referencia y solo significa algo si las unidades coinciden.
  opts=Object.assign({h:320,pad:{t:16,r:18,b:40,l:58},d:2,xlab:'real',ylab:'sintético'},opts);
  const draw=()=>{
    host.innerHTML='';
    const W=host.clientWidth||680,H=opts.h,p=opts.pad;
    let lo=Infinity,hi=-Infinity;
    points.forEach(pt=>[pt.x,pt.y,pt.lo,pt.hi].forEach(v=>{
      if(v===null||v===undefined||Number.isNaN(v))return; if(v<lo)lo=v; if(v>hi)hi=v;}));
    if(!isFinite(lo)){lo=0;hi=1;}
    const pad=(hi-lo)*0.10||Math.abs(hi)*0.1||1; lo-=pad; hi+=pad;
    const X=v=>p.l+(W-p.l-p.r)*(v-lo)/(hi-lo||1);
    const Y=v=>p.t+(H-p.t-p.b)*(1-(v-lo)/(hi-lo||1));
    const svg=el('svg',{viewBox:`0 0 ${W} ${H}`,width:'100%',height:H});
    for(let g=0;g<=4;g++){
      const val=lo+(hi-lo)*g/4;
      svg.appendChild(el('line',{x1:p.l,y1:Y(val),x2:W-p.r,y2:Y(val),stroke:cssv('--grid'),'stroke-width':1}));
      svg.appendChild(el('line',{x1:X(val),y1:p.t,x2:X(val),y2:H-p.b,stroke:cssv('--grid'),'stroke-width':1}));
      const ty=el('text',{x:p.l-7,y:Y(val)+3,'text-anchor':'end',fill:cssv('--muted'),'font-size':10});
      ty.textContent=fmt(val,opts.d);svg.appendChild(ty);
      const tx=el('text',{x:X(val),y:H-p.b+15,'text-anchor':'middle',fill:cssv('--muted'),'font-size':10});
      tx.textContent=fmt(val,opts.d);svg.appendChild(tx);
    }
    svg.appendChild(el('line',{x1:X(lo),y1:Y(lo),x2:X(hi),y2:Y(hi),stroke:cssv('--axis'),
      'stroke-width':1.5,'stroke-dasharray':'5 4'}));
    const dl=el('text',{x:W-p.r-6,y:Y(hi)+15,'text-anchor':'end',fill:cssv('--muted'),'font-size':10});
    dl.textContent='sintético = real';svg.appendChild(dl);
    const xl=el('text',{x:(p.l+W-p.r)/2,y:H-4,'text-anchor':'middle',fill:cssv('--ink2'),'font-size':11});
    xl.textContent=opts.xlab;svg.appendChild(xl);
    const cy=(p.t+H-p.b)/2;
    const yl=el('text',{x:13,y:cy,'text-anchor':'middle',fill:cssv('--ink2'),'font-size':11,
      transform:`rotate(-90 13 ${cy})`});
    yl.textContent=opts.ylab;svg.appendChild(yl);
    const tt=tip();
    points.forEach(pt=>{
      if(pt.lo!==null&&pt.hi!==null)
        svg.appendChild(el('line',{x1:X(pt.x),y1:Y(pt.lo),x2:X(pt.x),y2:Y(pt.hi),stroke:cssv('--s1'),
          'stroke-width':2.5,opacity:.25,'stroke-linecap':'round'}));
      svg.appendChild(el('circle',{cx:X(pt.x),cy:Y(pt.y),r:5,fill:cssv('--s1'),
        stroke:cssv('--surface'),'stroke-width':2}));
      const hit=el('circle',{cx:X(pt.x),cy:Y(pt.y),r:15,fill:'transparent'});
      hit.addEventListener('mousemove',ev=>{
        tt.style.display='block';tt.style.left=(ev.clientX+12)+'px';tt.style.top=(ev.clientY-10)+'px';
        tt.innerHTML=`<b>${esc(pt.name)}</b><br>real: ${fmt(pt.x,opts.d)}<br>`+
          `sintético: ${fmt(pt.y,opts.d)} <span style="opacity:.7">[${fmt(pt.lo,opts.d)}, ${fmt(pt.hi,opts.d)}]</span>`+
          `<br>${pt.inside?'✓ el real cae dentro del rango':'✗ fuera del rango sintético'}`;
      });
      hit.addEventListener('mouseleave',()=>tt.style.display='none');
      svg.appendChild(hit);
    });
    host.appendChild(svg);
  };
  draw();RECHART.push(draw);
}

function foldStrip(host, folds, opts={}){
  // Geometría real de la validación: una fila por fold, con las bandas de train y test
  // sobre el mismo eje temporal. El hueco visible entre una banda azul y la verde ES la
  // purga; el que queda a la derecha del test, el embargo. Nada de esto es ilustrativo:
  // son los intervalos que se ejecutaron.
  opts=Object.assign({rowH:17,pad:{t:14,r:14,b:22,l:74}},opts);
  const draw=()=>{
    host.innerHTML='';
    const W=host.clientWidth||680,p=opts.pad,H=p.t+p.b+opts.rowH*folds.length;
    const X=f=>p.l+(W-p.l-p.r)*f;
    const svg=el('svg',{viewBox:`0 0 ${W} ${H}`,width:'100%',height:H});
    for(let g=0;g<=4;g++){const x=X(g/4);
      svg.appendChild(el('line',{x1:x,y1:p.t-4,x2:x,y2:H-p.b+2,stroke:cssv('--grid'),'stroke-width':1}));
      const t=el('text',{x,y:H-p.b+15,'text-anchor':'middle',fill:cssv('--muted'),'font-size':10});
      t.textContent=(g*25)+'%';svg.appendChild(t);}
    const tt=tip();
    folds.forEach((f,i)=>{
      const y=p.t+opts.rowH*i, h=opts.rowH-6;
      const lab=el('text',{x:p.l-8,y:y+h-1,'text-anchor':'end',fill:cssv('--ink2'),'font-size':10.5});
      lab.textContent=f.label;svg.appendChild(lab);
      const band=(b,color,kind)=>{
        const r=el('rect',{x:X(b.a),y,width:Math.max(1.5,X(b.b)-X(b.a)),height:h,rx:2,fill:color});
        r.addEventListener('mousemove',ev=>{tt.style.display='block';tt.style.left=(ev.clientX+12)+'px';
          tt.style.top=(ev.clientY-10)+'px';
          tt.innerHTML=`<b>${esc(f.label)}</b><br>${kind}: ${(b.a*100).toFixed(1)}% – ${(b.b*100).toFixed(1)}% del rango`;});
        r.addEventListener('mouseleave',()=>tt.style.display='none');
        svg.appendChild(r);};
      f.train.forEach(b=>band(b,cssv('--s1'),'train'));
      f.test.forEach(b=>band(b,cssv('--s2'),'test (OOS)'));
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
    ['A','Métrica y ranking honestos (headline, CVaR, baselines, DSR/PBO)','hecho'],
    ['A','Calibración medida de λ y κ (rejilla + auditoría de costes)','hecho'],
    ['C','Costes que muerden (spread por símbolo, impacto, capacidad)','hecho'],
    ['B','Fidelidad sintético-vs-real medida contra CCXT (rank-corr)','hecho'],
    ['D','Validación (CPCV/walk-forward)','pendiente'],
    ['E','Limpieza de consistencia (universo, anualización, diseñador)','hecho'],
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
    <div class="note"><b>Qué es reproducible aquí, y qué no.</b> Todo lo que va del <i>spec.json</i> hacia abajo
      —caminos, velas, backtests, métricas— es determinista dado el spec y la semilla. El <b>diseño</b> en sí no lo es
      ni puede serlo: los modelos actuales retiraron los parámetros de muestreo (<span class="mono">temperature</span>
      y compañía devuelven error), así que no existe palanca de determinismo y rehacer una librería con IA produce
      siempre una librería nueva. Por eso el <i>spec.json</i> se guarda: es la única salida cara e insustituible.
      <br><br><b>MATIC/USDT.</b> Está deslistado en Binance (migró a POL) y por eso salió del universo que se opera en
      vivo, pero se mantiene <i>a propósito</i> en el sintético: aquí el símbolo es la etiqueta de un perfil de cargas
      factoriales, no un par que se pida a un exchange, y retirarlo cambiaría el universo a 34 activos, desincronizando
      la calibración de pesos y el estudio de fidelidad, ambos medidos sobre 35.</div>
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
    <div class="note"><b>Esta tabla compara el mundo sintético consigo mismo.</b> Que ai_v2 tenga colas y
      agrupamiento no dice que los tenga <i>en la magnitud del mercado</i>. Esa pregunta se responde
      midiendo contra el histórico real: sección <b>Fidelidad</b>.</div>
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

function renderFidelity(){
  const host=$('#fidelity'),F=D.fidelity;
  if(!F){
    host.innerHTML=`<h1>Fidelidad contra el mercado real</h1>
      <div class="card"><p class="tag">No hay informe publicado. Genéralo con
      <span class="mono">python -m ai_trader.synthetic.fidelity_study</span>.</p></div>`;
    return;
  }
  const S=F.summary, all=[...F.metrics,F.cross], M=k=>all.find(r=>r.key===k);
  const rows=[...F.metrics.filter(m=>m.is_target),F.cross,...F.metrics.filter(m=>!m.is_target)];
  const kurt=M('excess_kurtosis'),exc=M('exceed_3sigma_pct'),clus=M('ac_abs1'),
        vol=M('vol_annual_pct'),ac=M('ac1'),cross=F.cross;
  const pct=v=>(v===null||v===undefined)?'—':fmt(v,0)+'%';
  const ratio=r=>r.ratio===null?'—':fmt(r.ratio,2)+'×';
  const tiles=[
    ['Rank corr · pares',fmt(S.rank_corr_cross,2),'ordena los '+S.n_pairs+' pares como el mercado'],
    ['Cobertura media',pct(S.coverage_mean_pct),'valores reales dentro del rango sintético'],
    ['Curtosis real / sint.',fmt(kurt.real_median,1)+' / '+fmt(kurt.synth_median,1),'el hueco más grande'],
    ['Muestras',S.n_real_windows+' / '+S.n_synthetic_samples,'ventanas reales / paths sintéticos'],
  ];
  host.innerHTML=`
    <h1>Fidelidad contra el mercado real</h1>
    <p class="lead">Un mercado sintético solo vale si se parece al real en sus propiedades estadísticas.
      Aquí se miden los mismos <i>stylized facts</i> sobre <b>${esc(F.library)}</b> y sobre el histórico
      diario real de ${esc(F.exchange)} (${esc(F.real_start)} → ${esc(F.real_end)}, ${S.n_symbols} criptos),
      y se comparan en tres ejes distintos: <b>nivel</b> (¿la magnitud es la del mercado?),
      <b>ordenación</b> (¿ordena los activos como el mercado?, correlación de rangos de Spearman) y
      <b>cobertura</b> (¿produce el ensemble sintético el valor real como una realización plausible?).</p>
    <div class="grid tiles">${tiles.map(t=>`<div class="card tile"><div class="k">${t[0]}</div>
      <div class="v">${t[1]}</div><div class="s">${t[2]}</div></div>`).join('')}</div>
    <div class="note"><b>Comparación pareada en tamaño de muestra.</b> El histórico real se trocea en
      ventanas de <b>${F.window_days} días</b> —el mismo horizonte que un camino sintético— porque la
      autocorrelación y sobre todo la curtosis están sesgadas en muestras cortas: medir el real sobre ocho
      años seguidos y el sintético sobre dos compararía el sesgo, no el mundo. Las ventanas avanzan
      ${F.step_days} días${F.overlap?' y por tanto <b>solapan</b>: sirven para la tendencia central, no como estimaciones independientes':''}.</div>
    <h2>Métrica a métrica</h2>
    <div class="card"><div class="tblwrap"><table>
      <thead><tr><th>Stylized fact</th><th class="num">real</th><th class="num">sintético</th>
        <th class="num">ratio</th><th class="num">rank corr</th><th class="num">cobertura</th></tr></thead>
      <tbody>${rows.map(r=>`<tr>
        <td>${esc(r.label)} ${r.key==='cross_corr'?'<span class="pill">pares</span>':(r.is_target?'':'<span class="pill">contexto</span>')}</td>
        <td class="num">${fmt(r.real_median,r.decimals)}</td>
        <td class="num">${fmt(r.synth_median,r.decimals)}</td>
        <td class="num">${ratio(r)}</td>
        <td class="num">${fmt(r.rank_corr,2)}</td>
        <td class="num">${fmt(r.coverage_pct,0)}%</td></tr>`).join('')}
      </tbody></table></div></div>
    <p class="tag"><b>ratio</b> = sintético / real (1,00 sería clavarlo). <b>rank corr</b> = Spearman entre
      la sección cruzada real y la sintética: es invariante a la escala, así que mide la ordenación aunque
      el nivel esté mal. <b>cobertura</b> = qué fracción de los valores reales cae dentro del rango
      [p10, p90] que el ensemble sintético produce para ese mismo activo (o par).</p>
    <h2>Activo a activo</h2>
    <p class="lead">Cada punto es un activo (o un par, en las correlaciones cruzadas): en el eje X su valor
      real, en el Y el que produce el mundo sintético, con la barra vertical marcando el rango p10–p90 del
      ensemble. Si el generador fuese fiel, los puntos caerían sobre la diagonal.</p>
    <div class="rowbtns" id="fidbtns"></div>
    <div class="card" style="margin-top:12px">
      <div class="legend"><span><span class="swatch" style="background:var(--s1)"></span>mediana sintética</span>
        <span><span class="swatch" style="background:var(--s1);opacity:.3"></span>rango p10–p90 del ensemble</span>
        <span class="tag">· diagonal = fidelidad perfecta</span></div>
      <div id="fidchart"></div>
      <div class="tblwrap" id="fidtable" style="margin-top:10px;max-height:360px;overflow:auto"></div>
    </div>
    <h2>Veredicto</h2>
    <div class="note"><b>1 · Lo que el mundo sintético acierta.</b> El <b>nivel de riesgo</b> es el del
      mercado (volatilidad anualizada ${fmt(vol.real_median,0)}% real vs ${fmt(vol.synth_median,0)}%
      sintético, ratio ${ratio(vol)}), y el modelo de factores <b>ordena los pares como la realidad</b>:
      rank corr ${fmt(cross.rank_corr,2)} sobre ${cross.n} pares. Que las correlaciones cruzadas emerjan de
      cargas compartidas —y no de una matriz inventada— produce un acoplamiento con la forma correcta.</div>
    <div class="note"><b>2 · Lo que no acierta: las colas.</b> La curtosis en exceso real es
      <b>${fmt(kurt.real_median,1)}</b> y la sintética <b>${fmt(kurt.synth_median,1)}</b>
      (${fmt(kurt.coverage_pct,0)}% de cobertura), y las exceedances más allá de 3σ son
      ${fmt(exc.real_median,2)}% frente a ${fmt(exc.synth_median,2)}%. Cobertura ${fmt(kurt.coverage_pct,0)}%
      no significa "el sintético se queda corto de media": significa que <b>ni en su percentil 90</b> el
      ensemble llega al valor real. El agrupamiento de volatilidad va por el mismo camino, a
      <b>${ratio(clus)}</b> del real.</div>
    <div class="note"><b>3 · Qué implica para lo que se mide con este sustrato.</b> Un mundo con colas más
      finas que las reales <b>subestima la pérdida de cola</b>: los drawdowns de crisis, los huecos que se
      saltan un stop y el peor cuartil del CVaR salen mejores de lo que saldrían contra el mercado. Los
      rankings de estrategias siguen siendo comparaciones honestas <i>entre sí</i> (todas compiten en el
      mismo mundo), pero sus <b>cifras absolutas de riesgo son optimistas</b>. Cerrarlo es la evolución
      "Subir colas y clustering de ai_v2 al nivel medido" de la sección Evoluciones.</div>
    <div class="note"><b>4 · La autocorrelación merece leerse aparte.</b> Real ${fmt(ac.real_median,3)},
      sintético ${fmt(ac.synth_median,3)}: el mercado no regala estructura serial —si la regalara, sería
      dinero gratis— mientras que el generador la <b>fija a propósito</b>, con signo según el régimen, para
      que la reversión a la media y el momentum tengan algo que capturar. Aquí un ratio lejos de 1 no es un
      defecto sino el diseño: sin ella no habría <i>edge</i> que buscar y el evaluador mediría ruido. Lo que
      sí es una advertencia es que el <i>edge</i> sintético es más limpio que el real.</div>
    <div class="note"><b>Límites.</b> Solo cripto: la renta variable del universo va por otro proveedor y
      otra sesión de mercado, así que ${S.n_symbols} activos y ${S.n_pairs} pares es todo el ancho que hay.
      Con ${S.n_symbols} puntos, una correlación de rangos tiene un error grande: sirve para distinguir
      "ordena como el mercado" de "no ordena", no para comparar 0,55 con 0,65.
      ${F.missing.length?('Sin contraparte real: <span class="mono">'+F.missing.map(esc).join(', ')+'</span>.'):''}
      Además, el histórico real es un único camino de la historia: sus siete ventanas comparten los mismos
      ciclos de 2018-2025, mientras que el sintético son ${S.n_synthetic_samples} mundos distintos.</div>
    <p class="tag">Evidencia completa: <span class="mono">data/fidelity/report_${esc(F.library)}.json</span> ·
      reproducible con <span class="mono">python -m ai_trader.synthetic.fidelity_study</span>
      (${F.n_scenarios} escenarios × ${F.n_paths} caminos; ${esc(F.generated_at)}).</p>`;

  const selectable=[...F.metrics.filter(m=>m.is_target),F.cross];
  const btns=$('#fidbtns');
  selectable.forEach((m,i)=>{
    const b=document.createElement('button');
    b.className='btn'+(i?' ghost':'');
    b.textContent=m.label.replace(/ \(.*/,'');
    b.onclick=()=>{
      $$('#fidbtns .btn').forEach(x=>x.className='btn ghost');
      b.className='btn';show(m);
    };
    btns.appendChild(b);
  });
  if(selectable.length)show(selectable[0]);

  function show(m){
    const pts=m.items.map(i=>({name:i.name.replace('|',' · '),x:i.real,y:i.synth,
      lo:i.synth_p10,hi:i.synth_p90,inside:i.inside}));
    scatterChart($('#fidchart'),pts,{d:m.decimals,xlab:'real · '+m.label,ylab:'sintético'});
    $('#fidtable').innerHTML=`<table><thead><tr><th>${m.key==='cross_corr'?'Par':'Activo'}</th>
      <th class="num">real</th><th class="num">sintético</th><th class="num">p10</th>
      <th class="num">p90</th><th>cubierto</th></tr></thead><tbody>
      ${m.items.map(i=>`<tr><td class="mono">${esc(i.name.replace('|',' · '))}</td>
        <td class="num">${fmt(i.real,m.decimals)}</td><td class="num">${fmt(i.synth,m.decimals)}</td>
        <td class="num">${fmt(i.synth_p10,m.decimals)}</td><td class="num">${fmt(i.synth_p90,m.decimals)}</td>
        <td>${i.inside?'<span class="chip hecho">sí</span>':'<span class="chip pend">no</span>'}</td></tr>`).join('')}
      </tbody></table>`;
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

function calibrationPanel(C){
  if(!C) return '';
  const w=C.weights,ch=C.chosen,nu=C.neutral||{},co=C.cost;
  const ics=C.points.map(p=>p.rank_ic_mean), lo=Math.min(...ics), hi=Math.max(...ics);
  const shade=v=>{const t=hi>lo?(v-lo)/(hi-lo):0.5;
    return `background:color-mix(in srgb,var(--accent) ${(8+t*30).toFixed(0)}%,transparent)`;};
  const at=(l,k)=>C.points.find(p=>p.lambda_turnover===l&&p.kappa_maxdd===k);
  const head=C.kappas.map(k=>`<th class="num">κ=${k}</th>`).join('');
  const rows=C.lambdas.map(l=>`<tr><td class="mono">λ=${l}</td>${C.kappas.map(k=>{
    const p=at(l,k),sel=(l===w.lambda_turnover&&k===w.kappa_maxdd);
    return `<td class="num mono" style="${shade(p.rank_ic_mean)}">${sel?'<b>'+fmt(p.rank_ic_mean,3)+'</b>':fmt(p.rank_ic_mean,3)}</td>`;
  }).join('')}</tr>`).join('');
  return `
    <h2>De dónde salen λ y κ</h2>
    <p class="lead">Los pesos no son una preferencia estética: son una <b>regla de selección</b>, y se juzgan por lo que eligen.
      Se barrieron en rejilla sobre <b>${C.n_backtests} backtests reales</b> de <span class="mono">${esc(C.library)}</span>
      (${C.n_configs} configuraciones × ${C.n_samples} muestras). Los componentes del score (Sharpe, turnover, maxDD) no dependen
      de los pesos, así que se backtestea una vez y la rejilla entera se evalúa después en memoria.</p>
    <div class="grid tiles">
      <div class="card tile"><div class="k">λ · turnover</div><div class="v">${w.lambda_turnover}</div><div class="s">medido, no supuesto</div></div>
      <div class="card tile"><div class="k">κ · maxDD</div><div class="v">${w.kappa_maxdd}</div><div class="s">${w.kappa_maxdd===0?'desactivado por la medición':'medido, no supuesto'}</div></div>
      <div class="card tile"><div class="k">ganadora única</div><div class="v">${C.n_winners===1?'sí':'no'}</div>
        <div class="s">misma config en los ${C.points.length} puntos</div></div>
      <div class="card tile"><div class="k">λ implícito</div><div class="v">${fmt(co.implied_lambda_median,1)}</div>
        <div class="s">el que ya paga la curva de equity</div></div>
    </div>
    <p class="lead"><b>Rank IC medio</b> por combinación: correlación de rangos entre el ranking in-sample y el out-of-sample
      de las ${C.n_configs} configuraciones. Más alto = la elección sobrevive al salir de muestra. En negrita, los pesos fijados.</p>
    <div class="card"><div class="tblwrap"><table><thead><tr><th>&nbsp;</th>${head}</tr></thead><tbody>${rows}</tbody></table></div></div>
    <div class="note"><b>1 · Los pesos no cambian la decisión.</b> En los ${C.points.length} puntos de la rejilla —de no penalizar nada a
      penalizar ocho veces más que antes— gana <b>siempre la misma configuración</b>, y también al repetir el barrido sólo con las
      configuraciones que operan de verdad. En el rango medido, λ y κ no arbitran nada: quien decide es el Sharpe.</div>
    <div class="note"><b>2 · Penalizar no estabiliza: degrada un poco.</b> No hay punto dulce. El rank IC es <b>máximo sin penalizar</b>
      (${fmt(nu.rank_ic_mean,3)} ± ${fmt(nu.rank_ic_se,3)}) y baja de forma <b>monótona</b> al subir cualquiera de los dos pesos.
      Los pesos anteriores (λ=0,5; κ=1,0) costaban <b>${fmt(Math.abs(C.prev.rank_ic_gain),3)} ± ${fmt(C.prev.rank_ic_gain_se,3)}</b> de rank IC
      frente a no penalizar —un 17% del nivel de la señal—, y el gap train-validation se movía de ${fmt(nu.selection_gap_norm,2)} a
      ${fmt(C.prev.selection_gap_norm,2)}. El término de drawdown es el que más cuesta, lo cual encaja: el maxDD es el estadístico más
      ruidoso de una curva de equity, la misma objeción que retiró al Calmar.</div>
    <div class="note"><b>3 · ¿Se cobra dos veces la rotación? No — al contrario.</b> La curva de equity ya paga
      <span class="mono">fee_rate + slippage = ${(co.cost_rate*100).toFixed(3)}%</span> de cada notional rotado, en las dos patas.
      En unidades de λ eso es <span class="mono">cost_rate × 365 / σ</span> ≈ <b>${fmt(co.implied_lambda_median,1)}</b>
      (IQR ${fmt(co.implied_lambda_p25,1)}–${fmt(co.implied_lambda_p75,1)}: depende de la volatilidad de cada configuración, no es una constante).
      La penalización explícita λ=${w.lambda_turnover} añade un <b>${fmt(C.share_pct,0)}%</b> sobre lo ya pagado: es un margen de seguridad, no una segunda factura.
      Control de que la cadena es real: las comisiones efectivamente cobradas sobre el notional reconstruido desde el turnover dan
      <span class="mono">${(co.measured_fee_rate*100).toFixed(4)}%</span>, que reproduce el <span class="mono">fee_rate</span> configurado.</div>
    <div class="note"><b>Por qué λ no es 0 aunque la evidencia lo prefiera.</b> El headline se comprometió a que rotar más sobre la
      <i>misma</i> curva de equity puntúe peor; con λ=0 esa puerta se reabre. Se fija el menor valor no nulo de la rejilla, cuyo precio medido
      (${fmt(Math.abs(ch.rank_ic_gain),3)} ± ${fmt(ch.rank_ic_gain_se,3)}) es indistinguible de cero entre las configuraciones activas.
      κ, en cambio, no lo exige ninguna propiedad del diseño: el mecanismo sigue disponible pasando <span class="mono">kappa_maxdd</span>,
      pero ya no se cobra por defecto sin que nadie lo pida.</div>
    <div class="note"><b>Límites.</b> Un solo corte 70/30 por muestra, un camino por escenario y ${C.n_configs} configuraciones, sobre un
      sustrato de rotación baja. Sirve para descartar que penalizar fuerte ayude; no para afinar decimales ni para extrapolar a costes más
      duros. Cuando aterrice la línea C, hay que repetirlo: re-analizar cuesta segundos porque los componentes están cacheados.</div>
    <p class="tag">Evidencia completa: <span class="mono">data/calibration/report_${esc(C.library)}.json</span> ·
      reproducible con <span class="mono">python -m ai_trader.scoring.weight_study</span> (${esc(C.generated_at)}).</p>`;
}

function renderRanking(){
  const host=$('#ranking'),R=D.ranking,sc=R.scope||{},W=sc.weights||{},G=R.gate||{},OV=R.overfit||{};
  const bl=R.baselines||[];
  const statCols=`<th class="num">CVaR@25%</th><th class="num">media</th><th class="num">P25</th><th class="num">std</th><th class="num">peor</th><th class="num">mejor</th><th class="num">n</th>`;
  const statCells=r=>`<td class="num"><b>${fmt(r.cvar25)}</b></td><td class="num">${fmt(r.mean)}</td><td class="num">${fmt(r.p25)}</td>
      <td class="num">${fmt(r.std)}</td><td class="num">${fmt(r.worst)}</td><td class="num">${fmt(r.best)}</td><td class="num">${r.n}</td>`;
  const pbo=OV.pbo||{},dsr=OV.dsr||{};
  host.innerHTML=`
    <h1>Ranking de estrategias</h1>
    <p class="lead">La unidad de evaluación es la DISTRIBUCIÓN sobre muestras (no un path). La puntuación por muestra es el
      <b>headline out-of-sample</b> = <span class="mono">Sharpe − ${W.lambda_turnover ?? 'λ'}·turnover${W.kappa_maxdd?' − '+W.kappa_maxdd+'·maxDD':''}</span>${W.kappa_maxdd===0?' (el término de maxDD queda en 0 por la calibración medida, más abajo)':''},
      y el ranking es el <b>CVaR@25%</b> de esa distribución (media del peor cuartil: se compite por la cola mala, no por el centro).</p>
    ${calibrationPanel(D.calibration)}
    <div class="note"><b>Muestra reducida.</b> ${sc.n_scenarios} escenarios × ${sc.n_paths} paths sobre <b>${esc(sc.library)}</b>,
      universo de ${sc.universe?sc.universe.length:'?'} activos, ventana ${sc.window_days} días. Para ampliar el scope,
      edita las constantes <span class="mono">RANK_*</span> en <span class="mono">dashboard/build_dashboard.py</span> y regenera.</div>
    <div class="note"><b>Cómo se anualiza el Sharpe.</b> Por número de observaciones al año, y eso depende del mercado:
      cripto cotiza 24/7 (365 barras) y la renta variable solo en sesión (252). Anualizar acciones por 365 inflaba su
      Sharpe y su volatilidad un 20% (<span class="mono">√(365/252)=1,204</span>). El factor lo fija el <b>universo</b>
      —365 en cuanto hay un activo 24/7, porque el backtest recorre la unión de días con barra— y se aplica igual a la
      estrategia y a sus baselines: comparar dos Sharpe con escalas distintas no significaría nada. Este universo mezcla
      cripto y renta variable, así que todo lo de esta vista está anualizado por <b>365</b>; las métricas lo reportan en
      <span class="mono">periods_per_year</span>. El CAGR es aparte: vive en tiempo de calendario (365 días naturales
      para toda clase de activo).</div>
    ${(R.rows&&R.rows.length)?`
    <h2>Estrategias</h2>
    <p class="lead">La columna <b>gate</b> es el veredicto: una estrategia solo <i>aprueba</i> si su CVaR@25% supera al del
      <b>mejor baseline pasivo</b> sobre las mismas muestras. <b>margen</b> es cuánto lo supera; <b>gana</b>, en qué porcentaje
      de muestras bate al mejor rival de ese mundo concreto.</p>
    <div class="card"><div class="tblwrap"><table><thead><tr><th>#</th><th>Estrategia</th><th>gate</th><th class="num">margen</th><th class="num">gana</th>${statCols}</tr></thead><tbody>
    ${R.rows.map((r,i)=>`<tr><td class="num">${i+1}</td><td><b>${esc(r.label)}</b> <span class="pill">${esc(r.type)}</span></td>
      <td><span class="chip ${r.approved?'hecho':'pend'}">${r.approved?'aprueba':'no aprueba'}</span></td>
      <td class="num">${fmt(r.margin)}</td><td class="num">${fmt(r.win_rate_pct,0)}%</td>${statCells(r)}</tr>`).join('')}
    </tbody></table></div></div>
    ${bl.length?`
    <h2>Baselines: lo que consigue no hacer nada</h2>
    <p class="lead">Mismas muestras, misma ventana out-of-sample y las mismas comisiones y slippage que paga la estrategia.
      Comprar y mantener no es gratis, y por eso es un rival honesto.</p>
    <div class="card"><div class="tblwrap"><table><thead><tr><th>Baseline</th><th class="num">activos</th>${statCols}</tr></thead><tbody>
    ${bl.map(b=>`<tr><td><b>${esc(b.label)}</b> ${b.name===G.best_baseline?'<span class="pill">mejor</span>':''}</td>
      <td class="num">${b.symbols}</td>${statCells(b)}</tr>`).join('')}
    </tbody></table></div></div>
    ${(G.missing&&G.missing.length)?`<div class="note"><b>Baselines no disponibles en este scope:</b>
      <span class="mono">${G.missing.map(esc).join(', ')}</span>. No se sustituyen por nada: si un rival no se puede construir, se dice.</div>`:''}`:''}
    ${costPanel(D.costs)}
    ${pbo.computable||dsr.computable?`
    <h2>Descuento por múltiples pruebas</h2>
    <p class="lead">Probar muchas configuraciones garantiza encontrar una que brilla aunque no haya nada que encontrar.
      Estas dos cifras ponen número a eso sobre la distribución de scores de este mismo ranking.</p>
    <div class="grid cards">
      <div class="card"><h3>PBO <span class="pill">${fmt((pbo.pbo??0)*100,0)}%</span></h3>
        <p class="tag">Probability of Backtest Overfitting por CSCV: en qué fracción de las ${pbo.n_splits||0} particiones
          train/test la ganadora in-sample cae por debajo de la mediana fuera de muestra. 50% es tirar una moneda;
          cerca de 0% significa que elegir por backtest acierta. ${pbo.n_trials||0} configuraciones, ${pbo.n_blocks||0} bloques.</p></div>
      <div class="card"><h3>DSR <span class="pill">${fmt((dsr.dsr??0)*100,0)}%</span></h3>
        <p class="tag">Deflated Sharpe Ratio de <b>${esc(OV.winner||'-')}</b>: probabilidad de que su Sharpe verdadero sea &gt; 0
          una vez descontado el máximo esperado por azar con ${dsr.n_trials||0} intentos
          (umbral deflactado: ${fmt(dsr.expected_max_sharpe)}; Sharpe observado: ${fmt(dsr.observed_sharpe)}).</p></div>
    </div>`:''}
    <h2>Distribución detrás del ranking</h2>
    <p class="lead">Cada punto es el headline OOS de una muestra; la barra vertical es la mediana. La cola izquierda es lo que
      mide el CVaR — y los baselines aparecen en la misma escala para que la comparación se vea, no se afirme.</p>
    <div class="card"><div id="distchart"></div></div>`
    :`<div class="card"><p class="tag">Ranking no disponible (¿falta ai_v2 en disco? corre <span class="mono">build_dashboard.py</span>).</p></div>`}`;
  if(R.rows&&R.rows.length){
    const pal=[cssv('--s1'),cssv('--s2'),cssv('--s4'),cssv('--s7'),cssv('--s6'),cssv('--s5')];
    const groups=R.rows.map((r,i)=>({label:r.label,values:(R.distributions[r.label]||[]),color:pal[i%pal.length]}));
    bl.forEach(b=>groups.push({label:b.label,values:(R.distributions[b.name]||[]),color:cssv('--muted')}));
    dotStrip($('#distchart'),groups.filter(g=>g.values.length));
  }
}

function usd(v){
  if(v>=1e9)return '$'+fmt(v/1e9,1)+' B';
  if(v>=1e6)return '$'+fmt(v/1e6,1)+' M';
  if(v>=1e3)return '$'+fmt(v/1e3,0)+' k';
  return '$'+fmt(v,0);
}

function costPanel(C){
  if(!C||!C.rows||!C.rows.length)return '';
  const sizes=C.sizes_usd||[];
  const head=sizes.map(s=>`<th class="num">${usd(s)}</th>`).join('');
  return `
    <h2>Lo que cuesta ejecutar</h2>
    <p class="lead">El deslizamiento <b>no es una constante</b>. Cada fill paga
      <span class="mono">medio spread del símbolo + volatilidad reciente + impacto</span>, y el impacto sigue la
      <b>ley de raíz cuadrada</b> sobre la fracción del volumen de la barra que consume la orden: cuadruplicar el tamaño
      duplica el coste. Cifras calculadas sobre barras reales de <span class="mono">${esc(C.library)}</span>, en puntos básicos.</p>
    <div class="card"><div class="tblwrap"><table>
      <thead><tr><th>Símbolo</th><th class="num">spread base</th><th class="num">vol. diaria</th>${head}<th class="num">capacidad / barra</th></tr></thead>
      <tbody>${C.rows.map(r=>`<tr>
        <td><b>${esc(r.symbol)}</b></td>
        <td class="num">${fmt(r.spread_bps,1)} pb</td>
        <td class="num">${fmt(r.vol_pct,1)}%</td>
        ${(r.slippage_bps||[]).map(b=>`<td class="num">${fmt(b,1)}</td>`).join('')}
        <td class="num">${usd(r.capacity_usd)}</td></tr>`).join('')}
      </tbody></table></div></div>
    <div class="note"><b>Cómo leerlo.</b> A tamaño pequeño manda el <b>spread</b>: el orden lo fija el símbolo, y ahí un
      altcoin cuesta ya varias veces lo que BTC. A tamaño grande manda el <b>impacto</b>, y la separación se dispara.
      La última columna es el techo de capacidad (${fmt((C.max_participation||0)*100,0)}% del volumen de la barra):
      por encima de esa cifra la orden se llena <b>parcialmente</b>, no entera. Con capitales de cinco cifras ese techo
      no llega a morder — es justo lo que hace que un resultado deje de escalar cuando el capital crece.</div>`;
}

function renderValidation(){
  const host=$('#validation'),V=D.validation;
  if(!V){
    host.innerHTML=`<h1>Validación temporal</h1>
      <p class="lead">El estudio comparativo aún no está publicado.</p>
      <div class="note">Genéralo con <span class="mono">python -m ai_trader.scoring.validation_study</span>.</div>`;
    return;
  }
  const o=V.optimism,dsp=V.dispersion,ra=V.rank_agreement,fl=V.flips,g=V.geometry,svn=V.svn;
  const ex=V.example;

  host.innerHTML=`
    <h1>Validación temporal</h1>
    <p class="lead">Un backtest se puede partir en train y test de muchas formas, y la forma elegida
      <b>cambia la respuesta</b>. Hasta ahora cada muestra se partía con un único corte 70/30: un solo
      número out-of-sample, sin cola, sin dispersión y con el día del corte cayendo en las dos ventanas.
      Aquí ese corte convive con dos esquemas que evalúan <b>varias ventanas</b> con purga y embargo, y
      se mide qué diferencia hay — sobre ${V.n_units} unidades (${V.n_configs} configuraciones ×
      ${V.n_samples} muestras) de <span class="mono">${esc(V.library)}</span>.</p>

    <div class="grid tiles">
      <div class="card tile"><div class="k">sesgo de nivel</div>
        <div class="v">${o.walk_forward.median>=0?'+':''}${fmt(o.walk_forward.median,2)}</div>
        <div class="s">vs mediana walk-forward: ninguno</div></div>
      <div class="card tile"><div class="k">vs la cola (CVaR)</div>
        <div class="v">${o.vs_tail.median>=0?'+':''}${fmt(o.vs_tail.median,2)}</div>
        <div class="s">el estadístico que rankea</div></div>
      <div class="card tile"><div class="k">ruido temporal / señal</div>
        <div class="v">×${fmt(svn.ratio,1)}</div>
        <div class="s">mover la ventana pesa más que cambiar de estrategia</div></div>
      <div class="card tile"><div class="k">cambia la elección</div>
        <div class="v">${fl.walk_forward}/${fl.n_samples}</div>
        <div class="s">muestras donde gana otra config</div></div>
    </div>
    <div class="note"><b>El resultado no es el que se esperaba, y se reporta como salió.</b> La hipótesis
      de partida —"el corte único sobre-estima la robustez"— era que puntuaría <i>sistemáticamente</i> más
      alto. <b>No es así:</b> frente a la mediana de las ventanas honestas el sesgo es
      ${fmt(o.walk_forward.median,3)}, indistinguible de cero, y el rango va de ${fmt(o.walk_forward.min,2)}
      a +${fmt(o.walk_forward.max,2)}: el corte único es tan capaz de regalar como de castigar, según dónde
      caiga. Lo que sí sostiene la medición son las otras tres columnas de abajo, y la tercera es la que
      importa para decidir.</div>

    <h2>La geometría, dibujada</h2>
    <p class="lead">Cada fila es un fold; el eje es el rango completo de la muestra
      (${esc(g.start)} → ${esc(g.end)}, ${g.span_days} días). <span class="swatch" style="background:var(--s1)"></span>
      train · <span class="swatch" style="background:var(--s2)"></span> test out-of-sample. El hueco
      <b>entre</b> el azul y el verde es la <b>purga</b> (${V.purge_days} días, exactamente lo que puede
      seguir viva una posición abierta el último día de train); el hueco a la derecha del verde es el
      <b>embargo</b> (${V.embargo_days} días, el eco serial de los retornos).</p>
    <div class="grid cards">
      <div class="card"><h3>Corte único 70/30 <span class="chip pend">1 ventana</span></h3>
        <div id="geo_single"></div>
        <p class="tag">Una sola ventana OOS, al final. Todo lo que se sepa del sistema sale del último 30% del rango.</p></div>
      <div class="card"><h3>Walk-forward <span class="chip hecho">${V.n_folds_wf} ventanas</span></h3>
        <div id="geo_wf"></div>
        <p class="tag">Entrena con el pasado y se puntúa en el tramo siguiente, avanzando. Es lo que hace un operador real: no tira historia.</p></div>
      <div class="card"><h3>CPCV <span class="chip hecho">${V.n_folds_cpcv} ventanas</span></h3>
        <div id="geo_cpcv"></div>
        <p class="tag">Todas las combinaciones de ${V.n_test_groups} de ${V.n_groups} grupos como test. Cada tramo se evalúa acompañado de contextos distintos.</p></div>
    </div>

    <h2>La misma evidencia, tres lecturas</h2>
    <p class="lead">Cada punto es una de las ${V.n_units} unidades (una configuración sobre una muestra), y las tres filas
      son <b>los mismos</b> backtests leídos con los tres esquemas. La marca vertical es la mediana de cada fila: lo que
      separa la fila roja de las otras dos es el optimismo del corte único.</p>
    <div class="card"><div id="paireddist"></div></div>

    ${ex?`<h2>Una muestra, tres respuestas</h2>
    <p class="lead">La misma configuración (<span class="mono">${esc(ex.config_id)}</span>) sobre la misma muestra
      (<span class="mono">${esc(ex.scenario_id)}</span>), la de optimismo más cercano a la mediana del estudio —no el caso
      más favorable—. Cada punto es una ventana; la marca vertical es la mediana.</p>
    <div class="card"><div id="exdist"></div></div>
    <p class="tag">El corte único entrega el punto único de arriba. Los otros dos entregan una distribución: de ahí salen la cola y la dispersión.</p>`:''}

    <h2>Qué cambia, en orden de importancia</h2>
    <div class="grid cards">
      <div class="card"><h3>1 · La cola no existía</h3>
        <p class="lead">El corte único puntúa <b>+${fmt(o.vs_tail.median,3)}</b> por encima del CVaR@25% de las ventanas
          (IQR ${fmt(o.vs_tail.p25,2)} … ${fmt(o.vs_tail.p75,2)}), y es positivo en más de tres de cada cuatro unidades.
          No es un sesgo del corte: es que <b>el CVaR de un solo número es ese número</b>. El sistema rankea por la cola
          mala y, con una ventana, no había cola mala que promediar. Ésta es la brecha estructural.</p></div>
      <div class="card"><h3>2 · El ruido temporal supera a la señal</h3>
        <p class="lead">Entre ventanas de una misma muestra el headline se mueve una desviación típica de
          <b>${fmt(dsp.walk_forward_std.median,3)}</b>, con un rango mediano de <b>${fmt(dsp.walk_forward_range.median,2)}</b> puntos
          entre la mejor y la peor. En la misma muestra, lo que separa a la mejor de la peor <b>configuración</b> es
          <b>${fmt(svn.config_spread_walk_forward.median,2)}</b>: mover la ventana mueve el resultado <b>${fmt(svn.ratio,1)} veces más</b>
          que cambiar de estrategia. Ése es el argumento entero — no hace falta que el corte único esté sesgado para que sea
          una mala regla de decisión, basta con que sea arbitrario.</p></div>
      <div class="card"><h3>3 · La elección cambia</h3>
        <p class="lead">Lo que decide no es el nivel, sino el orden. El acuerdo de rangos entre ordenar configuraciones por el
          corte único y ordenarlas por la recompensa multiventana tiene mediana <b>${fmt(ra.walk_forward.median,2)}</b> pero
          media <b>${fmt(ra.walk_forward.mean,2)}</b> y llega a <b>${fmt(ra.walk_forward.min,1)}</b> (orden invertido). La
          configuración ganadora cambia en <b>${fl.walk_forward}/${fl.n_samples}</b> muestras (${fmt(fl.walk_forward_pct,0)}%)
          con walk-forward y <b>${fl.cpcv}/${fl.n_samples}</b> (${fmt(fl.cpcv_pct,0)}%) con CPCV.</p></div>
    </div>

    <div class="note"><b>Lo que la purga NO hace, dicho claro.</b> Dentro de un backtest no se ajusta nada: la configuración
      entra fija y el motor construye reloj, estado y estrategias nuevos por ventana, así que el train de un fold no influye en su
      test. Purgar y embargar no mejoran —ni empeoran— ninguna cifra out-of-sample de las de arriba, y hay un test que lo fija como
      invariante. Sirven para otra cosa: que la referencia in-sample no esté contaminada por operaciones que siguen vivas dentro del
      test, y que la geometría ya sea correcta cuando algo <b>sí</b> se ajuste sobre el train (el CEM, o una política aprendida).
      Lo que aporta valor <i>hoy</i> es tener varias ventanas OOS en vez de una.</div>

    <div class="note"><b>Límites declarados, y son serios.</b> ${V.n_units} unidades sobre ${V.n_configs} configuraciones y
      ${V.n_samples} muestras de una sola librería, un camino por escenario. El acuerdo de rangos y los cambios de elección se miden
      sobre <b>${fl.n_samples} muestras</b>: "4 de 8" es una señal, no una tasa — el intervalo de confianza de esa proporción cubre
      medio rango. Las ventanas de un mismo esquema comparten historia (en CPCV cada tramo entra en ${fmt(V.n_folds_cpcv/V.n_groups*V.n_test_groups,0)}
      folds), así que la dispersión medida <b>no</b> son observaciones independientes y no debe leerse como un error estándar. Y el
      scoring que usa el optimizador sigue puntuando con el corte único: cablearlo es la evolución pendiente de la línea D.</div>
    <div class="note"><b>Auditoría de fuga.</b> Los ${V.leakage.folds_audited} folds ejecutados en este estudio pasaron la
      comprobación de fuga temporal (${V.leakage.clean?'ninguno con solape ni con purga o embargo insuficientes':'⚠ alguno falló'}).
      No es una comprobación de adorno: el motor la corre <b>antes</b> de gastar cómputo y aborta el plan entero si falla.</div>

    <p class="tag">Evidencia completa: <span class="mono">data/validation/report_${esc(V.library)}.json</span> ·
      reproducible con <span class="mono">python -m ai_trader.scoring.validation_study</span> (${esc(V.generated_at)}) ·
      un backtest suelto: <span class="mono">ai-trader backtest --validation cpcv</span>.</p>`;

  foldStrip($('#geo_single'),g.single_split);
  foldStrip($('#geo_wf'),g.walk_forward);
  foldStrip($('#geo_cpcv'),g.cpcv);
  const pd=V.paired||{};
  dotStrip($('#paireddist'),[
    {label:'corte único 70/30',values:pd.single||[],color:cssv('--s8')},
    {label:'walk-forward (mediana)',values:pd.walk_forward||[],color:cssv('--s1')},
    {label:'CPCV (mediana)',values:pd.cpcv||[],color:cssv('--s5')},
  ],{pad:{t:10,r:16,b:24,l:170}});
  if(ex){
    dotStrip($('#exdist'),[
      {label:'corte único 70/30',values:[ex.single],color:cssv('--s8')},
      {label:`walk-forward (${ex.walk_forward.length})`,values:ex.walk_forward,color:cssv('--s1')},
      {label:`CPCV (${ex.cpcv.length})`,values:ex.cpcv,color:cssv('--s5')},
    ],{pad:{t:10,r:16,b:24,l:170}});
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
  renderOverview();renderSynthetic();renderFidelity();renderStrategies();
  renderRanking();renderValidation();renderPaper();renderRoadmap();
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
      <li><button data-sec="fidelity">Fidelidad</button></li>
      <li><button data-sec="strategies">Estrategias</button></li>
      <li><button data-sec="ranking">Ranking</button></li>
      <li><button data-sec="validation">Validación</button></li>
      <li><button data-sec="paper">Paper trading</button></li>
      <li><button data-sec="roadmap">Evoluciones</button></li>
    </ul>
  </aside>
  <main class="main">
    <button id="themeBtn" class="btn ghost toggle">☾ Oscuro</button>
    <section id="overview" class="section active"></section>
    <section id="synthetic" class="section"></section>
    <section id="fidelity" class="section"></section>
    <section id="strategies" class="section"></section>
    <section id="ranking" class="section"></section>
    <section id="validation" class="section"></section>
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
