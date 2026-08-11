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
/* --- evoluciones: lista ordenada por criticidad --- */
.rmgroup{margin:30px 0 0;padding-top:16px;border-top:1px solid var(--border)}
.rmgroup:first-of-type{border-top:0;padding-top:0;margin-top:18px}
.rmgroup h2{margin:0 0 2px}
.rmgroup .sub{color:var(--ink2);font-size:13px;margin:0 0 14px;max-width:82ch}
.rmlist{display:flex;flex-direction:column;gap:12px}
.rmitem{--pc:var(--muted);display:grid;grid-template-columns:52px 1fr;gap:14px;
  background:var(--surface);border:1px solid var(--border);border-left:4px solid var(--pc);
  border-radius:12px;padding:14px 16px}
.rmitem.critica{--pc:var(--crit)}
.rmitem.alta{--pc:var(--serious)}
.rmitem.media{--pc:var(--warn)}
.rmitem.baja{--pc:var(--s1)}
.rmitem.aparcada{--pc:var(--muted)}
.rmrank{text-align:center;line-height:1.05;padding-top:2px}
.rmrank b{font-size:24px;font-weight:700;color:var(--pc);font-variant-numeric:tabular-nums}
.rmrank small{display:block;margin-top:3px;font-size:10px;font-weight:600;color:var(--muted);
  text-transform:uppercase;letter-spacing:.04em}
.rmhead{display:flex;gap:10px;align-items:baseline;justify-content:space-between;flex-wrap:wrap}
.rmhead h3{margin:0;font-size:15.5px}
.rmev{border-left:3px solid var(--pc);padding:3px 0 3px 11px;margin:9px 0;color:var(--ink2);font-size:12.5px}
.rmwhy{color:var(--ink2);margin:4px 0 0;max-width:82ch}
details.pr{margin:10px 0 0}
details.pr summary{cursor:pointer;font-size:12.5px;font-weight:600;color:var(--s1);list-style:none;padding:3px 0}
details.pr summary::-webkit-details-marker{display:none}
details.pr summary:before{content:'▸ '}
details.pr[open] summary:before{content:'▾ '}
details.pr .prompt{max-height:340px}
.chip.critica{background:color-mix(in srgb,var(--crit) 20%,transparent);color:var(--crit)}
.chip.alta{background:color-mix(in srgb,var(--serious) 30%,transparent);color:var(--serious)}
.chip.media{background:color-mix(in srgb,var(--warn) 26%,transparent);color:var(--serious)}
.chip.baja{background:color-mix(in srgb,var(--s1) 16%,transparent);color:var(--s1)}
.chip.bloqueada{background:color-mix(in srgb,var(--s7) 20%,transparent);color:var(--s7)}
.chip.aparcada{background:color-mix(in srgb,var(--muted) 22%,transparent);color:var(--ink2)}
@media(max-width:820px){.app{grid-template-columns:1fr}.side{position:static;height:auto}.split{grid-template-columns:1fr}
  .rmitem{grid-template-columns:1fr}.rmrank{text-align:left;display:flex;gap:8px;align-items:baseline}
  .rmrank small{margin:0}}
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

function rankScatter(host, points, opts={}){
  // points: [{name,family,rank_real,rank_synth,delta,...}] con rango 1 = mejor.
  // Se dibuja n+1-rango en los dos ejes para que ARRIBA y DERECHA sea "mejor", que es
  // como se lee un ranking sin tener que pensarlo. La diagonal es el acuerdo perfecto:
  // cuanto mas lejos cae un punto de ella, mas discrepan los dos mundos sobre el.
  opts=Object.assign({h:400,pad:{t:16,r:20,b:44,l:60},n:16},opts);
  const draw=()=>{
    host.innerHTML='';
    const W=host.clientWidth||680,H=opts.h,p=opts.pad,n=opts.n;
    const lo=0.4,hi=n+0.6;
    const X=v=>p.l+(W-p.l-p.r)*(v-lo)/(hi-lo);
    const Y=v=>p.t+(H-p.t-p.b)*(1-(v-lo)/(hi-lo));
    const svg=el('svg',{viewBox:`0 0 ${W} ${H}`,width:'100%',height:H});
    const step=n>10?4:2;
    for(let r=1;r<=n;r+=step){
      const v=n+1-r;
      svg.appendChild(el('line',{x1:p.l,y1:Y(v),x2:W-p.r,y2:Y(v),stroke:cssv('--grid'),'stroke-width':1}));
      svg.appendChild(el('line',{x1:X(v),y1:p.t,x2:X(v),y2:H-p.b,stroke:cssv('--grid'),'stroke-width':1}));
      const ty=el('text',{x:p.l-7,y:Y(v)+3,'text-anchor':'end',fill:cssv('--muted'),'font-size':10});
      ty.textContent=r;svg.appendChild(ty);
      const tx=el('text',{x:X(v),y:H-p.b+15,'text-anchor':'middle',fill:cssv('--muted'),'font-size':10});
      tx.textContent=r;svg.appendChild(tx);
    }
    svg.appendChild(el('line',{x1:X(lo),y1:Y(lo),x2:X(hi),y2:Y(hi),stroke:cssv('--axis'),
      'stroke-width':1.5,'stroke-dasharray':'5 4'}));
    const dl=el('text',{x:W-p.r-6,y:Y(hi)+14,'text-anchor':'end',fill:cssv('--muted'),'font-size':10});
    dl.textContent='acuerdo perfecto';svg.appendChild(dl);
    // Banda de la "mitad buena" del real: es donde un pre-cribado tiene que acertar.
    const half=Math.ceil(n/2), edge=n+1-half;
    svg.appendChild(el('rect',{x:X(edge-0.5),y:p.t,width:Math.max(1,X(hi)-X(edge-0.5)),
      height:H-p.t-p.b,fill:cssv('--s5'),opacity:.07}));
    const bl=el('text',{x:X(hi)-6,y:p.t+13,'text-anchor':'end',fill:cssv('--muted'),'font-size':10});
    bl.textContent='mitad buena del real';svg.appendChild(bl);
    const xl=el('text',{x:(p.l+W-p.r)/2,y:H-6,'text-anchor':'middle',fill:cssv('--ink2'),'font-size':11});
    xl.textContent=opts.xlab||'puesto en el ranking REAL (1 = mejor)';svg.appendChild(xl);
    const cy=(p.t+H-p.b)/2;
    const yl=el('text',{x:14,y:cy,'text-anchor':'middle',fill:cssv('--ink2'),'font-size':11,
      transform:`rotate(-90 14 ${cy})`});
    yl.textContent=opts.ylab||'puesto en el ranking SINTÉTICO (1 = mejor)';svg.appendChild(yl);
    const tt=tip();
    points.forEach(pt=>{
      const x=X(n+1-pt.rank_real), y=Y(n+1-pt.rank_synthetic);
      const color=pt.family==='mean_reversion'?cssv('--s7'):cssv('--s1');
      // Los desacuerdos grandes se unen a la diagonal con un hilo: el largo del hilo ES
      // el error de ordenacion, y su signo dice si el sintetico sobrevalora o infravalora.
      if(Math.abs(pt.delta)>=Math.floor(n/4)){
        const d=Y(n+1-pt.rank_real);
        svg.appendChild(el('line',{x1:x,y1:y,x2:x,y2:d,stroke:color,'stroke-width':1.2,
          opacity:.35,'stroke-dasharray':'2 3'}));
      }
      svg.appendChild(el('circle',{cx:x,cy:y,r:6,fill:color,stroke:cssv('--surface'),'stroke-width':2}));
      const hit=el('circle',{cx:x,cy:y,r:15,fill:'transparent'});
      hit.addEventListener('mousemove',ev=>{
        tt.style.display='block';tt.style.left=(ev.clientX+12)+'px';tt.style.top=(ev.clientY-10)+'px';
        tt.innerHTML=`<b>${esc(pt.config_id)}</b><br>real: puesto ${pt.rank_real} `+
          `(CVaR ${fmt(pt.reward_real,2)})<br>sintético: puesto ${pt.rank_synthetic} `+
          `(CVaR ${fmt(pt.reward_synthetic,2)})<br>`+
          (pt.delta===0?'coinciden':(pt.delta>0
            ?`el sintético la <b>sobrevalora</b> en ${pt.delta} puestos`
            :`el sintético la <b>infravalora</b> en ${-pt.delta} puestos`));
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

function stackedBars(host, rows, opts={}){
  // rows: [{label, sub, parts:[{name,value,color}]}] -> barras horizontales apiladas al 100%.
  // Es la forma correcta para cuotas que suman 1: el ojo compara segmentos contiguos sobre
  // una longitud fija, no alturas sueltas, y el total queda garantizado por construcción.
  // `opts.refs` dibuja las referencias de RELOJ (dónde caería cada corte si la actividad
  // fuese uniforme): sin ellas, una cuota grande no se distingue de una sesión larga.
  opts=Object.assign({rowH:38,pad:{t:10,r:16,b:26,l:104},refs:[]},opts);
  const draw=()=>{
    host.innerHTML='';
    const W=host.clientWidth||680,p=opts.pad,H=p.t+p.b+opts.rowH*rows.length;
    const X=f=>p.l+(W-p.l-p.r)*f;
    const svg=el('svg',{viewBox:`0 0 ${W} ${H}`,width:'100%',height:H});
    for(let g=0;g<=4;g++){const x=X(g/4);
      svg.appendChild(el('line',{x1:x,y1:p.t-2,x2:x,y2:H-p.b+2,stroke:cssv('--grid'),'stroke-width':1}));
      const t=el('text',{x,y:H-p.b+16,'text-anchor':'middle',fill:cssv('--muted'),'font-size':10});
      t.textContent=(g*25)+'%';svg.appendChild(t);}
    const tt=tip();
    rows.forEach((r,i)=>{
      const y=p.t+opts.rowH*i, h=opts.rowH-14;
      const lab=el('text',{x:p.l-9,y:y+h/2+1,'text-anchor':'end',fill:cssv('--ink'),'font-size':12});
      lab.textContent=r.label;svg.appendChild(lab);
      if(r.sub){const s=el('text',{x:p.l-9,y:y+h/2+13,'text-anchor':'end',fill:cssv('--muted'),'font-size':10});
        s.textContent=r.sub;svg.appendChild(s);}
      let acc=0;
      r.parts.forEach(part=>{
        const v=part.value||0, x0=X(acc), x1=X(acc+v);
        const rect=el('rect',{x:x0,y,width:Math.max(0.5,x1-x0),height:h,fill:part.color,opacity:.92});
        rect.addEventListener('mousemove',ev=>{tt.style.display='block';tt.style.left=(ev.clientX+12)+'px';
          tt.style.top=(ev.clientY-10)+'px';
          tt.innerHTML=`<b>${esc(r.label)}</b><br><span style="color:${part.color}">■</span> ${esc(part.name)}: <b>${fmt(100*v,1)}%</b>`
            +(part.note?`<br><span style="opacity:.75">${esc(part.note)}</span>`:'');});
        rect.addEventListener('mouseleave',()=>tt.style.display='none');
        svg.appendChild(rect);
        if(x1-x0>36){const t=el('text',{x:(x0+x1)/2,y:y+h/2+4,'text-anchor':'middle',fill:'#fff','font-size':11,'font-weight':600});
          t.textContent=fmt(100*v,0)+'%';svg.appendChild(t);}
        acc+=v;});
    });
    // Referencias de reloj: encima de todas las barras, para leerlas de un vistazo.
    opts.refs.forEach(ref=>{
      svg.appendChild(el('line',{x1:X(ref.x),y1:p.t-2,x2:X(ref.x),y2:H-p.b+2,
        stroke:cssv('--ink'),'stroke-width':1.4,'stroke-dasharray':'3 3',opacity:.75}));
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
    ['Librerías sintéticas', (k.lineage||[]).length||'-', (k.lineage||[]).join(' → ')||'Monte Carlo'],
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
    ['D','Validación multiventana (CPCV/walk-forward): construida y medida','hecho'],
    ['E','Limpieza de consistencia (universo, anualización, diseñador)','hecho'],
    ['B','Hueco de fidelidad cerrado y aceptado: colas, clustering y correlación → ai_v3','hecho'],
    ['B/D','Transferencia de ranking real-vs-sintético: medida, y el sintético NO ordena','hecho'],
    ['Medición','La ventana ciega del backtest diario, medida por sesión horaria','hecho'],
    ['A','Suelo de actividad: el ranking ya no lo gana quien no opera','hecho'],
    ['D','Cablear el CPCV en el optimizador (juez en dos etapas)','pendiente'],
    ['Live','Paper trading corriendo en vivo (compra tiempo de calendario)','pendiente'],
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
    <p class="tag" style="margin-top:14px">Ve a <b>Evoluciones</b>: las pendientes están ordenadas de mayor a menor
      criticidad, con su evidencia medida y su prompt copiable. Foco en cripto; renta variable y mercados de
      predicción están aparcados a propósito.</p>`;
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
  const lineage=((D.kpis&&D.kpis.lineage)||Object.keys(f)).filter(l=>f[l]);
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
    <h2>Tres generaciones del mismo mundo</h2>
    <p class="lead"><b>ai_v1</b> era iid y gaussiano: la reversión a la media era inganable por construcción.
      <b>ai_v2</b> ganó colas, clustering y estructura serial con un retrofit determinista sobre los mismos
      <i>spec.json</i>. <b>ai_v3</b> ajusta la magnitud de esas propiedades contra el mercado real —colas
      también en la calma, más reacción a la noticia en el GARCH y cargas factoriales que suben en el
      pánico— hasta pasar el test de aceptación de la sección <b>Fidelidad</b>.</p>
    <div class="card"><div class="tblwrap"><table><thead><tr><th>Stylized fact</th>
      ${lineage.map(l=>`<th class="num">${esc(l)}${l==='ai_v1'?' (iid)':''}</th>`).join('')}
      <th>Qué significa</th></tr></thead><tbody>
    ${factRows.map(r=>{const vs=lineage.map(l=>f[l]?f[l][r[1]]:null);
      const mx=Math.max(...vs.map(v=>Math.abs(v||0)));
      return `<tr><td>${r[0]}</td>
      ${vs.map((v,j)=>`<td class="num">${fmt(v,r[2])} ${miniBar(v,mx,j?'':'v1')}</td>`).join('')}
      <td class="tag">${r[3]}</td></tr>`;}).join('')}
    </tbody></table></div></div>
    <div class="note"><b>Sustrato por defecto.</b> El harness de scoring sigue optimizando sobre
      <span class="mono">ai_v2</span> (<span class="mono">DEFAULT_LIBRARY_ID</span> en
      <span class="mono">src/ai_trader/scoring/optimize.py</span>) <b>a propósito</b>: la calibración de
      pesos y la validación multiventana se midieron sobre esa librería, y moverlo obligaría a recorrerlas
      antes de que sus cifras volvieran a significar algo. <span class="mono">ai_v3</span> existe y está
      medido; el cambio de sustrato es un paso separado, con su propio coste.</div>
    <div class="note"><b>Esta tabla compara el mundo sintético consigo mismo.</b> Que una librería tenga
      colas y agrupamiento no dice que los tenga <i>en la magnitud del mercado</i>. Esa pregunta se responde
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
  const B=F.before, prev=k=>(B&&B.by_key[k])||null;
  const A=F.acceptance, ok=A&&A.passed;
  const was=(k,f)=>{const p=prev(k);return p?f(p):'—';};
  const tiles=[
    ['Aceptación',ok?'cumple':'no cumple',
      A?('todos los umbrales declarados · cobertura mínima '+fmt(A.min_coverage_pct,0)+'%'):'sin umbrales'],
    ['Cobertura media',pct(S.coverage_mean_pct),
      B?('antes '+pct(B.summary.coverage_mean_pct)+' en '+esc(B.library)):'valores reales dentro del rango sintético'],
    ['Curtosis real / sint.',fmt(kurt.real_median,1)+' / '+fmt(kurt.synth_median,1),
      B?('antes '+was('excess_kurtosis',p=>fmt(p.synth_median,2))+' · el hueco que se cerró'):'el hueco más grande'],
    ['Muestras',S.n_real_windows+' / '+S.n_synthetic_samples,'ventanas reales / paths sintéticos'],
  ];
  host.innerHTML=`
    <h1>Fidelidad contra el mercado real</h1>
    <p class="lead">Un mercado sintético solo vale si se parece al real en sus propiedades estadísticas.
      Aquí se miden los mismos <i>stylized facts</i> sobre <b>${esc(F.library)}</b> y sobre el histórico
      diario real de ${esc(F.exchange)} (${esc(F.real_start)} → ${esc(F.real_end)}, ${S.n_symbols} criptos),
      y se comparan en tres ejes distintos: <b>nivel</b> (¿la magnitud es la del mercado?),
      <b>ordenación</b> (¿ordena los activos como el mercado?, correlación de rangos de Spearman) y
      <b>cobertura</b> (¿produce el ensemble sintético el valor real como una realización plausible?).
      ${B?`Cada cifra se enseña junto a la de <b>${esc(B.library)}</b>, el generador anterior: sin el antes,
      una corrección medida no se distingue de una afirmación.`:''}</p>
    <div class="grid tiles">${tiles.map(t=>`<div class="card tile"><div class="k">${t[0]}</div>
      <div class="v">${t[1]}</div><div class="s">${t[2]}</div></div>`).join('')}</div>
    ${A?`<h2>Test de aceptación</h2>
    <p class="lead">El estudio no publica cifras y ya: las contrasta con umbrales declarados en el código y
      <b>devuelve error si no se cumplen</b>. Un estudio que no puede fallar no es evidencia. Dos familias:
      <b>cobertura</b> (el valor real de cada activo cae dentro del [p10, p90] del ensemble en al menos
      ${fmt(A.min_coverage_pct,0)}% de los activos) y <b>mediana de mercado</b> (la mediana real de la sección
      cruzada cae dentro de la banda del ensemble entero) en los tres hechos que la corrección ataca.</p>
    <div class="card"><div class="tblwrap"><table>
      <thead><tr><th>Umbral</th><th>Qué exige</th><th class="num">medido</th><th></th></tr></thead>
      <tbody>${A.checks.map(c=>`<tr>
        <td>${esc(c.label)}</td>
        <td class="tag">${c.kind==='coverage'
          ? 'cobertura ≥ '+fmt(c.threshold,0)+'%'
          : 'mediana real dentro de '+(c.band?('['+fmt(c.band[0],3)+', '+fmt(c.band[1],3)+']'):'—')}</td>
        <td class="num">${c.kind==='coverage'?fmt(c.value,0)+'%':fmt(c.value,3)}</td>
        <td>${c.passed?'<span class="chip hecho">cumple</span>':'<span class="chip pend">falla</span>'}</td>
      </tr>`).join('')}</tbody></table></div></div>`:''}
    <div class="note"><b>Comparación pareada en tamaño de muestra.</b> El histórico real se trocea en
      ventanas de <b>${F.window_days} días</b> —el mismo horizonte que un camino sintético— porque la
      autocorrelación y sobre todo la curtosis están sesgadas en muestras cortas: medir el real sobre ocho
      años seguidos y el sintético sobre dos compararía el sesgo, no el mundo. Las ventanas avanzan
      ${F.step_days} días${F.overlap?' y por tanto <b>solapan</b>: sirven para la tendencia central, no como estimaciones independientes':''}.</div>
    <h2>Métrica a métrica</h2>
    <div class="card"><div class="tblwrap"><table>
      <thead><tr><th>Stylized fact</th><th class="num">real</th>
        ${B?`<th class="num">${esc(B.library)}</th>`:''}<th class="num">${esc(F.library)}</th>
        <th class="num">ratio</th><th class="num">rank corr</th>
        <th class="num">cobertura${B?' (antes)':''}</th></tr></thead>
      <tbody>${rows.map(r=>`<tr>
        <td>${esc(r.label)} ${r.key==='cross_corr'?'<span class="pill">pares</span>':(r.is_target?'':'<span class="pill">contexto</span>')}</td>
        <td class="num">${fmt(r.real_median,r.decimals)}</td>
        ${B?`<td class="num tag">${was(r.key,p=>fmt(p.synth_median,r.decimals))}</td>`:''}
        <td class="num">${fmt(r.synth_median,r.decimals)}</td>
        <td class="num">${ratio(r)}</td>
        <td class="num">${fmt(r.rank_corr,2)}</td>
        <td class="num">${fmt(r.coverage_pct,0)}%${B?` <span class="tag">(${was(r.key,p=>fmt(p.coverage_pct,0)+'%')})</span>`:''}</td></tr>`).join('')}
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
    <h2>Qué se cambió y qué movió cada cosa</h2>
    <p class="lead">Tres cambios en la física del generador, ninguno en los escenarios: ${esc(F.library)}
      se deriva de los mismos <span class="mono">spec.json</span> que ${esc(F.library==='ai_v3'?'ai_v1':'la librería base')}
      con un retrofit determinista, sin volver a llamar a la IA. Las constantes no son opinión: salen de
      iterar este mismo harness como función objetivo.</p>
    <div class="grid cards">
      <div class="card"><h3>1 · La calma no es gaussiana</h3>
        <p class="lead" style="margin:6px 0">La versión anterior daba <span class="mono">tail_dof = 0</span>
          —normal <i>exacta</i>— a toda fase tranquila, que es la mayor parte de un horizonte de 730 días.
          Ahora toda fase tiene cola de Student (calma 5, elevada 4,5, crisis 4).</p>
        <p class="tag">Mueve: curtosis ${B?was('excess_kurtosis',p=>fmt(p.synth_median,2))+' → ':''}${fmt(kurt.synth_median,2)}
          · exceedances ${B?was('exceed_3sigma_pct',p=>fmt(p.synth_median,2))+'% → ':''}${fmt(exc.synth_median,2)}%.</p></div>
      <div class="card"><h3>2 · Más reacción a la noticia</h3>
        <p class="lead" style="margin:6px 0">El GARCH repartía la persistencia 0,15 / 0,85 entre noticia de
          ayer e inercia. Ahora el reparto es un campo del <i>spec</i> (<span class="mono">vol_news</span>),
          calibrado a 0,25. Los dos pesos siguen sumando <span class="mono">p</span>: sube el agrupamiento
          medible, no el nivel de riesgo.</p>
        <p class="tag">Mueve: clustering ${B?was('ac_abs1',p=>fmt(p.synth_median,3))+' → ':''}${fmt(clus.synth_median,3)}
          (real ${fmt(clus.real_median,3)}).</p></div>
      <div class="card"><h3>3 · Betas que suben en el pánico</h3>
        <p class="lead" style="margin:6px 0">Con cargas congeladas, la correlación de un modelo de factores
          es una constante del universo: <b>no puede</b> dispararse en las caídas. Ahora cada fase declara
          un <span class="mono">beta_stress</span> y las betas del día se escalan con él (y la volatilidad
          diaria que calibra mechas y huecos usa la beta efectiva, no la congelada).</p>
        <p class="tag">Mueve: correlación cruzada ${B?was('cross_corr',p=>fmt(p.synth_median,3))+' → ':''}${fmt(cross.synth_median,3)}
          (real ${fmt(cross.real_median,3)}).</p></div>
    </div>
    <h2>Veredicto</h2>
    <div class="note"><b>1 · Lo que ahora sí está en la magnitud del mercado.</b> Curtosis en exceso
      <b>${fmt(kurt.synth_median,2)}</b> frente a ${fmt(kurt.real_median,2)} real
      (cobertura ${fmt(kurt.coverage_pct,0)}%${B?', antes '+was('excess_kurtosis',p=>fmt(p.coverage_pct,0)+'%'):''}),
      exceedances más allá de 3σ ${fmt(exc.synth_median,2)}% frente a ${fmt(exc.real_median,2)}%
      —una gaussiana pura daría 0,27%— y agrupamiento de volatilidad ${fmt(clus.synth_median,3)} frente a
      ${fmt(clus.real_median,3)}. El <b>nivel de riesgo</b> sigue en su sitio: volatilidad anualizada
      ${fmt(vol.real_median,0)}% real vs ${fmt(vol.synth_median,0)}% sintético (ratio ${ratio(vol)}), que es
      lo que hace que el arreglo sea un arreglo y no un cambio de escala.</div>
    <div class="note"><b>2 · Lo que mejora pero no llega: el co-movimiento.</b> La correlación cruzada media
      pasa a <b>${fmt(cross.synth_median,3)}</b> frente a ${fmt(cross.real_median,3)} real
      (rank corr ${fmt(cross.rank_corr,2)} sobre ${cross.n} pares: el modelo de factores <b>ordena</b> los
      pares como la realidad). Cerrar el resto exigiría subir más las betas de estrés, y la beta entra al
      cuadrado en la varianza: se compraría correlación inflando la volatilidad total, que hoy está bien.
      Se prefiere un nivel de riesgo correcto con algo menos de acoplamiento que lo contrario.</div>
    <div class="note"><b>3 · Límite declarado: la mediana del mercado, no los años de manía.</b> El p90 de
      la curtosis real va de 30 a 90 (DOGE y XRP en sus años de manía). Ni con
      <span class="mono">dof = 4</span> se reproduce eso, y perseguirlo rompería el nivel de volatilidad.
      Por eso el umbral de aceptación está sobre la <b>mediana</b>: el generador produce el cripto de un año
      cualquiera, no su cola histórica. Lo que se mida sobre este sustrato sigue siendo optimista para los
      escenarios de manía; ya no lo es para el mercado típico.</div>
    <div class="note"><b>4 · Lo que sigue sin funcionar, y se publica igual: la ordenación.</b> La columna
      <b>rank corr</b> de colas y agrupamiento es floja e incluso negativa
      (${fmt(clus.rank_corr,2)} en clustering, ${fmt(kurt.rank_corr,2)} en curtosis). El mundo sintético ya
      produce la <i>magnitud</i> correcta pero no sabe <b>qué</b> activo tiene más cola: en el mercado real
      son los más ruidosos (DOGE, XRP) y aquí el ruido idiosincrático —al que se le aplica un AR(1) por
      régimen— blanquea justo esa estructura. La cobertura y la mediana no lo tapan porque los tres ejes se
      publican por separado; es el siguiente hilo del que tirar.</div>
    <div class="note"><b>5 · La autocorrelación merece leerse aparte.</b> Real ${fmt(ac.real_median,3)},
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
      Además, el histórico real es un único camino de la historia: sus ${S.n_real_windows} ventanas comparten
      los mismos ciclos de 2018-2025, mientras que el sintético son ${S.n_synthetic_samples} mundos distintos.
      Y las librerías anterior y actual se miden contra <b>la misma</b> ventana real y el mismo harness, que
      es lo único que hace comparable el antes con el después.</div>
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

function renderTransfer(){
  const host=$('#transfer'),T=D.transfer;
  if(!T){
    host.innerHTML=`<h1>Transferencia de ranking</h1>
      <div class="card"><p class="tag">No hay informe publicado. Genéralo con
      <span class="mono">python -m ai_trader.scoring.transfer_study --offline</span>.</p></div>`;
    return;
  }
  const X=T.transfer, V=T.verdict, B=X.bootstrap_blocks, CB=X.bootstrap_configs,
        P=X.permutation, K=X.top_k, DS=X.discrepancies, VAL=T.validation, ACT=X.activity,
        EL=T.eligibility?T.eligibility.sides.real:null;
  const rho=X.spearman, n=T.n_configs;
  const ci=`[${fmt(B.lo,2)}, ${fmt(B.hi,2)}]`;
  const chip=V.transfers?'hecho':(V.key==='ordenacion_invertida'?'pend':'pendiente');
  const veredicto=V.transfers?'transfiere':(V.key==='ordenacion_invertida'?'ordena al revés':'no transfiere');
  const tiles=[
    ['Spearman ρ',fmt(rho,2),`IC${fmt(B.ci_pct,0)}% por bloques ${ci} · umbral ${fmt(V.threshold,2)}`],
    ['Veredicto',veredicto,V.decision],
    [`Top-${K.k} del sintético`,K.hits+'/'+K.k,
      `en la mitad buena del real · azar ${fmt(K.expected_by_chance,1)} · p ${fmt(K.p_value,3)}`],
    ['Desacuerdo medio',fmt(DS.mean_abs_delta,1)+' puestos',
      `${DS.n_large} configs discrepan ≥ ${DS.threshold} puestos (máx ${DS.max_abs_delta})`],
    ['ρ sin las inactivas',fmt(ACT.spearman_active,2),
      `sobre las ${ACT.n_active}/${n} que operan de verdad en los dos mundos`],
  ];
  host.innerHTML=`
    <h1>Transferencia de ranking: ¿ordena el sintético como el mercado?</h1>
    <p class="lead">La vista <b>Fidelidad</b> mide si el generador <i>se parece</i> al mercado en sus
      propiedades estadísticas. Esta vista mide lo único
      que el producto le pide de verdad: que si una configuración es mejor que otra en el mundo
      sintético, <b>tienda a serlo también en el mercado</b>. No es la misma pregunta —un generador
      puede clavar las colas y ordenar al revés—, y ninguna métrica de fidelidad lo detectaría.
      Las mismas <b>${n}</b> configuraciones (la rejilla del estudio de pesos, hipercubo latino con
      semilla ${T.grid.study_seed}) se puntúan dos veces, y se compara el orden.</p>
    <div class="grid tiles">${tiles.map(t=>`<div class="card tile"><div class="k">${t[0]}</div>
      <div class="v">${t[1]}</div><div class="s">${t[2]}</div></div>`).join('')}</div>

    <h2>Veredicto y regla de decisión</h2>
    <div class="note"><b>La regla se declaró en el código antes de mirar el resultado</b>
      (<span class="mono">transfer_study.RHO_ACCEPT = ${fmt(V.threshold,2)}</span>); un umbral que se
      elige después de ver la cifra no es un umbral. <b>ρ ≥ ${fmt(V.threshold,2)}</b> → el generador es
      un <b>pre-cribado legítimo</b> y el flujo definitivo es «real como sustrato primario del ranking,
      sintético como capa de estrés y veto». <b>ρ ≈ 0</b> → no se diseñan estrategias contra el
      sintético. <b>ρ ≤ −${fmt(V.threshold,2)}</b> → ordena al revés, y usarlo para elegir sería peor
      que no usarlo.</div>
    <div class="card">
      <h3>ρ = ${fmt(rho,3)} <span class="chip ${chip}">${veredicto}</span></h3>
      <p class="lead" style="margin:8px 0">${V.transfers
        ? `El orden sintético lleva señal sobre el orden real. El flujo que esto habilita es
           <b>${esc(V.flow)}</b>: el sintético puede <i>filtrar</i> candidatas barato, pero el ranking
           que decide lo sigue fijando el histórico real.`
        : (V.key==='ordenacion_invertida'
          ? `El mundo sintético ordena las configuraciones <b>al revés</b> que el mercado. No es
             "no informa": es que seguirlo sería peor que elegir al azar. ${esc(V.flow)}.`
          : `El orden sintético <b>no</b> lleva señal detectable sobre el orden real con esta
             evidencia. La consecuencia operativa, y hay que decirla en la metodología:
             <b>${esc(V.flow)}</b>.`)}</p>
      <div class="tblwrap"><table>
        <thead><tr><th>Lectura</th><th class="num">valor</th><th>qué dice</th></tr></thead><tbody>
        <tr><td>Spearman de los dos rankings</td><td class="num">${fmt(rho,3)}</td>
          <td class="tag">acuerdo global del orden sobre ${n} configuraciones</td></tr>
        <tr><td>IC${fmt(B.ci_pct,0)}% · bootstrap por bloques</td><td class="num">${ci}</td>
          <td class="tag">${B.n_blocks_real} bloques reales, ${B.n_blocks_synthetic} sintéticos ·
            ${B.excludes_zero?'excluye el cero':'<b>incluye el cero</b>'}</td></tr>
        <tr><td>IC${fmt(CB.ci_pct,0)}% · remuestreando configuraciones</td>
          <td class="num">[${fmt(CB.lo,2)}, ${fmt(CB.hi,2)}]</td>
          <td class="tag">otra pregunta: ¿saldría lo mismo con otras ${n} configuraciones?</td></tr>
        <tr><td>p (permutación, ${P.n_samples.toLocaleString('es')} barajados)</td>
          <td class="num">${fmt(P.p_value,4)}</td>
          <td class="tag">probabilidad de un acuerdo así de fuerte por azar</td></tr>
        <tr><td>Top-${K.k} del sintético en la mitad buena del real</td>
          <td class="num">${K.hits}/${K.k}</td>
          <td class="tag">la lectura operativa del pre-cribado · por azar ${fmt(K.expected_by_chance,1)}
            · p ${fmt(K.p_value,3)}</td></tr>
        <tr><td>Sobrevaloradas / infravaloradas por el sintético</td>
          <td class="num">${DS.n_overrated_by_synthetic} / ${DS.n_underrated_by_synthetic}</td>
          <td class="tag">de los ${DS.n_large} desacuerdos ≥ ${DS.threshold} puestos. Sobrevalorar es
            el error caro: asciende basura a la fase real</td></tr>
        </tbody></table></div>
    </div>

    <h2>El control que hay que hacer antes de creerse un ρ ≈ 0</h2>
    <p class="lead">Un ρ nulo admite dos lecturas muy distintas: «el sintético no transfiere» o
      «ninguno de los dos lados estaba rankeando estrategias». Hay una razón concreta para sospechar lo
      segundo: el headline de una configuración que <b>no opera</b> es <b>0 exacto</b> —curva plana,
      Sharpe 0, rotación 0, caída 0— y en un periodo donde casi todo lo que arriesga pierde, ese cero
      gana. El CVaR, que puntúa por la cola mala, lo premia el doble.</p>
    <div class="card">
      <div class="tblwrap"><table>
        <thead><tr><th>Control</th><th class="num">valor</th><th>qué dice</th></tr></thead><tbody>
        <tr><td>Spearman(recompensa, operaciones) · <b>real</b></td>
          <td class="num">${fmt(ACT.reward_vs_activity_real,3)}</td>
          <td class="tag">${ACT.reward_vs_activity_real<=-0.6
            ? '<b>el ranking real es, en gran parte, una lista de inactividad</b>: cuanto más opera una configuración, peor puntúa'
            : 'la actividad no domina el orden real'}</td></tr>
        <tr><td>Spearman(recompensa, operaciones) · <b>sintético</b></td>
          <td class="num">${fmt(ACT.reward_vs_activity_synthetic,3)}</td>
          <td class="tag">${Math.abs(ACT.reward_vs_activity_synthetic)<0.3
            ? 'el mundo sintético <b>no</b> castiga operar: ofrece suficiente oportunidad como para que actuar no sea automáticamente peor'
            : 'la actividad también condiciona el orden sintético'}</td></tr>
        <tr><td>ρ restringido a las que operan de verdad</td>
          <td class="num">${fmt(ACT.spearman_active,3)}</td>
          <td class="tag">${ACT.n_active}/${n} configuraciones con ≥ ${fmt(ACT.min_trades_per_fold,0)}
            operaciones por ventana OOS <b>en los dos mundos</b>${ACT.bootstrap_active
              ? ` · IC${fmt(ACT.bootstrap_active.ci_pct,0)}% [${fmt(ACT.bootstrap_active.lo,2)},
                  ${fmt(ACT.bootstrap_active.hi,2)}] · p ${fmt(ACT.permutation_active.p_value,3)}`
              : ''}</td></tr>
        </tbody></table></div>
    </div>
    ${ACT.spearman_active!==null&&ACT.spearman_active<=-V.threshold?`
    <div class="note"><b>Y aquí el resultado se pone peor, no mejor.</b> Sobre las ${ACT.n_active}
      configuraciones que <i>sí</i> operan en ambos mundos —el único subconjunto donde la pregunta
      original tiene sentido— el acuerdo no es nulo: es <b>negativo</b> (ρ = ${fmt(ACT.spearman_active,2)}).
      Entre estrategias de verdad, el sintético tiende a <b>invertir</b> el orden del mercado. Con eso, la
      conclusión no es «el sintético no informa» sino «seguirlo para elegir sería peor que tirar una
      moneda».
      ${ACT.bootstrap_active&&!ACT.bootstrap_active.excludes_zero?`<b>Cautela obligatoria:</b> es un
      subconjunto <i>post-hoc</i> de ${ACT.n_active} puntos y su intervalo por bloques
      [${fmt(ACT.bootstrap_active.lo,2)}, ${fmt(ACT.bootstrap_active.hi,2)}] cruza el cero
      (p = ${fmt(ACT.permutation_active.p_value,3)}). Es una señal fuerte, no una prueba; lo que sí
      descarta es que el ρ ≈ 0 de arriba sea un artefacto de la inactividad, porque al quitarla el
      acuerdo no aparece.`:''}</div>`:''}

    ${EL?`<div class="card">
      <h3>El ranking, con y sin suelo de actividad
        <span class="chip ${EL.changes_winner?'pend':'hecho'}">${EL.changes_winner
          ?'la elección cambia':'misma elección'}</span></h3>
      <p class="lead" style="margin:8px 0">Desde que se midió esto, una configuración solo entra en el
        ranking si <b>opera</b> (vista <b>Actividad</b>). Las dos listas se publican: sin suelo ganaba
        <span class="mono">${esc(EL.winner_all||'—')}</span>, con suelo gana
        <span class="mono">${esc(EL.winner_rankable||'—')}</span>, y quedan
        <b>${EL.n_rankable}/${EL.n_configs}</b> configuraciones rankeables en el lado real. El gate pasa
        de <b>${EL.approved_without_floor.length}</b> aprobadas a <b>${EL.approved_with_floor?EL.approved_with_floor.length:"n/d"}</b>.
        El suelo <i>no</i> cambia ninguna recompensa ni el ρ de arriba: cambia quién compite.</p>
      <p class="tag">Fuera del ranking real: ${EL.dropped.length
        ? EL.dropped.map(c=>`<span class="mono">${esc(c)}</span>`).join(', ')
        : 'ninguna'}.</p></div>`:''}

    <h2>Configuración a configuración</h2>
    <p class="lead">Cada punto es una de las ${n} configuraciones: en el eje X su puesto en el ranking
      <b>real</b>, en el Y su puesto en el <b>sintético</b>, con «mejor» hacia arriba y hacia la
      derecha. Si el sintético ordenara igual que el mercado, los puntos caerían sobre la diagonal.
      El hilo punteado mide, para los desacuerdos grandes, cuánto se aleja el punto de ella.</p>
    <div class="card">
      <div class="legend"><span><span class="swatch" style="background:var(--s1)"></span>momentum</span>
        <span><span class="swatch" style="background:var(--s7)"></span>reversión a la media</span>
        <span class="tag">· banda verde = mitad buena del ranking real</span></div>
      <div id="rankchart"></div>
    </div>
    <div class="card" style="margin-top:12px"><div class="tblwrap"><table>
      <thead><tr><th>Configuración</th><th>familia</th><th class="num">CVaR real</th>
        <th class="num">CVaR sint.</th><th class="num">ops/ventana real</th>
        <th class="num">puesto real</th><th class="num">puesto sint.</th>
        <th class="num">Δ</th><th>gate</th></tr></thead>
      <tbody>${T.points.map(c=>`<tr>
        <td class="mono">${esc(c.config_id)}${c.active?'':' <span class="pill">apenas opera</span>'}</td>
        <td class="tag">${c.family==='mean_reversion'?'reversión':'momentum'}</td>
        <td class="num">${fmt(c.reward_real,2)}</td>
        <td class="num">${fmt(c.reward_synthetic,2)}</td>
        <td class="num">${fmt(c.trades_real,1)}</td>
        <td class="num">${c.rank_real}</td><td class="num">${c.rank_synthetic}</td>
        <td class="num">${c.delta>0?'+':''}${c.delta}</td>
        <td class="tag">${c.approved_real?'<span class="chip hecho">real</span>':''}
          ${c.approved_synthetic?'<span class="chip tendencia">sint.</span>':''}
          ${(!c.approved_real&&!c.approved_synthetic)?'—':''}</td></tr>`).join('')}
      </tbody></table></div></div>
    <p class="tag"><b>Δ</b> = puesto real − puesto sintético. Positivo = el sintético la coloca mejor de
      lo que merece (<b>sobrevalora</b>, el error caro). Negativo = la coloca peor (<b>infravalora</b>,
      solo cuesta oportunidades). <b>ops/ventana</b> = operaciones por ventana OOS en el mercado real;
      por debajo de ${fmt(ACT.min_trades_per_fold,0)} la configuración apenas opera y su puesto mide
      inactividad. <b>gate</b> = supera a los baselines pasivos en ese mundo, sobre la distribución
      puesta en común.</p>

    <h2>Cómo está montada la comparación</h2>
    <p class="lead">Toda la construcción persigue una sola cosa: que entre los dos lados <b>lo único que
      cambie sea el mundo del que salen los precios</b>. Si cambiara algo más, el ρ mediría esa otra cosa.</p>
    <div class="grid cards">
      <div class="card"><h3>1 · Mismo universo, mismo config</h3>
        <p class="lead" style="margin:6px 0">Los <b>${T.symbols.length}</b> pares que existen a la vez
          en el mercado y en la librería: <span class="mono">${T.symbols.map(esc).join(', ')}</span>.
          Mismo riesgo, misma ejecución, misma microestructura, mismo factor de anualización
          (${VAL.periods_per_year}, cripto 24/7).</p>
        <p class="tag">${T.omitted.length} pares del universo operable se omiten y se declaran (menos de
          ${T.min_history_days} días de barras: calentamiento + un grupo de CPCV). No se rellenan.
          ${T.library_omitted.length?'De la librería se cae <span class="mono">'+T.library_omitted.map(esc).join(', ')+'</span> por no tener contraparte real.':''}</p></div>
      <div class="card"><h3>2 · Misma longitud de ventana</h3>
        <p class="lead" style="margin:6px 0">Un camino sintético dura <b>${VAL.window_days} días</b>. El
          histórico real se trocea en <b>${T.sub_windows.length}</b> sub-ventanas <i>disjuntas</i> de ese
          mismo tamaño, no en una sola de ocho años: comparar un Sharpe estimado sobre 8 años con otro
          sobre 18 meses compararía precisiones, no mundos.</p>
        <p class="tag">${T.head_discarded_days} días iniciales no completan una sub-ventana y quedan
          fuera; el corte se hace por la cabecera, que es el tramo con menos pares vivos.</p></div>
      <div class="card"><h3>3 · Mismo CPCV, misma purga</h3>
        <p class="lead" style="margin:6px 0">${VAL.n_folds} ventanas OOS por unidad
          (C(${VAL.n_groups},${VAL.n_test_groups})), purga de <b>${VAL.purge_days} días</b>
          —exactamente <span class="mono">max_holding_days</span>, lo que puede seguir viva una posición
          abierta al final del train— y embargo de ${VAL.embargo_days} días, en los dos lados.</p>
        <p class="tag">${T.leakage.folds_audited.toLocaleString('es')} folds auditados ·
          ${T.leakage.clean?'sin fuga temporal':'<b>CON FUGA</b>'}.</p></div>
      <div class="card"><h3>4 · Una sola cola, no dos</h3>
        <p class="lead" style="margin:6px 0">Cada configuración produce muchos scores (sub-ventana × fold
          en el real, muestra × fold en el sintético). Todos van a <b>una sola distribución</b> y se toma
          su CVaR@${fmt(VAL.cvar_alpha*100,0)}%. Tomar el CVaR de cada unidad y luego el CVaR de esos
          CVaR compondría dos conservadurismos y dispararía la varianza del estimador.</p>
        <p class="tag">Real: ${T.points.length?T.points[0].n_real:0} scores por configuración ·
          sintético: ${T.points.length?T.points[0].n_synthetic:0} (${T.n_samples} mundos × ${VAL.n_folds} folds).</p></div>
    </div>

    <h2>Lo que esta cifra no es</h2>
    <p class="lead">Los límites van aquí y no en una nota al pie, porque quien lee este ρ es quien decide
      si se diseña contra el sintético, y esa decisión necesita saber en qué dirección empuja cada sesgo.</p>
    ${T.caveats.map(c=>`<div class="note"><b>${esc(c.title)}.</b> ${esc(c.text)}</div>`).join('')}
    <p class="tag">Evidencia completa:
      <span class="mono">data/transfer/report_${esc(T.library)}.json</span> · reproducible con
      <span class="mono">python -m ai_trader.scoring.transfer_study --offline</span>
      (${T.sub_windows.length} sub-ventanas reales × ${n} configs + ${T.n_samples} muestras de
      ${esc(T.library)}${T.is_fallback?' <b>(librería de reserva: la pedida no existía)</b>':''};
      ${esc(T.generated_at)}).</p>`;

  rankScatter($('#rankchart'),T.points,{n:n});
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

function renderActivity(){
  const host=$('#activity'),A=D.activity;
  if(!A){
    host.innerHTML=`<h1>Actividad: quién puede ganar el ranking</h1>
      <div class="card"><p class="tag">No hay informe publicado. Genéralo con
      <span class="mono">python -m ai_trader.scoring.activity_study</span>.</p></div>`;
    return;
  }
  const F=A.floor, M=A.mechanism.sides.real, DEC=A.decision, G=A.gate.real,
        REP=A.reproducibility.sides.real, BAND=A.band.real;
  const nRank=A.rows.filter(r=>r.rankable).length, n=A.rows.length;
  const winnerAll=A.rows[0], winnerRank=A.rows.find(r=>r.rankable);
  const tiles=[
    ['ρ(recompensa, operaciones)',fmt(M.spearman_reward_activity,2),
      'en el lado real · negativo = cuanto menos opera, mejor puntúa'],
    ['Ventanas vacías',fmt(100*M.empty_windows/M.windows,0)+'%',
      `${M.empty_windows.toLocaleString('es')} de ${M.windows.toLocaleString('es')} · todas puntúan 0 exacto`],
    ['Rankeables',nRank+'/'+n,
      `≥ ${fmt(F.min_median_trades_per_window,0)} ops en la ventana mediana y ≤ ${fmt(F.max_zero_window_pct,0)}% vacías`],
    ['Ganador del ranking real',esc(winnerRank?winnerRank.config_id:'—'),
      winnerAll&&winnerRank&&winnerAll.config_id!==winnerRank.config_id
        ? `sin suelo ganaba ${esc(winnerAll.config_id)} con ${fmt(winnerAll.trades_per_window,2)} ops/ventana`
        : 'el suelo no cambia la elección'],
    ['Gate de baselines',G.approved_without_floor.length+' → '+G.approved_with_floor.length,
      `${G.n_lost} dejan de aprobar al exigir actividad`],
  ];
  host.innerHTML=`
    <h1>Actividad: quién puede ganar el ranking</h1>
    <p class="lead">El headline de una ventana es <span class="mono">Sharpe − λ·rotación − κ·maxDD</span>.
      Si una configuración <b>no abre ninguna posición</b>, la curva es una recta: Sharpe 0, rotación 0,
      caída 0 → headline <b>0 exacto</b>. Y como la recompensa es el <b>CVaR@25%</b> (la media del peor
      cuartil), en un periodo donde casi todo lo que arriesga pierde, <b>ese cero gana</b>. El ranking
      real medido era en buena parte una lista de <i>inactividad</i>: ρ(recompensa, operaciones) =
      <b>${fmt(M.spearman_reward_activity,2)}</b>.</p>
    <div class="grid tiles">${tiles.map(t=>`<div class="card tile"><div class="k">${t[0]}</div>
      <div class="v">${t[1]}</div><div class="s">${t[2]}</div></div>`).join('')}</div>

    <div class="note"><b>Esto NO es una penalización por rotar poco.</b> Esa ya existe
      (<span class="mono">λ_turnover</span>) y el estudio de pesos midió que penalizar
      <i>no estabiliza nada</i> (vista <b>Ranking</b>). Lo que falta no es restar puntos, es un
      <b>requisito de elegibilidad</b>: una configuración que no supera el suelo conserva su recompensa
      publicada intacta y aparece en todas las tablas —lo único que no puede es competir ni aprobar el
      gate—. No perder es legítimo; lo que no es legítimo es <i>ganar un ranking de estrategias</i> por
      no haber jugado.</div>

    <h2>La aritmética, comprobada sobre los datos</h2>
    <p class="lead">La afirmación «sin operaciones → headline 0 exacto» es comprobable, así que el
      estudio la comprueba en los dos mundos en vez de razonarla.</p>
    <div class="card"><div class="tblwrap"><table>
      <thead><tr><th>lado</th><th class="num">ventanas</th><th class="num">vacías</th>
        <th class="num">vacías que puntúan 0 exacto</th><th class="num">no vacías que puntúan 0</th>
        <th class="num">cola del CVaR vacía</th><th>ρ(recompensa, ops)</th></tr></thead><tbody>
      ${['real','synthetic'].map(s=>{const m=A.mechanism.sides[s];return `<tr>
        <td><b>${s==='real'?'mercado real':'sintético'}</b></td>
        <td class="num">${m.windows.toLocaleString('es')}</td>
        <td class="num">${m.empty_windows.toLocaleString('es')}</td>
        <td class="num">${m.empty_windows_scoring_exactly_zero.toLocaleString('es')}</td>
        <td class="num">${m.non_empty_windows_scoring_exactly_zero}</td>
        <td class="num">${fmt(m.cvar_tail_empty_pct,1)}%</td>
        <td class="num">${fmt(m.spearman_reward_activity,3)}</td></tr>`;}).join('')}
      </tbody></table></div>
      <p class="tag">La última columna es la firma del problema y la penúltima es dónde duele: el
        cuartil que <i>fija</i> la recompensa hecho de ceros estructurales.</p></div>

    <h2>Las ${n} configuraciones, con su actividad al lado de su recompensa</h2>
    <p class="lead">Es el cambio de fondo: la actividad ya no hay que ir a buscarla. Viaja con la
      recompensa en <span class="mono">RewardStats</span>, en
      <span class="mono">MultiWindowValidation</span> y en todo lo que publique un ranking.</p>
    <div class="card"><div class="tblwrap"><table>
      <thead><tr><th>configuración</th><th class="num">recompensa real</th>
        <th class="num">ops/ventana</th><th class="num">mediana</th><th class="num">vacías</th>
        <th class="num">ops/ventana (sint.)</th><th>rankeable</th></tr></thead><tbody>
      ${A.rows.map(r=>`<tr>
        <td class="mono">${esc(r.config_id)}</td>
        <td class="num">${fmt(r.reward,3)}</td>
        <td class="num">${fmt(r.trades_per_window,2)}</td>
        <td class="num">${fmt(r.median_trades_per_window,0)}</td>
        <td class="num">${fmt(r.zero_window_pct,0)}%</td>
        <td class="num">${fmt(r.trades_per_window_synthetic,2)}</td>
        <td><span class="chip ${r.rankable?'hecho':'pend'}">${r.rankable?'sí':'no'}</span></td>
        </tr>`).join('')}
      </tbody></table></div>
      <p class="tag">Ordenadas por recompensa real. Las de arriba sin marca verde son las que ganaban
        el ranking sin haber operado.</p></div>

    <h2>De dónde sale el umbral (y no de dónde no sale)</h2>
    <p class="lead">El suelo tiene dos condiciones. Una está <b>derivada</b> y la otra hay que
      <b>medirla</b>.</p>
    <div class="grid cards">
      <div class="card"><h3>≤ ${fmt(F.max_zero_window_pct,0)}% de ventanas vacías</h3>
        <p class="lead" style="margin:8px 0">Derivada, no elegida: es <b>α</b>, la fracción de cola que
          <i>es</i> la recompensa (CVaR@${fmt(F.max_zero_window_pct,0)}%). Por encima de ella, el
          cuartil que promedia el CVaR puede estar hecho de ceros estructurales, y entonces la cifra no
          es una medición.</p></div>
      <div class="card"><h3>≥ ${fmt(F.min_median_trades_per_window,0)} operaciones en la ventana mediana</h3>
        <p class="lead" style="margin:8px 0">Medida. Regla declarada antes de mirar: el valor de la
          rejilla ${DEC.grid.map(g=>fmt(g,0)).join(', ')} que reproduce con <b>menos desacuerdos</b> la
          condición derivada; a igualdad, el mayor. Resultado:
          <b>${fmt(DEC.chosen,0)}</b> (desacuerdos: ${DEC.disagreements}).</p></div>
    </div>
    <div class="card">
      <div class="tblwrap"><table>
        <thead><tr><th class="num">umbral</th><th class="num">rankeables (real)</th>
          <th class="num">desacuerdos</th><th>ganador del ranking real</th>
          <th class="num">aprueban el gate</th><th class="num">máx. vacías entre elegibles</th></tr></thead>
        <tbody>${A.sweep.map(s=>{const r=s.sides.real;const sel=s.threshold===DEC.chosen;return `<tr${sel?' style="font-weight:600"':''}>
          <td class="num">${fmt(s.threshold,0)}${sel?' ←':''}</td>
          <td class="num">${r.n_eligible}</td><td class="num">${s.disagreements}</td>
          <td class="mono">${esc(r.winner||'—')}</td>
          <td class="num">${r.approved.length}</td>
          <td class="num">${fmt(r.max_zero_window_pct_eligible,0)}%</td></tr>`;}).join('')}
        </tbody></table></div>
      <p class="tag">El barrido se publica <b>entero</b>: un umbral que solo se puede defender enseñando
        su resultado y escondiendo los de al lado no es un umbral defendible. Y la elección del número
        casi no importa —con <b>${BAND.same_partition.map(t=>fmt(t,0)).join('/')}</b> operaciones por
        ventana sale <b>exactamente el mismo conjunto rankeable</b> (${BAND.n_rankable} de ${n})—, porque
        a esta escala quien excluye es la condición derivada. La condición medida solo muerde una vez en
        toda la rejilla: una configuración del mundo sintético que se queda fuera con el 25,0 % justo de
        ventanas vacías. Se mantiene porque cubre un modo de fallo que estas 16 no contienen: operar
        poquísimo pero con regularidad, sin dejar ni una ventana vacía.</p></div>

    <h2>El control que parecía obvio y habría elegido al revés</h2>
    <div class="card">
      <h3>Reproducibilidad del ranking entre mitades del histórico</h3>
      <p class="lead" style="margin:8px 0">Con las ${n} configuraciones sale
        <b>${fmt(REP.all_configs,3)}</b>; solo con las rankeables, <b>${fmt(REP.rankable_only,3)}</b>; y
        en subconjuntos <i>aleatorios</i> del mismo tamaño, ${fmt(REP.random_same_size_mean,3)} (el
        control necesario: un ranking de ${nRank} y otro de ${n} no tienen la misma varianza por azar).
        Es decir: la estabilidad <b>sube</b> al meter las inactivas. Y se entiende en cuanto se ve —una
        configuración que no opera puntúa 0 en todos los bloques y su puesto no se mueve jamás—: es
        estabilidad de cementerio. ${esc(A.reproducibility.why_not)}</p>
      <p class="tag">Por eso se publica como control y <b>no</b> como criterio.</p></div>

    <h2>El gate: batir a los pasivos sin jugar no es batirlos</h2>
    <p class="lead">El gate exige ahora <b>dos</b> cosas: batir al mejor baseline pasivo y ser
      rankeable. Los pasivos del periodo real están en torno a <b>−1,4</b>, así que una curva plana
      (0,000) los batía con margen sin haber abierto una posición.</p>
    <div class="card">
      <h3>${G.approved_without_floor.length} → ${G.approved_with_floor.length} aprobadas
        <span class="chip ${G.n_lost?'pend':'hecho'}">${G.n_lost} pierden la aprobación</span></h3>
      ${A.lost.length?`<div class="tblwrap"><table>
        <thead><tr><th>configuración</th><th class="num">recompensa</th><th class="num">ops/ventana</th>
          <th class="num">vacías</th><th>por qué no es rankeable</th></tr></thead><tbody>
        ${A.lost.map(l=>`<tr><td class="mono">${esc(l.config_id)}</td>
          <td class="num">${fmt(l.reward,3)}</td><td class="num">${fmt(l.trades_per_window,2)}</td>
          <td class="num">${fmt(l.zero_window_pct,0)}%</td>
          <td class="tag">${esc(l.reasons.join(' · '))}</td></tr>`).join('')}
        </tbody></table></div>`:'<p class="tag">Ninguna configuración perdía la aprobación.</p>'}
      <p class="tag">Las que caen conservan su recompensa publicada: el suelo no resta nada, decide
        quién compite.</p></div>
    <p class="tag">Evidencia completa: <span class="mono">data/activity/report_${esc(A.library)}.json</span> ·
      unidades: <span class="mono">${esc(A.source.units)}</span> ·
      código: <span class="mono">scoring/activity.py</span>,
      <span class="mono">scoring/activity_study.py</span> · ${esc(A.generated_at)}</p>`;
}

// Un color por sesión, estable en las dos vistas donde aparecen (descomposición y
// tendencia): si el ámbar es Asia en un gráfico, lo es en todos.
const SESSION_COLOR={asia:'--s5',europe:'--s1',us:'--s6'};
function sessionParts(shares,field,sessions,note){
  return sessions.map(s=>({name:s.label,value:shares[s.key][field],
    color:cssv(SESSION_COLOR[s.key]),note:note?note(s):''}));
}

function renderSessions(){
  const host=$('#sessions'),S=D.sessions;
  if(!S){
    host.innerHTML=`<h1>Sesiones: la ventana ciega</h1>
      <div class="card"><p class="tag">No hay informe publicado. Genéralo con
      <span class="mono">python -m ai_trader.backtest.session_study --offline</span>.</p></div>`;
    return;
  }
  const SS=S.sessions, G=S.gap, L=S.latency, T=S.trend, V=S.verdicts, O=S.overall.sessions;
  const l1=L.rows.find(r=>r.hours===1)||{};
  const pct=(v,d=1)=>v===null||v===undefined?'-':fmt(100*v,d)+'%';
  // Referencias de reloj: los cortes acumulados, para ver si una cuota supera a su duración.
  let acc=0; const refs=SS.slice(0,-1).map(s=>{acc+=s.clock_share;return {x:acc};});
  const tiles=[
    ['Hueco cierre → open',pct(G.share_of_range.median,2),
      `del rango del día (mediana) · ${fmt(G.bps.median,1)} pb · umbral ${pct(G.threshold,0)}`],
    ['Con 1 h de retraso',pct(l1.slip_share_of_range_median),
      `desplazamiento del precio de llenado · ${fmt(l1.slip_bps_median,0)} pb`],
    ['Cuota EE.UU. (varianza)',pct(O[S.us_key].variance),
      `en ${fmt(100*SS.find(s=>s.key===S.us_key).clock_share,0)}% del reloj · intensidad ${fmt(O[S.us_key].variance_intensity,2)}`],
    ['Δ EE.UU. tras '+T.split.slice(0,4),(T.mean_delta_variance>=0?'+':'')+pct(T.mean_delta_variance,2),
      `${T.n_positive}/${T.n_symbols} pares al alza · signos p ${fmt(T.sign_test_p,3)}`],
    ['Fija el mínimo del día',S.leader_sets_low.label,
      `${pct(S.leader_sets_low.value)} de los días · ahí muerde la convención pesimista`],
  ];

  host.innerHTML=`
    <h1>Sesiones: dónde se forma el precio dentro de la barra diaria</h1>
    <p class="lead">Todo el sistema es <b>diario</b>: decide con velas 1D ya cerradas, llena al
      <b>open</b> del día siguiente y comprueba los stops contra el <b>máximo/mínimo</b> de la barra
      (§ vista <b>Validación</b> y metodología 3.3). Esa convención trata las 24 horas de la vela como
      un bloque opaco. Este estudio la abre con barras <b>1H</b> de ${S.overall.n_symbols} pares y
      ${S.timeframe==='1H'?'':''}${S.window.start} → ${S.window.end} (ventana cerrada, no «hasta hoy»),
      y responde a una pregunta que nadie había medido: <b>¿cuánto de la formación de precio cae en el
      hueco que el backtest no ve?</b></p>
    <div class="grid tiles">${tiles.map(t=>`<div class="card tile"><div class="k">${t[0]}</div>
      <div class="v">${t[1]}</div><div class="s">${t[2]}</div></div>`).join('')}</div>

    <h2>La cifra que decide</h2>
    <p class="lead">La estrategia ve el <b>cierre de ayer</b>; el motor llena al <b>open de hoy</b>.
      Entre esos dos precios hay una ventana que el backtest no modela. Medirla es medir si la
      convención de llenado regala o cobra de más.</p>
    <div class="card">
      <h3>${pct(G.share_of_range.median,2)} del rango diario
        <span class="chip ${V.gap.material?'pend':'hecho'}">${V.gap.material?'hueco material':'hueco inmaterial'}</span></h3>
      <p class="lead" style="margin:8px 0">${esc(V.gap.text)}</p>
      <div class="tblwrap"><table>
        <thead><tr><th>|hueco| medido contra</th><th class="num">mediana</th><th class="num">media</th>
          <th class="num">p90</th><th class="num">p99</th><th>qué dice</th></tr></thead><tbody>
        <tr><td>el <b>rango</b> del día (máx − mín)</td><td class="num">${pct(G.share_of_range.median,3)}</td>
          <td class="num">${pct(G.share_of_range.mean,3)}</td><td class="num">${pct(G.share_of_range.p90,3)}</td>
          <td class="num">${pct(G.share_of_range.p99,3)}</td>
          <td class="tag">qué parte del recorrido del día se pierde el motor</td></tr>
        <tr><td>el <b>movimiento</b> del día (cierre a cierre)</td>
          <td class="num">${pct(G.share_of_daily_move.median,3)}</td>
          <td class="num">${pct(G.share_of_daily_move.mean,3)}</td>
          <td class="num">${pct(G.share_of_daily_move.p90,3)}</td><td class="num">—</td>
          <td class="tag">qué parte del resultado se decide ahí</td></tr>
        <tr><td>el precio, en <b>puntos básicos</b></td><td class="num">${fmt(G.bps.median,2)}</td>
          <td class="num">${fmt(G.bps.mean,2)}</td><td class="num">${fmt(G.bps.p90,2)}</td>
          <td class="num">${fmt(G.bps.p99,2)}</td>
          <td class="tag">comparable con el coste de ejecución que el motor sí cobra</td></tr>
        </tbody></table></div>
      <p class="tag">${G.n_days.toLocaleString('es')} días-símbolo. Días con el hueco por encima del
        umbral declarado (${pct(G.threshold,0)} del rango): <b>${fmt(G.days_above_threshold_pct,2)}%</b>.
        La cola importa tanto como la mediana: un motor puede estar bien de media y tener un problema
        puntual.</p>
    </div>

    <h2>Lo que sí queda sin modelar: la latencia</h2>
    <p class="lead">Que el hueco valga cero significa que llenar «al open» es exacto <i>si la orden se
      manda en el instante del open</i>. El backtest lo supone; un ciclo real, no. Cada fila responde a
      cuánto cuesta llegar tarde.</p>
    <div class="card"><div class="tblwrap"><table>
      <thead><tr><th>retraso sobre el open</th><th>cae en la sesión</th>
        <th class="num">desplazamiento / rango</th><th class="num">p90</th><th class="num">pb</th>
        <th class="num">rango ya gastado</th></tr></thead>
      <tbody>${L.rows.map(r=>`<tr>
        <td><b>+${r.hours} h</b></td><td class="tag">${esc(_sessLabel(SS,r.session))}</td>
        <td class="num">${pct(r.slip_share_of_range_median)}</td>
        <td class="num">${pct(r.slip_share_of_range_p90)}</td>
        <td class="num">${fmt(r.slip_bps_median,1)}</td>
        <td class="num">${pct(r.path_share_of_range_median)}</td></tr>`).join('')}
      </tbody></table></div>
      <p class="lead" style="margin:10px 0 0">${esc(V.latency.text)}</p></div>
    <div class="note"><b>Primero se mide, después se declara, y solo entonces se toca el motor.</b>
      Este estudio no cambia ni una línea de <span class="mono">execution/market_model.py</span>. Lo que
      hace es convertir una convención que se daba por buena en una cifra con umbral declarado, y dejar
      escrita la limitación en la metodología (3.3) y en el README. Cambiar el motor antes de tener la
      cifra habría sido sustituir una suposición por otra.</div>

    <h2>Descomposición por sesión</h2>
    <p class="lead">Tres tramos del día UTC, y los cortes no son redondos: cada frontera es la hora de
      una <b>apertura de mercado real</b>, tomada en su versión más temprana del año. Como no duran lo
      mismo, la línea de puntos marca dónde caería cada corte si la actividad fuese uniforme: lo que
      sobresale de ella es exceso <i>real</i>, no duración.</p>
    <div class="card">
      <div class="legend">${SS.map(s=>`<span><span class="swatch" style="background:var(${SESSION_COLOR[s.key]})"></span>${esc(s.label)}
        <span class="tag">${String(s.start_hour).padStart(2,'0')}–${String(s.end_hour).padStart(2,'0')} UTC · ${s.hours} h</span></span>`).join('')}
        <span class="tag">· línea de puntos = reparto por reloj</span></div>
      <div id="sesschart"></div>
    </div>
    <div class="card" style="margin-top:12px"><div class="tblwrap"><table>
      <thead><tr><th>Sesión</th><th>UTC</th><th class="num">reloj</th><th class="num">|retorno|</th>
        <th class="num">varianza</th><th class="num">rango</th><th class="num">intensidad</th>
        <th class="num">cubre del rango diario</th><th class="num">fija el máximo</th>
        <th class="num">fija el mínimo</th></tr></thead>
      <tbody>${SS.map(s=>{const r=O[s.key];return `<tr>
        <td><span class="swatch" style="background:var(${SESSION_COLOR[s.key]})"></span> <b>${esc(s.label)}</b></td>
        <td class="mono">${String(s.start_hour).padStart(2,'0')}–${String(s.end_hour).padStart(2,'0')}</td>
        <td class="num tag">${pct(s.clock_share,1)}</td>
        <td class="num">${pct(r.abs_return)}</td><td class="num">${pct(r.variance)}</td>
        <td class="num">${pct(r.range)}</td>
        <td class="num"><b>${fmt(r.variance_intensity,2)}</b></td>
        <td class="num">${pct(r.range_vs_daily)}</td>
        <td class="num">${pct(r.sets_high)}</td><td class="num">${pct(r.sets_low)}</td></tr>`;}).join('')}
      </tbody></table></div>
      <p class="tag"><b>intensidad</b> = cuota de varianza ÷ fracción de reloj; 1,00 es neutro y es lo
        único comparable entre tramos de distinta duración. <b>cubre del rango diario</b> no suma 1 a
        propósito: mide solape con el rango del día, no reparto. <b>fija el máximo/mínimo</b> es la
        probabilidad de que el extremo del día caiga en ese tramo, y es la lectura que le importa a la
        convención pesimista de los stops.</p></div>
    ${_sessionRationale(SS)}

    <h2>Tendencia: ¿crece el peso de EE.UU. tras enero de 2024?</h2>
    <p class="lead">La hipótesis se declaró <b>antes</b> de mirar, con su mecanismo
      (${esc(T.mechanism)}) y su umbral (<span class="mono">US_SHARE_GROWTH_MIN =
      ${pct(T.threshold,0)}</span>). Se mide sobre una <b>cohorte equilibrada</b>: los
      ${T.cohort.length} pares presentes en todos los años. Con el universo completo, la serie
      mediría qué se listó cuándo, no dónde se mueve el precio.</p>
    <div class="card">
      <div class="legend">${SS.map(s=>`<span><span class="swatch" style="background:var(${SESSION_COLOR[s.key]})"></span>${esc(s.label)}</span>`).join('')}
        <span class="tag">· cuota de varianza realizada, año a año</span></div>
      <div id="trendchart"></div>
      <h3 style="margin-top:14px">${(T.mean_delta_variance>=0?'+':'')+pct(T.mean_delta_variance,2)}
        <span class="chip ${T.verdict==='us_crece'?'hecho':(T.verdict==='us_decrece'?'pend':'pendiente')}">
        ${T.verdict==='us_crece'?'dirección: se sostiene':(T.verdict==='us_decrece'?'sale al revés':'no se sostiene')}</span>
        <span class="chip ${T.shape==='salto_en_el_corte'?'hecho':'pendiente'}">
        ${T.shape==='salto_en_el_corte'?'mecanismo: encaja':(T.shape==='deriva_previa'?'mecanismo: NO':'mecanismo: sin contrastar')}</span></h3>
      <p class="lead" style="margin:8px 0">${esc(V.trend.text)}</p>
      <div class="tblwrap"><table>
        <thead><tr><th>Contraste</th><th class="num">valor</th><th>qué dice</th></tr></thead><tbody>
        <tr><td>Δ cuota EE.UU. de varianza (post − pre ${T.split})</td>
          <td class="num">${(T.mean_delta_variance>=0?'+':'')+pct(T.mean_delta_variance,2)}</td>
          <td class="tag">media pareada sobre ${T.n_symbols} pares · desv. ${pct(T.sd_delta_variance,2)}</td></tr>
        <tr><td>Pares que suben</td><td class="num">${T.n_positive}/${T.n_symbols}</td>
          <td class="tag">test de signos exacto, p = ${fmt(T.sign_test_p,4)} · sin scipy y sin
            aproximación normal: con una decena de pares correlacionados, una t fingiría precisión</td></tr>
        <tr><td>t pareada (descriptivo)</td><td class="num">${fmt(T.t_stat,2)}</td>
          <td class="tag">se publica como descriptivo, <b>no</b> como prueba: los pares no son
            independientes entre sí</td></tr>
        <tr><td>Spearman(año, cuota EE.UU.) · toda la ventana</td>
          <td class="num">${fmt(T.spearman_year_vs_us_variance,2)}</td>
          <td class="tag">la cuota sube año a año a lo largo de toda la ventana</td></tr>
        <tr><td>Spearman(año, cuota EE.UU.) · <b>solo antes del corte</b></td>
          <td class="num">${fmt(T.pre_split_spearman,2)}</td>
          <td class="tag">${T.pre_split_years.length?`sobre ${T.pre_split_years[0]}–${T.pre_split_years[T.pre_split_years.length-1]} · `:''}umbral
            declarado ${fmt(T.pre_trend_rho_threshold,2)}: por encima, la cuota <b>ya venía
            subiendo</b> y el mecanismo no puede reclamar el crédito</td></tr>
        <tr><td>Escalón en el corte (último año previo → primero posterior)</td>
          <td class="num">${(T.step_at_split>=0?'+':'')+pct(T.step_at_split,2)}</td>
          <td class="tag">lo que un salto atribuible a enero de 2024 tendría que haber sido:
            <b>positivo y grande</b></td></tr>
        </tbody></table></div>
    </div>
    ${T.shape==='deriva_previa'?`
    <div class="note"><b>Dirección sí, mecanismo no.</b> El contraste pre/post sale limpio
      (${T.n_positive}/${T.n_symbols} pares, p = ${fmt(T.sign_test_p,3)}), pero la serie año a año
      enseña por qué eso no basta: la cuota estadounidense sube de forma sostenida <i>desde
      ${T.pre_split_years[0]}</i> (Spearman ${fmt(T.pre_split_spearman,2)} solo en los años previos) y
      el escalón que cruza el corte es <b>${(T.step_at_split>=0?'+':'')+pct(T.step_at_split,2)}</b> —
      negativo. Lo que el contraste pre/post mide es la <b>pendiente acumulada</b> de varios años
      partida por la mitad, no un efecto de enero de 2024. La hipótesis acierta en la dirección y falla
      en la causa, y las dos cosas hay que decirlas por separado.</div>`:''}
    <div class="card" style="margin-top:12px"><div class="tblwrap"><table>
      <thead><tr><th>Par</th><th class="num">EE.UU. antes</th><th class="num">EE.UU. después</th>
        <th class="num">Δ varianza</th><th class="num">Δ |retorno|</th><th class="num">días</th></tr></thead>
      <tbody>${T.per_symbol.map(r=>`<tr>
        <td class="mono">${esc(r.symbol)}</td><td class="num">${pct(r.us_variance_pre)}</td>
        <td class="num">${pct(r.us_variance_post)}</td>
        <td class="num"><b style="color:${r.delta_variance>=0?'var(--good)':'var(--crit)'}">${(r.delta_variance>=0?'+':'')+pct(r.delta_variance,2)}</b></td>
        <td class="num">${(r.delta_abs>=0?'+':'')+pct(r.delta_abs,2)}</td>
        <td class="num tag">${r.n_days_pre.toLocaleString('es')} / ${r.n_days_post.toLocaleString('es')}</td></tr>`).join('')}
      </tbody></table></div>
      <p class="tag">Comparación <b>pareada</b>: cada par consigo mismo, así que el nivel de cada
        símbolo —que varía mucho— se cancela y lo que queda es el desplazamiento.</p></div>

    <h2>Cobertura de los datos</h2>
    <div class="card"><div class="tblwrap"><table>
      <thead><tr><th>Par</th><th class="num">barras 1H</th><th class="num">días utilizables</th>
        <th class="num">días descartados</th><th>desde</th><th>hasta</th><th>cohorte</th></tr></thead>
      <tbody>${S.symbols.map(s=>`<tr><td class="mono">${esc(s.symbol)}</td>
        <td class="num">${s.n_bars.toLocaleString('es')}</td>
        <td class="num">${s.n_days.toLocaleString('es')}</td>
        <td class="num tag">${s.n_days_dropped.toLocaleString('es')}</td>
        <td class="tag">${esc(s.first_day)}</td><td class="tag">${esc(s.last_day)}</td>
        <td>${s.in_cohort?'<span class="chip hecho">sí</span>':'<span class="chip placeholder">no</span>'}</td></tr>`).join('')}
      </tbody></table></div>
      <p class="tag">Un día entra solo si tiene sus 24 barras <b>y</b> existe la de las 23:00 del día
        anterior: sin ella no hay retorno para la hora 0 y la sesión asiática saldría infravalorada de
        forma sistemática. Los días incompletos se caen enteros; rellenarlos inventaría justo el dato
        que se quiere medir.${S.omitted.length?' Omitidos: '+S.omitted.map(o=>esc(o.symbol)).join(', ')+'.':''}</p></div>

    <h2>Lo que esta cifra no es</h2>
    ${S.caveats.map(c=>`<div class="note"><b>${esc(c.title)}.</b> ${esc(c.text)}</div>`).join('')}
    <p class="tag">Evidencia completa: <span class="mono">data/sessions/report.json</span> ·
      reproducible con <span class="mono">python -m ai_trader.backtest.session_study --offline</span>
      (${S.exchange} 1H, ${S.overall.n_days.toLocaleString('es')} días-símbolo;
      ${esc(S.generated_at)}).</p>`;

  stackedBars($('#sesschart'),[
    {label:'|retorno| acumulado',sub:'cuánto camino recorre',
     parts:sessionParts(O,'abs_return',SS,s=>`intensidad ${fmt(O[s.key].abs_intensity,2)}`)},
    {label:'varianza realizada',sub:'suma de r² · la volatilidad',
     parts:sessionParts(O,'variance',SS,s=>`intensidad ${fmt(O[s.key].variance_intensity,2)}`)},
    {label:'rango',sub:'media de cuotas diarias',parts:sessionParts(O,'range',SS)},
  ],{refs:refs});

  stackedBars($('#trendchart'),T.yearly.map(y=>({
    label:String(y.year),sub:`${y.n_symbols} pares`,
    parts:SS.map(s=>({name:s.label,value:y[s.key].variance,color:cssv(SESSION_COLOR[s.key]),
      note:`${y.n_days.toLocaleString('es')} días`})),
  })),{refs:refs,rowH:34});
}
function _sessLabel(sessions,key){const s=sessions.find(x=>x.key===key);return s?s.label:key;}
function _sessionRationale(sessions){
  return `<div class="grid cards">${sessions.map(s=>`<div class="card">
    <h3><span class="swatch" style="background:var(${SESSION_COLOR[s.key]})"></span>
      ${esc(s.label)} <span class="tag">${String(s.start_hour).padStart(2,'0')}:00–${String(s.end_hour).padStart(2,'0')}:00 UTC</span></h3>
    <p class="lead" style="margin:6px 0">${esc(s.rationale)}</p></div>`).join('')}</div>`;
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
    <div class="note">Conectarla es parte de la evolución <b>#4 · "Poner el paper trading a correr en vivo"</b>
      (sección <b>Evoluciones</b>), que incluye además el diario de ciclos en JSONL del que esta vista se alimentará:
      hoy el estado guarda la foto —posiciones y PnL— pero no la película, y la película es el material con el que
      dentro de unos meses se medirá la divergencia live-vs-backtest.</div>`;
}

const PRIORITY_LABEL={critica:'crítica',alta:'alta',media:'media',baja:'baja',aparcada:'aparcada'};

function roadmapItem(r,i){
  const dep=r.depends?`<span class="tag">depende de #${r.depends}</span>`:'';
  return `<div class="rmitem ${r.priority}">
    <div class="rmrank"><b>${r.rank}</b><small>${PRIORITY_LABEL[r.priority]||r.priority}</small></div>
    <div>
      <div class="rmhead"><h3>${esc(r.title)}</h3>
        <span class="tag"><span class="pill">línea ${esc(r.line)}</span>esfuerzo ${esc(r.effort)}</span></div>
      <div class="rowbtns" style="margin:9px 0 0">
        <span class="chip ${r.priority}">prioridad ${PRIORITY_LABEL[r.priority]||r.priority}</span>
        <span class="chip ${r.status}">${esc(r.status)}</span>
        <span class="chip ${r.impact}">impacto ${esc(r.impact)}</span>${dep}</div>
      ${r.evidence?`<div class="rmev"><b>Evidencia medida:</b> ${esc(r.evidence)}</div>`:''}
      <p class="rmwhy">${esc(r.why)}</p>
      <details class="pr"><summary>Prompt para Claude Code</summary>
        <div class="prompt" id="pr_${i}">${esc(r.prompt)}</div>
        <div class="rowbtns"><button class="btn" onclick="copyPrompt(${i},this)">Copiar prompt</button></div>
      </details>
    </div>
  </div>`;
}

function renderRoadmap(){
  const host=$('#roadmap');
  const items=[...D.roadmap].sort((a,b)=>a.rank-b.rank);
  const groups=(D.roadmap_groups||[]).map(g=>{
    const rows=items.filter(r=>r.group===g.key);
    if(!rows.length) return '';
    return `<div class="rmgroup">
      <h2>${esc(g.title)} <span class="tag">· ${rows.length}</span></h2>
      <p class="sub">${esc(g.subtitle)}</p>
      <div class="rmlist">${rows.map(r=>roadmapItem(r,D.roadmap.indexOf(r))).join('')}</div>
    </div>`;
  }).join('');
  host.innerHTML=`
    <h1>Evoluciones pendientes</h1>
    <p class="lead">Ordenadas <b>de mayor a menor criticidad</b>, no por gusto. El criterio es una asimetría de coste:
      una estrategia añadida hoy se re-evalúa gratis cuando el juez mejore, pero un juez malo contamina todo lo que
      puntúe mientras siga malo. Por eso el sustrato (fidelidad) y el juez (validación multiventana) van delante de la
      cosecha (estrategias), y el paper trading en vivo se lanza en paralelo: es lo único que compra tiempo de
      calendario. Cada ficha lleva su prompt detallado para Claude Code.</p>
    <div class="note"><b>Foco: cripto.</b> Toda la evidencia empírica del repo —fidelidad contra Binance, calibración
      de pesos, estudio de validación— es cripto, y la pata de renta variable del generador no tiene ni un solo dato
      real detrás. Renta variable y mercados de predicción quedan en <b>segundo plano de forma explícita</b>, no por
      olvido. Los stocks <i>sí</i> siguen en el universo sintético: GLD, TLT y UUP son lo que hace que los escenarios
      de tipos y de dólar signifiquen algo para cripto vía los factores compartidos. Se genera con los 35, se puntúa y
      se opera solo cripto.</div>
    ${groups}`;
}
function copyPrompt(i,btn){const t=D.roadmap[i].prompt;navigator.clipboard.writeText(t).then(()=>{const o=btn.textContent;btn.textContent='✓ Copiado';setTimeout(()=>btn.textContent=o,1500);});}
window.copyPrompt=copyPrompt;

const PIT_LABEL={forward_capture:'solo hacia adelante',archive_revisable:'backfill revisable',
  derived_from_price:'derivada del precio'};
const PIT_NOTE={forward_capture:'nadie publica su pasado: lo que no se capture hoy no existirá',
  archive_revisable:'hay histórico, pero el proveedor puede reescribirlo',
  derived_from_price:'point-in-time por construcción'};

function renderSignals(){
  const host=$('#signals'),S=D.signals_platform;
  if(!S){host.innerHTML='<h1>Señales externas</h1><div class="card"><p class="tag">Sin datos.</p></div>';return;}
  const M=S.summary,E=S.entities,A=S.archive,C=S.capture,N=S.normalization;
  const nConn=S.sources.filter(s=>s.connected).length;
  const measured=S.sources.filter(s=>s.measured_from);
  const fwd=M.by_pit.forward_capture;
  const deepest=measured.slice().sort((a,b)=>(a.measured_from<b.measured_from?-1:1))[0];
  const tiles=[
    ['Fuentes declaradas',M.n_sources,`${M.by_tier.A} mecánicas (A) · ${M.by_tier.B} estadísticas (B)`],
    ['Con adaptador',nConn+'/'+M.n_sources,'las '+M.by_tier.B+' continuas (Tier B); las mecánicas van por otra puerta'],
    ['Con profundidad MEDIDA',measured.length+'/'+M.n_sources,
      deepest?`la más honda arranca en ${deepest.measured_from} (${esc(deepest.key)})`:'sin medición todavía'],
    ['Backtesteables hoy',M.n_backtestable+'/'+M.n_sources,'solo las que la sonda pudo medir'],
    ['Cobertura de entidades',fmt(E.coverage_pct,0)+'%',
      `${E.by_source.rule} por regla · ${E.by_source.override} overrides · ${E.by_source.unmapped} sin resolver`],
    ['Registros archivados',A.records.toLocaleString('es'),'crudo, append-only, con su fetched_at'],
  ];
  host.innerHTML=`
    <h1>Señales externas: once fuentes conectadas y su profundidad medida</h1>
    <p class="lead">Hoy las estrategias solo ven <b>precio y volumen</b>. Sobre el esqueleto ya construido
      —catálogo, puerto de dos capas, archivo crudo, captura, auditoría— este paso conecta el primer lote
      de fuentes <b>continuas</b> (Tier B: entran como features, nunca como vetos) y mide lo único que
      decide si sirven para algo: <b>desde cuándo hay dato de verdad</b>. Sigue sin cablearse nada a
      ninguna estrategia.</p>
    <div class="grid tiles">${tiles.map(t=>`<div class="card tile"><div class="k">${t[0]}</div>
      <div class="v">${t[1]}</div><div class="s">${t[2]}</div></div>`).join('')}</div>

    <div class="note"><b>El día cero decía «0 conectadas, 0 backtesteables».</b> Hoy dice
      <b>${nConn} conectadas</b>, <b>${measured.length} medidas</b> y <b>${M.n_backtestable}
      backtesteables</b>, y que las tres cifras no coincidan es el contenido. Tener adaptador no es
      tener historia: <span class="mono">history_from</span> solo se escribe en el catálogo cuando la
      sonda (<span class="mono">ai-trader signals depth</span>) descarga la serie, la deriva con el
      adaptador de verdad y apunta la fecha en
      <span class="mono">data/signals/history_depth.json</span>${S.depth_measured_at?` (última medición:
      ${esc(S.depth_measured_at)})`:''}. Y tener medición tampoco basta: hace falta <b>al menos un año
      medido</b>, porque en una fuente <i>forward capture</i> el primer día con dato es sencillamente
      el día que arrancó la captura, y declararlo pondría «backtesteable» en una serie de un día. Un
      test compara las tres cosas y falla si alguien declara una fecha que el registro no respalda: la
      honestidad deja de depender de que nadie se despiste.</div>

    <h2>Por qué la captura se arranca antes que los adaptadores</h2>
    <p class="lead"><b>${fwd} de las ${M.n_sources}</b> fuentes son <span class="mono">forward_capture</span>:
      su pasado no se puede descargar porque nadie lo publica —una cola de staking, un libro P2P, la lista
      SDN de hace tres meses—. Para esas, la profundidad histórica no depende de escribir mejor código
      después, sino de una sola variable: <b>desde cuándo se está capturando</b>. Cada día sin capturar es
      profundidad que no se recupera ni pagando. Por eso
      <span class="mono">ai-trader signals capture</span> corre hoy, sin un solo adaptador, y declara
      ${C?`${C.n_pending} fuentes pendientes y ${C.records} registros`:'el estado real'} en vez de fallar.</p>

    <h2>Las ${M.n_sources} fuentes declaradas</h2>
    <p class="lead"><b>Dos puertas, y la separación vive en el catálogo, no en un comentario.</b> El tier
      <b>A</b> es oferta calendarizada y determinista —unlocks, colas de staking, dificultad, hacks,
      sanciones— con muestras de <i>decenas</i>: entra como <b>elegibilidad</b> y nunca como feature de un
      optimizador, porque un CEM suelto sobre catorce observaciones construye una estrategia preciosa y
      falsa. El tier <b>B</b> son series continuas con efectos pequeños y decadentes: entra como features.</p>
    <div class="card"><div class="tblwrap"><table>
      <thead><tr><th>fuente</th><th>tier</th><th>point-in-time</th>
        <th class="num">features</th><th class="num">ent.</th><th>declarada</th><th>MEDIDA</th>
        <th class="num">días</th><th class="num">registros</th><th>acceso</th></tr></thead>
      <tbody>${S.sources.map(s=>`<tr>
        <td><b>${esc(s.title)}</b><div class="tag mono">${esc(s.key)} · ${esc(s.scope)} · ${esc(s.cadence)}</div></td>
        <td><span class="chip ${s.tier==='A'?'critica':'bajo'}">${s.tier}</span>
          ${s.connected?'<div class="tag">conectada</div>':'<div class="tag">sin adaptador</div>'}</td>
        <td>${esc(PIT_LABEL[s.pit]||s.pit)}<div class="tag">${esc(PIT_NOTE[s.pit]||'')}</div></td>
        <td class="num">${s.features.length}</td>
        <td class="num">${s.n_entities}</td>
        <td>${s.history_from?esc(s.history_from):'<span class="chip pend">sin medir</span>'}</td>
        <td>${s.measured_from?`${esc(s.measured_from)}<div class="tag">${esc(s.measure_method||'')} · ${s.measured_entities} ent.</div>`
          :`<span class="chip pend">—</span>${s.measure_error?`<div class="tag">${esc(String(s.measure_error).slice(0,60))}</div>`:''}`}</td>
        <td class="num">${s.measured_days?s.measured_days.toLocaleString('es'):'·'}</td>
        <td class="num">${s.archived_records?s.archived_records.toLocaleString('es'):'·'}</td>
        <td class="tag">${s.auth_env?esc(s.auth_env):'abierta'}</td></tr>`).join('')}
      </tbody></table></div>
      <p class="tag"><b>Declarada</b> es lo que dice el catálogo y <b>medida</b> lo que devolvió el
        proveedor al sondearlo; tienen que coincidir y hay un test que lo exige. Un guion en la columna
        medida no significa «no hay historia»: significa que no se pudo medir, y el motivo está al lado
        —una credencial que falta, una cuota agotada—. Las fuentes
        <span class="mono">forward_capture</span> miden lo que llevan capturado, que arranca en cero por
        definición y crece un día por día real.</p></div>

    <h2>Símbolo → entidad: una regla, y una tabla que arranca vacía</h2>
    <p class="lead">Las fuentes externas no hablan en símbolos de mercado. La clave común se deriva con una
      <b>regla</b> (<span class="mono">XYZ/USDT → XYZ</span>) que funciona el día que se añade un listado
      nuevo, sin que nadie edite nada. La tabla de overrides existe para los casos en que la regla acierta
      la forma y falla el fondo, y <b>empieza vacía</b>: cada línea tendrá que venir de un fallo observado
      aquí. El descubrimiento es <b>medición</b>, no una sesión de curación previa.</p>
    <div class="grid cards">
      <div class="card"><h3>${E.n_symbols} símbolos → ${E.n_entities} entidades</h3>
        <p class="lead" style="margin:8px 0">Cobertura <b>${fmt(E.coverage_pct,0)}%</b>:
          ${E.by_source.rule} por regla derivada, ${E.by_source.override} por override,
          ${E.by_source.unmapped} sin resolver.
          ${E.unmapped.length?`Sin resolver: <span class="mono">${E.unmapped.map(esc).join(', ')}</span>.`:
            'Ninguno sin resolver, así que la tabla de overrides sigue sin necesitar una sola línea.'}</p></div>
      <div class="card"><h3>Colisiones: ${Object.keys(E.collisions).length}</h3>
        <p class="lead" style="margin:8px 0">Dos símbolos que apuntan a la misma entidad
          <i>no</i> son un error —BTC/USDT y BTC/USD son el mismo activo, y compartir entidad es lo que
          evita descargar dos veces—. Se publican porque una colisión <b>inesperada</b> es exactamente el
          fallo que la regla puede cometer, y sin esta lista sería invisible.
          ${Object.keys(E.collisions).length?'':'Hoy no hay ninguna en el universo configurado.'}</p></div>
    </div>

    <h2>Cómo está montado (y qué se paga después si se monta mal)</h2>
    <div class="grid cards">
      <div class="card"><h3>El puerto tiene dos capas</h3>
        <p class="lead" style="margin:8px 0"><span class="mono">fetch_raw</span> toca la red y devuelve el
          payload <b>intacto</b>; <span class="mono">daily_from_raw</span> es una función <b>pura</b> que
          lo traduce. La parte que se equivoca es siempre el mapeo, y separadas corregirlo
          <b>re-deriva</b> sobre lo archivado en vez de re-descargar —que con una fuente cuyo pasado no
          existe no es una molestia, es información perdida—.</p></div>
      <div class="card"><h3>El crudo va en <span class="mono">data/</span>, no en <span class="mono">.cache/</span></h3>
        <p class="lead" style="margin:8px 0">Append-only en
          <span class="mono">data/signals_raw/&lt;fuente&gt;/&lt;entidad&gt;/&lt;mes&gt;.jsonl.gz</span> con
          su <span class="mono">fetched_at</span>. No se re-deriva: se <b>guarda</b>. Es el mismo principio
          por el que se guarda el <span class="mono">spec.json</span> de cada escenario sintético. Lo
          derivado —los frames diarios— vive en <span class="mono">.cache/signals/</span> y se borra sin
          perder nada.</p></div>
      <div class="card"><h3>Un día tiene varias observaciones</h3>
        <p class="lead" style="margin:8px 0">El esquema canónico es
          <span class="mono">(entidad, día) → features + observed</span>, y <b>no</b> reutiliza el de
          barras: aquel deduplica con <span class="mono">keep='last'</span> y se comería tres emisores
          publicando el mismo día. La agregación vive en un solo sitio y cada feature declara <i>cómo</i>
          se agrega (sumar un estado o quedarse con el último de un flujo son errores que no se ven en la
          gráfica).</p></div>
      <div class="card"><h3><span class="mono">observed</span>: los tres estados</h3>
        <p class="lead" style="margin:8px 0">La convención del sistema es «feature no disponible = 0.0
          neutro», y para señales es <b>peligrosa</b>: tono 0 es neutral, pero <i>no tengo datos</i> no es
          <i>no hay señal</i>. La columna cuenta las observaciones crudas detrás de cada celda, que es lo
          que permitirá que una puerta de señales <b>falle abierta</b> en vez de confundir ausencia con
          neutralidad.</p></div>
    </div>

    <h2>Toda feature se publica normalizada, y con dos varas de medir</h2>
    <p class="lead">Las once fuentes producen números que no se pueden ni sumar ni comparar: 90.000
      millones de oferta de stablecoins, 3,4 puntos básicos de dispersión de funding, 23.000 visitas, una
      prima del 12%, 47 commits. Y comparar <i>entre activos</i> es peor: BTC tiene 23.000 visitas y SEI
      40, lo cual no dice nada sobre cuál está recibiendo atención <b>inusual</b>. Por eso nada sale de la
      ingesta en crudo.</p>
    <div class="grid cards">
      <div class="card"><h3><span class="mono">&lt;feature&gt;${esc(N.suffix_self)}</span> · contra su propia historia</h3>
        <p class="lead" style="margin:8px 0">Ventana <b>expansiva y causal</b> (hasta el día <i>t</i>
          incluido, jamás futuro). Responde: <i>¿esto es alto para este activo?</i> Necesita al menos
          <b>${N.min_history}</b> observaciones propias, así que un listado nuevo no la tiene.</p></div>
      <div class="card"><h3><span class="mono">&lt;feature&gt;${esc(N.suffix_cross)}</span> · contra la sección cruzada del día</h3>
        <p class="lead" style="margin:8px 0">Responde: <i>¿esto es alto hoy, comparado con los demás?</i>
          Exige <b>${N.min_cross_section}</b> entidades con dato ese día. Es la que <b>sí</b> funciona
          desde el primer día de un listado nuevo, y por eso hacen falta las dos.</p></div>
      <div class="card"><h3>Mediana e IQR, no media y desviación típica</h3>
        <p class="lead" style="margin:8px 0">Estas series tienen colas gruesas por construcción: un hack,
          un unlock, un pico de menciones. Con media y sigma, <b>un</b> día extremo infla la escala y
          aplasta todo lo que viene después —la señal se apaga justo cuando empieza a pasar algo—. Con
          <span class="mono">${esc(N.scale)}</span> el mismo día se sale de rango sin mover la vara.</p></div>
      <div class="card"><h3>Recorte declarado y huecos a NaN</h3>
        <p class="lead" style="margin:8px 0">Todo se recorta a <b>±${N.clip}</b>, un número publicado y no
          escondido en el código: una z de 300 no es una señal fortísima, es una división por casi cero.
          Y lo que no se puede calcular queda en <b>NaN</b>, nunca en 0: un cero diría «normal, en la
          media», que es una afirmación sobre el mundo que no se ha observado.</p></div>
    </div>

    <h2>Lo que se aprendió midiendo (y que no estaba en ningún folleto)</h2>
    <div class="card"><div class="tblwrap"><table>
      <thead><tr><th>hallazgo</th><th>qué cambió</th></tr></thead><tbody>
      ${S.etf_dispersion&&S.etf_dispersion.days?`<tr><td><b>Bajar al emisor no es un adorno</b></td>
        <td>Sobre los ${S.etf_dispersion.days} días de flujos de ETF: la dispersión
          bruto/|neto| tiene mediana <b>${fmt(S.etf_dispersion.median,2)}</b> —casi todos los días son
          flujo neto puro— y percentil 95 de <b>${fmt(S.etf_dispersion.p95,1)}</b>, con
          <b>${S.etf_dispersion.rotation_days}</b> días por encima de 3. Esos son días en los que unos
          emisores entran mientras otros salen y el agregado, que es la cifra que publica todo el
          mundo, marca poco o nada.</td></tr>`:''}
      <tr><td><b>El COT se conoce tres días después de su fecha</b></td>
        <td>La foto es del martes y se publica el viernes. La serie se archiva con el día de
          <b>publicación</b>, no el de referencia: fecharla el martes metería tres días de futuro en
          cualquier cruce, y el recorte por día de observación no puede verlo.</td></tr>
      <tr><td><b>El sello de funding de CCXT es el próximo cobro, no la observación</b></td>
        <td>Se descubrió porque la sonda devolvió una serie que empezaba <i>mañana</i>. El sello sigue
          usándose para agrupar venues en la misma ventana; el día sale de la observación.</td></tr>
      <tr><td><b>Un slug con 400 se llevó por delante otras 23 series</b></td>
        <td>Una llamada rechazada ya no tumba a las demás de su fuente, igual que una fuente caída no
          tumba la captura. El TVL de una cadena vive en otro endpoint que el de un protocolo.</td></tr>
      <tr><td><b>Wikimedia contesta 403 al User-Agent genérico y 429 en cuanto hay prisa</b></td>
        <td>Adaptador secuencial, pausa de 5 s y reintento largo. Con seis idiomas y quince artículos el
          barrido tarda minutos: una fuente diaria puede permitírselo.</td></tr>
      <tr><td><b>TFTC publica los ETF de BTC, no los de ETH</b></td>
        <td>ETH se queda sin cobertura por esta puerta en vez de rellenarse con un agregado sin desglose,
          que haría que la dispersión entre emisores significase otra cosa según el activo.</td></tr>
      <tr><td><b>FRED discontinuó las series de oro de la LBMA</b></td>
        <td>El factor COMMODITY se cubre con WTI. Declarar el id del oro «porque siempre estuvo» habría
          dejado una serie que falla en silencio en cada captura.</td></tr>
      </tbody></table></div></div>

    <div class="note"><b>Nada de esto está cableado todavía.</b> Ni a las estrategias, ni al runner, ni al
      espacio de búsqueda del optimizador. El catálogo declara, la captura archiva, la sonda mide y la
      normalización publica; cablearlo es el paso siguiente y tiene su propia compuerta: con las puertas
      neutras, la validación multiventana tiene que devolver <b>exactamente</b> los scores ya
      publicados.</div>`;
}

function main(){
  initTheme();initNav();
  renderOverview();renderSynthetic();renderFidelity();renderTransfer();renderStrategies();
  renderRanking();renderActivity();renderValidation();renderSessions();renderSignals();
  renderPaper();renderRoadmap();
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
      <li><button data-sec="transfer">Transferencia</button></li>
      <li><button data-sec="strategies">Estrategias</button></li>
      <li><button data-sec="ranking">Ranking</button></li>
      <li><button data-sec="activity">Actividad</button></li>
      <li><button data-sec="validation">Validación</button></li>
      <li><button data-sec="sessions">Sesiones</button></li>
      <li><button data-sec="signals">Señales</button></li>
      <li><button data-sec="paper">Paper trading</button></li>
      <li><button data-sec="roadmap">Evoluciones</button></li>
    </ul>
  </aside>
  <main class="main">
    <button id="themeBtn" class="btn ghost toggle">☾ Oscuro</button>
    <section id="overview" class="section active"></section>
    <section id="synthetic" class="section"></section>
    <section id="fidelity" class="section"></section>
    <section id="transfer" class="section"></section>
    <section id="strategies" class="section"></section>
    <section id="ranking" class="section"></section>
    <section id="activity" class="section"></section>
    <section id="validation" class="section"></section>
    <section id="sessions" class="section"></section>
    <section id="signals" class="section"></section>
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
