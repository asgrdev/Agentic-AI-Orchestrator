# api/dashboard.py
"""
Management Dashboard — مدیریت فلوی orchestrator، دانش، skill ها و ابزارها

سه بخش:
1. 📊 Orchestrator: آمار فلو (query type، مراحل، زمان‌بندی، confidence)
2. 🕸 Knowledge Graph: آمار گراف + نمایش تعاملی force-directed
3. 🧠 Knowledge & Skills: افزودن دانش، اجرای refresh دستی، لیست skill ها

همه callback ها defensive هستند — خطای هر بخش نباید کل UI را بشکند.
"""
from __future__ import annotations

import html
import json
import logging
from datetime import datetime

import gradio as gr

logger = logging.getLogger(__name__)

# ── پالت validated (روشن/تاریک) — reference palette از متد dataviz ──────────
_CSS = """
<style>
.agr-dash {
  --surface-1: #fcfcfb; --surface-2: #f0efec;
  --text-primary: #0b0b0b; --text-secondary: #52514e;
  --series-1: #2a78d6; --series-2: #1baf7a; --series-3: #eda100;
  --series-4: #008300; --series-5: #4a3aa7; --series-6: #e34948;
  --good: #0ca30c; --critical: #d03b3b;
  font-family: -apple-system, "Segoe UI", sans-serif;
  color: var(--text-primary);
}
@media (prefers-color-scheme: dark) {
  .agr-dash {
    --surface-1: #1a1a19; --surface-2: #383835;
    --text-primary: #ffffff; --text-secondary: #c3c2b7;
    --series-1: #3987e5; --series-2: #199e70; --series-3: #c98500;
    --series-4: #008300; --series-5: #9085e9; --series-6: #e66767;
  }
}
.agr-tiles { display: flex; flex-wrap: wrap; gap: 12px; margin: 8px 0; }
.agr-tile {
  background: var(--surface-2); border-radius: 10px; padding: 14px 18px;
  min-width: 140px; flex: 1;
}
.agr-tile .v { font-size: 1.7em; font-weight: 700; color: var(--text-primary); }
.agr-tile .l { font-size: 0.8em; color: var(--text-secondary); margin-top: 2px; }
.agr-bars { margin: 10px 0; }
.agr-bar-row { display: flex; align-items: center; gap: 8px; margin: 4px 0; }
.agr-bar-label { width: 110px; font-size: 0.8em; color: var(--text-secondary); text-align: right; }
.agr-bar-track { flex: 1; background: var(--surface-2); border-radius: 4px; height: 16px; position: relative; }
.agr-bar-fill { background: var(--series-1); border-radius: 4px 4px 4px 4px; height: 12px; margin: 2px; }
.agr-bar-val { font-size: 0.75em; color: var(--text-primary); min-width: 54px; }
.agr-note { color: var(--text-secondary); font-size: 0.85em; }
</style>
"""


def _tile(value, label) -> str:
    return (
        f'<div class="agr-tile"><div class="v">{html.escape(str(value))}</div>'
        f'<div class="l">{html.escape(str(label))}</div></div>'
    )


def _tiles(items: list[tuple]) -> str:
    return _CSS + '<div class="agr-dash"><div class="agr-tiles">' + "".join(
        _tile(v, l) for v, l in items
    ) + "</div></div>"


def _bar_list(rows: list[tuple[str, float]], unit: str = "s") -> str:
    """لیست میله‌ای تک‌رنگ (sequential) برای مقایسه زمان مراحل"""
    if not rows:
        return _CSS + '<div class="agr-dash"><p class="agr-note">داده‌ای موجود نیست</p></div>'
    mx = max(v for _, v in rows) or 1.0
    body = "".join(
        f'<div class="agr-bar-row"><div class="agr-bar-label">{html.escape(k)}</div>'
        f'<div class="agr-bar-track"><div class="agr-bar-fill" style="width:{max(v / mx * 100, 1):.1f}%"></div></div>'
        f'<div class="agr-bar-val">{v:.2f}{unit}</div></div>'
        for k, v in rows
    )
    return _CSS + f'<div class="agr-dash"><div class="agr-bars">{body}</div></div>'


class DashboardService:
    """پل بین orchestrator و UI — همه دسترسی‌ها این‌جا متمرکز است"""

    def __init__(self, orchestrator):
        self.orch = orchestrator

    # ── Orchestrator stats ────────────────────────────────────────────

    def flow_stats_html(self) -> str:
        hist = getattr(self.orch, "flow_history", []) or []
        total = len(hist)
        answered = sum(1 for h in hist if h.get("answered"))
        avg_conf = (sum(h.get("confidence", 0) for h in hist) / total) if total else 0.0
        avg_time = (
            sum(sum(h.get("timings", {}).values()) for h in hist) / total
            if total else 0.0
        )
        errors = sum(len(h.get("errors", [])) for h in hist)
        return _tiles([
            (total, "کل کوئری‌ها"),
            (f"{(answered / total * 100) if total else 0:.0f}%", "نرخ پاسخ‌دهی"),
            (f"{avg_conf:.2f}", "میانگین Confidence"),
            (f"{avg_time:.1f}s", "میانگین زمان کل"),
            (errors, "خطاها"),
        ])

    def step_timings_html(self) -> str:
        hist = getattr(self.orch, "flow_history", []) or []
        agg: dict[str, list[float]] = {}
        for h in hist:
            for step, t in h.get("timings", {}).items():
                agg.setdefault(step, []).append(t)
        rows = sorted(
            ((step, sum(ts) / len(ts)) for step, ts in agg.items()),
            key=lambda x: x[1], reverse=True,
        )
        return _bar_list(rows)

    def flow_table(self) -> list[list]:
        hist = getattr(self.orch, "flow_history", []) or []
        rows = []
        for h in reversed(hist[-50:]):
            rows.append([
                datetime.fromtimestamp(h.get("timestamp", 0)).strftime("%H:%M:%S"),
                h.get("query", ""),
                h.get("query_type", ""),
                h.get("strategy", ""),
                h.get("final_step", ""),
                round(h.get("confidence", 0), 2),
                h.get("iterations", 0),
                h.get("chunks_retrieved", 0),
                "; ".join(h.get("errors", []))[:120],
            ])
        return rows

    def skills_table(self) -> list[list]:
        try:
            from agents.skill_executor import get_global_registry
            skills = get_global_registry().list_skills()
            return [
                [s.get("name", ""), s.get("category", s.get("type", "")),
                 s.get("description", "")[:100]]
                for s in sorted(skills, key=lambda x: (x.get("category", ""), x.get("name", "")))
            ]
        except Exception as e:
            logger.warning("skills_table failed: %s", e)
            return [["error", "", str(e)]]

    def skill_exec_html(self) -> str:
        try:
            s = self.orch.skill_executor.get_execution_summary()
            return _tiles([
                (s["total_executions"], "اجراهای skill"),
                (s["successful"], "موفق"),
                (s["failed"], "ناموفق"),
                (f"{s['success_rate'] * 100:.0f}%", "نرخ موفقیت"),
                (f"{s['average_time']:.2f}s", "میانگین زمان"),
            ])
        except Exception as e:
            return _tiles([("—", f"خطا: {e}")])

    def refresh_history_table(self) -> list[list]:
        try:
            hist = self.orch.refresher.get_refresh_history(limit=20)
            return [
                [h.get("timestamp", "")[:19], h.get("total_items", 0),
                 h.get("chunks_ingested", 0), h.get("nodes_added", 0),
                 h.get("edges_added", 0), round(h.get("duration_seconds", 0), 1)]
                for h in reversed(hist)
            ]
        except Exception as e:
            logger.warning("refresh_history failed: %s", e)
            return []

    # ── Knowledge Graph ───────────────────────────────────────────────

    async def graph_stats_html(self) -> str:
        stats = {}
        try:
            if not self.orch.retriever._ready:
                await self.orch.retriever.startup()
            stats = await self.orch.retriever._kuzu.stats()
        except Exception as e:
            logger.warning("graph stats failed: %s", e)
        return _tiles([
            (stats.get("entities", stats.get("entity_count", 0)), "Entities"),
            (stats.get("relations", stats.get("relation_count", 0)), "Relations"),
            (stats.get("chunks", stats.get("chunk_count", 0)), "Chunks"),
            (stats.get("documents", stats.get("document_count", 0)), "Documents"),
        ])

    async def graph_html(self, entity_filter: str = "", limit: int = 120) -> str:
        """نمایش force-directed گراف — self-contained، بدون CDN"""
        nodes, edges = [], []
        try:
            if not self.orch.retriever._ready:
                await self.orch.retriever.startup()
            kuzu = self.orch.retriever._kuzu

            if entity_filter.strip():
                q_nodes = """
                    MATCH (e:Entity)
                    WHERE lower(e.name) CONTAINS lower($f)
                    RETURN e.id AS id, e.name AS name, e.type AS type
                    LIMIT $lim
                """
                res = await kuzu.execute(q_nodes, {"f": entity_filter.strip(), "lim": limit})
            else:
                res = await kuzu.execute(
                    "MATCH (e:Entity) RETURN e.id AS id, e.name AS name, e.type AS type LIMIT $lim",
                    {"lim": limit},
                )
            nodes = [dict(r) for r in (res.rows if res else [])]

            if nodes:
                ids = [n["id"] for n in nodes]
                res_e = await kuzu.execute(
                    """
                    MATCH (a:Entity)-[r:RELATION]->(b:Entity)
                    WHERE list_contains($ids, a.id) AND list_contains($ids, b.id)
                    RETURN a.id AS src, b.id AS dst, r.relation_type AS type
                    LIMIT 400
                    """,
                    {"ids": ids},
                )
                edges = [dict(r) for r in (res_e.rows if res_e else [])]
        except Exception as e:
            logger.warning("graph_html failed: %s", e)
            return _CSS + (
                f'<div class="agr-dash"><p class="agr-note">خطا در خواندن گراف: '
                f"{html.escape(str(e))}</p></div>"
            )

        if not nodes:
            return _CSS + (
                '<div class="agr-dash"><p class="agr-note">گراف خالی است — '
                "با پرسیدن سوال یا افزودن دانش، گراف ساخته می‌شود.</p></div>"
            )

        payload = json.dumps({"nodes": nodes, "edges": edges}, ensure_ascii=False)
        doc = _GRAPH_TEMPLATE.replace("__DATA__", payload)
        return (
            f'<iframe style="width:100%;height:560px;border:0;border-radius:10px;" '
            f'srcdoc="{html.escape(doc)}"></iframe>'
        )

    # ── Knowledge & Skills management ─────────────────────────────────

    async def ingest_text(self, text: str, source: str) -> str:
        if not text or not text.strip():
            return "⚠️ متنی وارد نشده است."
        try:
            if not self.orch.refresher._ready:
                await self.orch.refresher.startup()
            n = await self.orch.refresher._chunk_pipeline.ingest_text(
                text.strip(), source=source.strip() or "manual"
            )
            return f"✅ {n} chunk وارد دانش شد (Weaviate + KuzuDB)."
        except Exception as e:
            logger.error("ingest_text failed: %s", e, exc_info=True)
            return f"❌ خطا در ingest: {e}"

    async def manual_refresh(self, topic: str) -> str:
        if not topic or not topic.strip():
            return "⚠️ موضوعی وارد نشده است."
        try:
            from agents.state import AgentState
            if not self.orch.refresher._ready:
                await self.orch.refresher.startup()
            state = AgentState(query=topic.strip(), session_id="dashboard-manual")
            state.refresh_needed = True
            # throttle را دور بزن — refresh دستی همیشه باید اجرا شود
            self.orch.refresher._last_refresh.pop(
                self.orch.refresher._build_topic(state), None
            )
            await self.orch.refresher.refresh(state)
            m = state.metadata.get("last_refresh_metrics", {})
            if state.errors:
                return f"❌ خطا: {state.errors[-1]}"
            return (
                f"✅ Refresh کامل شد: {m.get('total_items', 0)} آیتم، "
                f"{m.get('chunks_ingested', 0)} chunk در {m.get('duration_seconds', 0):.1f}s"
            )
        except Exception as e:
            logger.error("manual_refresh failed: %s", e, exc_info=True)
            return f"❌ خطا در refresh: {e}"


# ── قالب گراف — canvas force layout بدون وابستگی خارجی ─────────────────────
_GRAPH_TEMPLATE = """<!doctype html><html><head><meta charset='utf-8'><style>
:root { --surface: #fcfcfb; --text: #0b0b0b; --muted: #52514e; --edge: #d5d4d0; }
@media (prefers-color-scheme: dark) {
  :root { --surface: #1a1a19; --text: #ffffff; --muted: #c3c2b7; --edge: #45443f; }
}
html,body { margin:0; background:var(--surface); font-family:-apple-system,sans-serif; }
#c { display:block; width:100vw; height:100vh; }
#tip { position:fixed; pointer-events:none; background:var(--surface); color:var(--text);
  border:1px solid var(--edge); border-radius:6px; padding:6px 10px; font-size:12px;
  display:none; max-width:260px; box-shadow:0 2px 8px rgba(0,0,0,.25); }
#legend { position:fixed; top:8px; left:8px; font-size:11px; color:var(--muted);
  background:var(--surface); border:1px solid var(--edge); border-radius:6px; padding:6px 10px; }
.li { display:flex; align-items:center; gap:6px; margin:2px 0; }
.sw { width:10px; height:10px; border-radius:50%; }
</style></head><body>
<canvas id='c'></canvas><div id='tip'></div><div id='legend'></div>
<script>
const DATA = __DATA__;
const dark = matchMedia('(prefers-color-scheme: dark)').matches;
const PALETTE = dark
  ? ['#3987e5','#199e70','#c98500','#008300','#9085e9','#e66767','#d55181','#d95926']
  : ['#2a78d6','#1baf7a','#eda100','#008300','#4a3aa7','#e34948','#e87ba4','#eb6834'];
const cv = document.getElementById('c'), ctx = cv.getContext('2d');
const tip = document.getElementById('tip');
let W, H, dpr = devicePixelRatio || 1;
function resize(){ W = innerWidth; H = innerHeight; cv.width = W*dpr; cv.height = H*dpr;
  cv.style.width = W+'px'; cv.style.height = H+'px'; ctx.setTransform(dpr,0,0,dpr,0,0); }
resize(); addEventListener('resize', resize);

// نوع → رنگ (ترتیب ثابت — هرگز چرخشی)
const types = [...new Set(DATA.nodes.map(n => n.type || 'Entity'))].sort();
const typeColor = {};
types.forEach((t,i) => typeColor[t] = PALETTE[Math.min(i, PALETTE.length-1)]);
const extra = types.length > PALETTE.length;
document.getElementById('legend').innerHTML = types.slice(0,PALETTE.length).map(t =>
  `<div class='li'><span class='sw' style='background:${typeColor[t]}'></span>${t}</div>`
).join('') + (extra ? `<div class='li'>… سایر انواع با رنگ آخر</div>` : '');

const nodes = DATA.nodes.map((n,i) => ({...n,
  x: W/2 + Math.cos(i*2.4)* (60 + i*2), y: H/2 + Math.sin(i*2.4)* (60 + i*1.5),
  vx:0, vy:0 }));
const byId = {}; nodes.forEach(n => byId[n.id] = n);
const edges = DATA.edges.filter(e => byId[e.src] && byId[e.dst]);
const deg = {}; edges.forEach(e => { deg[e.src]=(deg[e.src]||0)+1; deg[e.dst]=(deg[e.dst]||0)+1; });

let dragging = null, hover = null;
function tick(){
  // repulsion
  for (let i=0;i<nodes.length;i++) for (let j=i+1;j<nodes.length;j++){
    const a=nodes[i], b=nodes[j];
    let dx=b.x-a.x, dy=b.y-a.y, d2=dx*dx+dy*dy || 1, d=Math.sqrt(d2);
    const f = Math.min(1800/d2, 4);
    dx/=d; dy/=d; a.vx-=dx*f; a.vy-=dy*f; b.vx+=dx*f; b.vy+=dy*f;
  }
  // springs
  edges.forEach(e => { const a=byId[e.src], b=byId[e.dst];
    let dx=b.x-a.x, dy=b.y-a.y, d=Math.sqrt(dx*dx+dy*dy)||1;
    const f=(d-90)*0.01; dx/=d; dy/=d;
    a.vx+=dx*f; a.vy+=dy*f; b.vx-=dx*f; b.vy-=dy*f; });
  // center + integrate
  nodes.forEach(n => { if (n===dragging) return;
    n.vx += (W/2-n.x)*0.002; n.vy += (H/2-n.y)*0.002;
    n.vx*=0.85; n.vy*=0.85; n.x+=n.vx; n.y+=n.vy; });
}
function draw(){
  ctx.clearRect(0,0,W,H);
  ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--edge');
  ctx.lineWidth = 1;
  edges.forEach(e => { const a=byId[e.src], b=byId[e.dst];
    ctx.beginPath(); ctx.moveTo(a.x,a.y); ctx.lineTo(b.x,b.y); ctx.stroke(); });
  nodes.forEach(n => {
    const r = 5 + Math.min((deg[n.id]||0)*1.2, 9);
    ctx.beginPath(); ctx.arc(n.x,n.y,r,0,7);
    ctx.fillStyle = typeColor[n.type || 'Entity'] || PALETTE[0];
    ctx.fill();
    // حلقه ۲px هم‌رنگ سطح برای تفکیک گره‌های هم‌پوشان
    ctx.lineWidth = 2;
    ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--surface');
    ctx.stroke();
    if (n === hover || (deg[n.id]||0) > 2) {
      ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--text');
      ctx.font = '11px -apple-system, sans-serif';
      ctx.fillText((n.name||n.id).slice(0,24), n.x + r + 4, n.y + 4);
    }
  });
}
function loop(){ tick(); draw(); requestAnimationFrame(loop); }
loop();
function pick(mx,my){ return nodes.find(n => { const dx=n.x-mx, dy=n.y-my;
  return dx*dx+dy*dy < 180; }); }
cv.addEventListener('mousemove', ev => {
  const n = pick(ev.offsetX, ev.offsetY);
  hover = n || null;
  if (dragging){ dragging.x = ev.offsetX; dragging.y = ev.offsetY; }
  if (n){ tip.style.display='block';
    tip.style.left = (ev.clientX+12)+'px'; tip.style.top = (ev.clientY+12)+'px';
    tip.innerHTML = `<b>${n.name||n.id}</b><br>type: ${n.type||'—'}<br>degree: ${deg[n.id]||0}`;
    cv.style.cursor='pointer';
  } else { tip.style.display='none'; cv.style.cursor='default'; }
});
cv.addEventListener('mousedown', ev => { dragging = pick(ev.offsetX, ev.offsetY); });
addEventListener('mouseup', () => dragging = null);
</script></body></html>"""


def build_dashboard_tabs(orchestrator) -> None:
    """
    باید داخل یک gr.Blocks/gr.Tabs context صدا زده شود؛
    تب‌های dashboard را همان‌جا اضافه می‌کند.
    """
    svc = DashboardService(orchestrator)

    with gr.Tab("📊 Orchestrator"):
        gr.Markdown("### آمار فلوی Orchestrator")
        flow_tiles = gr.HTML()
        with gr.Row():
            with gr.Column():
                gr.Markdown("**میانگین زمان هر مرحله**")
                timing_bars = gr.HTML()
            with gr.Column():
                gr.Markdown("**اجرای Skill ها**")
                skill_tiles = gr.HTML()
        gr.Markdown("**تاریخچه کوئری‌ها**")
        flow_df = gr.Dataframe(
            headers=["زمان", "Query", "نوع", "استراتژی", "مرحله نهایی",
                     "Confidence", "تکرار", "Chunks", "خطاها"],
            interactive=False, wrap=True,
        )
        gr.Markdown("**تاریخچه Refresh دانش**")
        refresh_df = gr.Dataframe(
            headers=["زمان", "آیتم‌ها", "Chunks", "Nodes+", "Edges+", "مدت (s)"],
            interactive=False,
        )
        refresh_btn = gr.Button("🔄 بروزرسانی آمار", variant="primary")

        def _load_flow():
            return (
                svc.flow_stats_html(), svc.step_timings_html(),
                svc.skill_exec_html(), svc.flow_table(), svc.refresh_history_table(),
            )

        refresh_btn.click(
            _load_flow, outputs=[flow_tiles, timing_bars, skill_tiles, flow_df, refresh_df]
        )

    with gr.Tab("🕸 Knowledge Graph"):
        gr.Markdown("### گراف دانش")
        kg_tiles = gr.HTML()
        with gr.Row():
            kg_filter = gr.Textbox(
                label="فیلتر entity (نام)", placeholder="مثلاً: Iran", scale=3
            )
            kg_limit = gr.Slider(20, 300, value=120, step=20, label="حداکثر گره", scale=1)
            kg_btn = gr.Button("🕸 نمایش گراف", variant="primary", scale=1)
        kg_view = gr.HTML()

        async def _load_graph(f, lim):
            return await svc.graph_stats_html(), await svc.graph_html(f, int(lim))

        kg_btn.click(_load_graph, inputs=[kg_filter, kg_limit], outputs=[kg_tiles, kg_view])

    with gr.Tab("🧠 Knowledge & Skills"):
        gr.Markdown("### مدیریت دانش")
        with gr.Row():
            with gr.Column():
                gr.Markdown("**افزودن دانش دستی** — متن به chunk تبدیل و در Weaviate + KuzuDB ذخیره می‌شود")
                ing_text = gr.Textbox(label="متن", lines=6)
                ing_src = gr.Textbox(label="نام منبع", value="manual")
                ing_btn = gr.Button("📥 Ingest", variant="primary")
                ing_out = gr.Markdown()
                ing_btn.click(svc.ingest_text, inputs=[ing_text, ing_src], outputs=ing_out)
            with gr.Column():
                gr.Markdown("**Refresh دستی از منابع خارجی** (RSS، Wikipedia، ...)")
                rf_topic = gr.Textbox(label="موضوع", placeholder="مثلاً: Iran")
                rf_btn = gr.Button("🌐 جمع‌آوری و Ingest", variant="primary")
                rf_out = gr.Markdown()
                rf_btn.click(svc.manual_refresh, inputs=rf_topic, outputs=rf_out)

        gr.Markdown("### Skill ها و ابزارها")
        skills_df = gr.Dataframe(
            headers=["نام", "دسته", "توضیح"], interactive=False, wrap=True
        )
        skills_btn = gr.Button("🔄 بارگذاری لیست skill ها")
        skills_btn.click(svc.skills_table, outputs=skills_df)
