# api/flow_visualizer.py
"""
نمایش گراف اجرای فلو — شبیه LangGraph/LangSmith.

- render_flow_html(entry):  نمای کامل — هدر آمار + گراف SVG + جدول مراحل
- render_flow_strip(...):   نوار فشرده‌ی زنده برای بالای صفحه‌ی چت
- build_flow_tab(orch):     تب «Flow» با انتخاب کوئری، auto-refresh و export
- build_chat_flow_strip():  نوار زنده داخل تب Chat

ویژگی‌ها: گره‌های طی‌شده سبز با گرادیان و سایه، گره‌ی در حال اجرا نارنجیِ
چشمک‌زن، خطا قرمز، شماره‌ی ترتیب روی یال‌های طی‌شده، tooltip روی هر گره،
نشان ×N برای حلقه‌ها، جدول مراحل با نوار سهم زمانی. بدون وابستگی خارجی.
"""
from __future__ import annotations

import html
import time
from typing import Any, Optional

# ── چیدمان ثابت گراف (مرکز جعبه‌ها) ─────────────────────────────────
_NODE_W, _NODE_H = 128, 48

_NODES: dict[str, tuple[int, int, str, str]] = {
    # name: (cx, cy, label, توضیح tooltip)
    "start":      (75,  225, "START",         "ورود پرامپت"),
    "shortcut":   (245, 360, "DIRECT ANSWER", "میان‌بر: پاسخ مستقیم بدون RAG (گفت‌وگو/محاسبه)"),
    "files":      (430, 360, "FILE ANALYSIS", "پردازش فایل آپلودشده با skill ها (YOLO/Whisper/docling)"),
    "understand": (245, 225, "UNDERSTAND",    "طبقه‌بندی کوئری + فهم و برنامه‌ریزی (phi3)"),
    "retrieve":   (420, 225, "RETRIEVE",      "بازیابی هیبریدی: Weaviate + گراف KuzuDB"),
    "assess":     (590, 225, "ASSESS",        "ارزیابی کفایت دانش بازیابی‌شده"),
    "refresh":    (505, 100, "REFRESH",       "جمع‌آوری دانش تازه: ویکی‌پدیا/RSS/arXiv"),
    "reason":     (760, 225, "REASON",        "استدلال و تولید پاسخ (granite)"),
    "validate":   (925, 225, "VALIDATE",      "اعتبارسنجی پاسخ"),
    "answer":     (1080, 225, "ANSWER",       "پاسخ نهایی"),
    "error":      (925, 360, "ERROR",         "خطا در اجرای فلو"),
}

# یال‌های تعریف‌شده: (from, to) → نقطه‌ی کنترل منحنی (None = خط مستقیم)
_EDGES: dict[tuple[str, str], Optional[tuple[int, int]]] = {
    ("start", "understand"):      None,
    ("start", "shortcut"):        None,
    ("shortcut", "answer"):       (700, 445),
    ("start", "files"):           (150, 320),
    ("files", "answer"):          (800, 430),
    ("understand", "retrieve"):   None,
    ("understand", "reason"):     (505, 320),
    ("understand", "answer"):     (660, 30),
    ("retrieve", "assess"):       None,
    ("retrieve", "reason"):       (590, 305),
    ("assess", "reason"):         None,
    ("assess", "refresh"):        None,
    ("refresh", "retrieve"):      None,
    ("reason", "validate"):       None,
    ("reason", "answer"):         (925, 130),
    ("validate", "answer"):       None,
    ("validate", "retrieve"):     (670, 440),
    ("validate", "refresh"):      (720, 40),
    ("validate", "error"):        None,
    ("reason", "error"):          None,
}

_C_EDGE     = "#454b5c"
_C_WALK     = "#34d399"
_C_TEXT     = "#f1f5f9"
_C_TEXT_DIM = "#94a3b8"


def _edge_path(a: str, b: str) -> str:
    ax, ay = _NODES[a][0], _NODES[a][1]
    bx, by = _NODES[b][0], _NODES[b][1]
    ctrl = _EDGES.get((a, b))
    if ctrl:
        return f"M {ax} {ay} Q {ctrl[0]} {ctrl[1]} {bx} {by}"
    return f"M {ax} {ay} L {bx} {by}"


def _edge_mid(a: str, b: str) -> tuple[float, float]:
    """نقطه‌ی میانی یال (برای نشان ترتیب) — منحنی درجه‌دو: M=(A+2C+B)/4"""
    ax, ay = _NODES[a][0], _NODES[a][1]
    bx, by = _NODES[b][0], _NODES[b][1]
    ctrl = _EDGES.get((a, b))
    if ctrl:
        return (ax + 2 * ctrl[0] + bx) / 4, (ay + 2 * ctrl[1] + by) / 4
    return (ax + bx) / 2, (ay + by) / 2


def _normalize_step(step: str) -> str:
    return "shortcut" if step in ("direct_chat", "direct_calculation") else step


def _fmt_dur(sec: float) -> str:
    if sec >= 60:
        return f"{int(sec // 60)}m{sec % 60:.0f}s"
    if sec >= 1:
        return f"{sec:.1f}s"
    if sec >= 0.05:
        return f"{sec * 1000:.0f}ms"
    return "—"


# ═══════════════════════════════════════════════════════════════════
# SVG گراف
# ═══════════════════════════════════════════════════════════════════

def render_flow_svg(
    trace: list[dict],
    current_step: Optional[str] = None,
    running: bool = False,
) -> str:
    steps = [_normalize_step(e.get("step", "")) for e in trace]
    steps = [s for s in steps if s in _NODES]

    visits: dict[str, dict] = {}
    for e in trace:
        s = _normalize_step(e.get("step", ""))
        if s not in _NODES:
            continue
        v = visits.setdefault(s, {"count": 0, "duration": 0.0, "error": False, "detail": ""})
        v["count"] += 1
        v["duration"] += float(e.get("duration") or 0.0)
        v["error"] = v["error"] or e.get("status") == "error"
        if e.get("detail"):
            v["detail"] = str(e["detail"])

    # یال‌های طی‌شده به‌همراه ترتیب عبور
    walked: dict[tuple[str, str], list[int]] = {}
    order = 0
    for a, b in zip(steps, steps[1:]):
        if a == b:
            continue
        order += 1
        walked.setdefault((a, b), []).append(order)

    cur = _normalize_step(current_step or "")

    p: list[str] = []
    p.append(
        '<svg viewBox="0 0 1170 470" xmlns="http://www.w3.org/2000/svg" '
        'style="width:100%;max-width:1250px;background:'
        'linear-gradient(160deg,#0d1017 0%,#141926 100%);'
        'border:1px solid #262c3d;border-radius:14px;">'
    )
    p.append(
        "<style>"
        ".pulse{animation:fp 1.1s ease-in-out infinite}"
        "@keyframes fp{0%,100%{opacity:1}50%{opacity:.35}}"
        ".flow{stroke-dasharray:10 6;animation:fd 1s linear infinite}"
        "@keyframes fd{to{stroke-dashoffset:-16}}"
        "text{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}"
        "</style>"
    )
    p.append(
        "<defs>"
        '<linearGradient id="gVisited" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0%" stop-color="#14532d"/><stop offset="100%" stop-color="#166534"/></linearGradient>'
        '<linearGradient id="gCurrent" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0%" stop-color="#b45309"/><stop offset="100%" stop-color="#92400e"/></linearGradient>'
        '<linearGradient id="gIdle" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0%" stop-color="#343a4a"/><stop offset="100%" stop-color="#2a2f3d"/></linearGradient>'
        '<linearGradient id="gError" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0%" stop-color="#991b1b"/><stop offset="100%" stop-color="#7f1d1d"/></linearGradient>'
        '<filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">'
        '<feDropShadow dx="0" dy="2.5" stdDeviation="3" flood-color="#000" flood-opacity="0.5"/></filter>'
        '<marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6.5" markerHeight="6.5" '
        'orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="' + _C_EDGE + '"/></marker>'
        '<marker id="arrw" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
        'orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="' + _C_WALK + '"/></marker>'
        "</defs>"
    )

    # ── یال‌ها ──
    for (a, b) in _EDGES:
        d = _edge_path(a, b)
        if (a, b) in walked:
            animated = " flow" if running and b == cur else ""
            p.append(
                f'<path d="{d}" fill="none" stroke="{_C_WALK}" stroke-width="3" '
                f'marker-end="url(#arrw)" opacity="0.95" class="edge{animated}"/>'
            )
        else:
            p.append(
                f'<path d="{d}" fill="none" stroke="{_C_EDGE}" '
                f'stroke-width="1.3" marker-end="url(#arr)" opacity="0.65"/>'
            )
    for (a, b) in walked:  # یال‌های خارج از گراف ثابت (نادر) — خط‌چین
        if (a, b) not in _EDGES:
            p.append(
                f'<path d="{_edge_path(a, b)}" fill="none" stroke="{_C_WALK}" '
                f'stroke-width="2" stroke-dasharray="6 5" marker-end="url(#arrw)"/>'
            )

    # نشان ترتیب عبور روی یال‌ها (۱، ۲، ۳…)
    for (a, b), orders in walked.items():
        mx, my = _edge_mid(a, b)
        label = ",".join(str(o) for o in orders[:3])
        r = 11 if len(label) <= 2 else 15
        p.append(
            f'<circle cx="{mx:.0f}" cy="{my:.0f}" r="{r}" fill="#0d1017" '
            f'stroke="{_C_WALK}" stroke-width="1.6"/>'
            f'<text x="{mx:.0f}" y="{my + 4:.0f}" font-size="11" font-weight="bold" '
            f'fill="{_C_WALK}" text-anchor="middle">{label}</text>'
        )

    # ── گره‌ها ──
    for name, (cx, cy, label, desc) in _NODES.items():
        x, y = cx - _NODE_W // 2, cy - _NODE_H // 2
        v = visits.get(name)
        is_current = running and name == cur

        if (v and v["error"]) or (name == "error" and v):
            fill, stroke = "url(#gError)", "#ef4444"
        elif is_current:
            fill, stroke = "url(#gCurrent)", "#f59e0b"
        elif v:
            fill, stroke = "url(#gVisited)", _C_WALK
        else:
            fill, stroke = "url(#gIdle)", "#4c5366"

        tip = desc
        if v:
            tip += f" | بازدید: {v['count']} | زمان: {_fmt_dur(v['duration'])}"
            if v["detail"]:
                tip += f" | {v['detail']}"

        cls = ' class="pulse"' if is_current else ""
        p.append(
            f'<g{cls}><title>{html.escape(tip)}</title>'
            f'<rect x="{x}" y="{y}" width="{_NODE_W}" height="{_NODE_H}" rx="11" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="2" filter="url(#shadow)"/>'
        )
        p.append(
            f'<text x="{cx}" y="{cy - 3}" font-size="13" font-weight="bold" '
            f'fill="{_C_TEXT}" text-anchor="middle">{label}</text>'
        )
        sub = "در حال اجرا…" if is_current else (_fmt_dur(v["duration"]) if v else "")
        if sub == "—":
            sub = "✓"
        if sub:
            p.append(
                f'<text x="{cx}" y="{cy + 15}" font-size="11" '
                f'fill="{_C_TEXT_DIM}" text-anchor="middle">{html.escape(sub)}</text>'
            )
        if v and v["count"] > 1:
            bx, by = x + _NODE_W - 4, y + 4
            p.append(
                f'<circle cx="{bx}" cy="{by}" r="11" fill="{_C_WALK}"/>'
                f'<text x="{bx}" y="{by + 4}" font-size="11" font-weight="bold" '
                f'fill="#08210f" text-anchor="middle">×{v["count"]}</text>'
            )
        p.append("</g>")

    p.append("</svg>")
    return "".join(p)


# ═══════════════════════════════════════════════════════════════════
# نمای کامل: هدر آمار + گراف + جدول مراحل
# ═══════════════════════════════════════════════════════════════════

def _badge(label: str, value: str, color: str = "#334155") -> str:
    return (
        f'<span style="background:{color};color:#e2e8f0;border-radius:8px;'
        f'padding:3px 10px;font-size:12px;font-family:monospace;white-space:nowrap">'
        f'{html.escape(label)}: <b>{html.escape(value)}</b></span>'
    )


def render_flow_html(entry: dict) -> str:
    trace = entry.get("trace") or []
    running = bool(entry.get("running"))
    query = str(entry.get("query", ""))

    total = sum(float(e.get("duration") or 0) for e in trace)
    n_steps = sum(1 for e in trace if _normalize_step(e.get("step", "")) not in ("", "start"))
    has_err = any(e.get("status") == "error" for e in trace)

    badges = [
        _badge("مراحل", str(n_steps)),
        _badge("زمان کل", _fmt_dur(total)),
    ]
    if entry.get("query_type"):
        badges.append(_badge("نوع", str(entry["query_type"]), "#1e3a5f"))
    if entry.get("strategy"):
        badges.append(_badge("استراتژی", str(entry["strategy"]), "#3b2f5f"))
    if entry.get("confidence") is not None:
        badges.append(_badge("confidence", f"{float(entry['confidence']):.2f}", "#14532d"))
    if running:
        badges.append(_badge("وضعیت", "⏳ در حال اجرا", "#92400e"))
    elif has_err:
        badges.append(_badge("وضعیت", "❌ خطا", "#7f1d1d"))
    else:
        badges.append(_badge("وضعیت", "✅ کامل", "#14532d"))

    header = (
        '<div style="display:flex;flex-wrap:wrap;gap:8px;align-items:center;'
        'margin:4px 0 10px 0;font-family:sans-serif">'
        f'<span style="color:#e2e8f0;font-size:14px;font-weight:bold;direction:auto">'
        f'💬 {html.escape(query[:120])}</span>'
        f'<span style="flex:1"></span>{"".join(badges)}</div>'
    )

    svg = render_flow_svg(trace, entry.get("current"), running)

    # جدول مراحل با نوار سهم زمانی
    rows = []
    idx = 0
    for e in trace:
        s = _normalize_step(e.get("step", ""))
        if s not in _NODES or s == "start":
            continue
        idx += 1
        dur = float(e.get("duration") or 0)
        pct = (dur / total * 100) if total > 0 else 0
        is_err = e.get("status") == "error"
        color = "#ef4444" if is_err else "#34d399"
        detail = html.escape(str(e.get("detail") or ""))
        ts = e.get("ts")
        tstr = time.strftime("%H:%M:%S", time.localtime(ts)) if ts else ""
        rows.append(
            f'<tr style="border-bottom:1px solid #232936">'
            f'<td style="padding:5px 10px;color:#64748b">{idx}</td>'
            f'<td style="padding:5px 10px;color:#e2e8f0;font-weight:bold">'
            f'{html.escape(_NODES[s][2])}{" ❌" if is_err else ""}</td>'
            f'<td style="padding:5px 10px;color:#94a3b8">{tstr}</td>'
            f'<td style="padding:5px 10px;color:#e2e8f0">{_fmt_dur(dur)}</td>'
            f'<td style="padding:5px 10px;min-width:140px">'
            f'<div style="background:#1c212e;border-radius:4px;height:9px;overflow:hidden">'
            f'<div style="width:{pct:.0f}%;height:100%;background:{color}"></div></div></td>'
            f'<td style="padding:5px 10px;color:#94a3b8;font-size:12px">{detail}</td></tr>'
        )
    table = ""
    if rows:
        table = (
            '<table style="width:100%;max-width:1250px;border-collapse:collapse;'
            'font-family:ui-monospace,monospace;font-size:13px;margin-top:10px;'
            'background:#11151f;border:1px solid #262c3d;border-radius:10px">'
            '<thead><tr style="color:#64748b;text-align:left;border-bottom:1px solid #2c3345">'
            '<th style="padding:6px 10px">#</th><th style="padding:6px 10px">گره</th>'
            '<th style="padding:6px 10px">ساعت</th><th style="padding:6px 10px">زمان</th>'
            '<th style="padding:6px 10px">سهم زمانی</th><th style="padding:6px 10px">جزئیات</th>'
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
        )

    return f'<div dir="ltr">{header}{svg}{table}</div>'


# ═══════════════════════════════════════════════════════════════════
# نوار فشرده‌ی زنده — بالای صفحه‌ی چت
# ═══════════════════════════════════════════════════════════════════

_STRIP_EMPTY = (
    '<div style="display:flex;align-items:center;gap:8px;padding:8px 14px;'
    'background:#12151d;border:1px solid #262c3d;border-radius:10px;'
    'font-family:ui-monospace,monospace;font-size:12px;color:#64748b">'
    "🔀 مسیر اجرای پرامپت این‌جا زنده نمایش داده می‌شود…</div>"
)


def render_flow_strip(
    trace: list[dict],
    current_step: Optional[str],
    running: bool,
    query: str = "",
) -> str:
    pills = []
    steps = [(e, _normalize_step(e.get("step", ""))) for e in trace]
    steps = [(e, s) for e, s in steps if s in _NODES and s != "start"]

    for e, s in steps[-9:]:
        is_err = e.get("status") == "error"
        bg, bd = ("#7f1d1d", "#ef4444") if is_err else ("#14532d", "#34d399")
        dur = _fmt_dur(float(e.get("duration") or 0))
        dur_html = f' <span style="opacity:.6">{dur}</span>' if dur != "—" else ""
        pills.append(
            f'<span style="background:{bg};border:1px solid {bd};color:#e2e8f0;'
            f'border-radius:7px;padding:2px 9px;white-space:nowrap">'
            f"{html.escape(_NODES[s][2])}{dur_html}</span>"
        )

    cur = _normalize_step(current_step or "")
    last_step = steps[-1][1] if steps else ""
    if running and cur in _NODES and cur != last_step:
        pills.append(
            f'<span class="fstrip-pulse" style="background:#92400e;border:1px solid #f59e0b;'
            f'color:#fff;border-radius:7px;padding:2px 9px;white-space:nowrap">'
            f"⏳ {html.escape(_NODES[cur][2])}</span>"
        )

    total = sum(float(e.get("duration") or 0) for e in trace)
    status = "⏳" if running else "✅"
    sep = '<span style="color:#3f4759">→</span>'
    return (
        "<style>.fstrip-pulse{animation:fsp 1.1s ease-in-out infinite}"
        "@keyframes fsp{0%,100%{opacity:1}50%{opacity:.4}}</style>"
        '<div dir="ltr" style="display:flex;align-items:center;gap:7px;padding:8px 14px;'
        'background:#12151d;border:1px solid #262c3d;border-radius:10px;'
        'font-family:ui-monospace,monospace;font-size:12px;overflow-x:auto">'
        f'<span style="color:#64748b;white-space:nowrap">🔀 {status} {_fmt_dur(total)}</span>'
        + sep.join(pills)
        + f'<span style="flex:1"></span>'
          f'<span style="color:#475569;white-space:nowrap;direction:auto">'
          f"{html.escape(query[:45])}</span></div>"
    )


# ═══════════════════════════════════════════════════════════════════
# Gradio wiring
# ═══════════════════════════════════════════════════════════════════

_EMPTY_MSG = (
    '<div style="padding:24px;color:#9ca3af;font-family:sans-serif">'
    "هنوز کوئری‌ای اجرا نشده — یک پیام در تب Chat بفرستید؛ مسیر اجرای آن "
    "همین‌جا به‌صورت زنده روشن می‌شود. (تاریخچه‌ی سشن‌های قبلی هم بعد از "
    "اولین اجرا این‌جا لود می‌شود)</div>"
)


def _collect_entries(orch) -> list[dict]:
    """live state + تاریخچه (جدیدترین اول) — شامل سشن‌های لودشده از دیسک"""
    entries: list[dict] = []

    st = getattr(orch, "active_state", None)
    if st is not None and getattr(st, "flow_trace", None):
        from agents.state import FlowStep
        running = st.current_step not in (FlowStep.ANSWER, FlowStep.ERROR)
        entries.append({
            "label": ("⏳ در حال اجرا: " if running else "🟢 آخرین اجرا: ") + st.query[:60],
            "trace": list(st.flow_trace),
            "current": st.current_step.value,
            "running": running,
            "query": st.query,
            "confidence": st.confidence,
            "strategy": getattr(st, "strategy", ""),
        })

    for h in reversed(getattr(orch, "flow_history", []) or []):
        if not h.get("trace"):
            continue
        ts = time.strftime("%m-%d %H:%M:%S", time.localtime(h.get("timestamp", 0)))
        entries.append({
            "label": f"{ts} — {h.get('query', '')[:55]}",
            "trace": h["trace"],
            "current": h.get("final_step", ""),
            "running": False,
            "query": h.get("query", ""),
            "confidence": h.get("confidence"),
            "strategy": h.get("strategy", ""),
            "query_type": h.get("query_type", ""),
            "record": h,
        })
    return entries


def build_flow_tab(orch) -> None:
    """ساخت تب «Flow» داخل gr.Blocks جاری"""
    import gradio as gr
    from core.session_store import get_session_store

    def refresh(selected: int):
        entries = _collect_entries(orch)
        if not entries:
            return gr.update(choices=[("—", 0)], value=0), _EMPTY_MSG
        choices = [(e["label"], i) for i, e in enumerate(entries)]
        idx = selected if isinstance(selected, int) and 0 <= selected < len(entries) else 0
        return gr.update(choices=choices, value=idx), render_flow_html(entries[idx])

    def export(selected: int):
        entries = _collect_entries(orch)
        if not entries:
            return gr.update(visible=False)
        idx = selected if isinstance(selected, int) and 0 <= selected < len(entries) else 0
        e = entries[idx]
        record = e.get("record") or {
            "query": e["query"], "trace": e["trace"],
            "confidence": e.get("confidence"), "strategy": e.get("strategy"),
            "timestamp": time.time(),
        }
        svg = render_flow_svg(e["trace"], e.get("current"), e.get("running", False))
        paths = get_session_store().export(record, svg=svg)
        return gr.update(value=paths, visible=True)

    with gr.Tab("🔀 Flow"):
        gr.Markdown(
            "### 🔀 گراف اجرای پرامپت\n"
            "مسیر عبور هر پرامپت از گره‌های سیستم — گره‌های طی‌شده سبز با زمان، "
            "گره‌ی در حال اجرا نارنجیِ چشمک‌زن، شماره روی یال‌ها = ترتیب عبور. "
            "روی هر گره hover کنید تا جزئیات را ببینید. تاریخچه روی دیسک "
            "(`data/sessions/`) ذخیره می‌شود و بعد از restart می‌ماند."
        )
        with gr.Row():
            selector = gr.Dropdown(label="کوئری", choices=[("—", 0)], value=0, scale=5)
            refresh_btn = gr.Button("🔄 تازه‌سازی", scale=1)
            export_btn = gr.Button("💾 Export (JSON+SVG)", scale=1)
        graph_html = gr.HTML(_EMPTY_MSG)
        export_files = gr.File(label="فایل‌های export شده", visible=False)

        selector.change(refresh, selector, [selector, graph_html])
        refresh_btn.click(refresh, selector, [selector, graph_html])
        export_btn.click(export, selector, export_files)

        # به‌روزرسانی خودکار — gr.Timer در gradio>=5؛ در نسخه‌های قدیمی‌تر
        # دکمه‌ی تازه‌سازی دستی کار می‌کند
        try:
            timer = gr.Timer(2.0)
            timer.tick(refresh, selector, [selector, graph_html])
        except AttributeError:
            pass


def build_chat_flow_strip(orch) -> None:
    """نوار زنده‌ی فلو — بالای ChatInterface در تب Chat"""
    import gradio as gr

    def tick():
        st = getattr(orch, "active_state", None)
        if st is None or not getattr(st, "flow_trace", None):
            return _STRIP_EMPTY
        from agents.state import FlowStep
        running = st.current_step not in (FlowStep.ANSWER, FlowStep.ERROR)
        return render_flow_strip(
            st.flow_trace, st.current_step.value, running, st.query
        )

    strip = gr.HTML(_STRIP_EMPTY)
    try:
        timer = gr.Timer(1.0)
        timer.tick(tick, None, strip)
    except AttributeError:
        pass
