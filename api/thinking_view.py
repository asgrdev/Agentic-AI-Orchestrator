# api/thinking_view.py
"""
ساخت متن «🧠 فرایند فکر» برای نمایش زنده در چت — مثل ChatGPT/DeepSeek.

ورودی: AgentState (در حال اجرا یا تمام‌شده)
خروجی: markdown شامل مراحل فلو + تصمیم‌های orchestrator + فکر مدل‌ها
        (thinking token ها و reasoning trace)
"""
from __future__ import annotations

from typing import Any

_STEP_FA = {
    "start":              ("🚀", "شروع"),
    "shortcut":           ("⚡", "پاسخ مستقیم"),
    "direct_chat":        ("⚡", "پاسخ مستقیم"),
    "direct_calculation": ("🧮", "محاسبه مستقیم"),
    "files":              ("📎", "پردازش فایل"),
    "understand":         ("🧭", "فهم و برنامه‌ریزی"),
    "retrieve":           ("🔎", "بازیابی دانش"),
    "assess":             ("⚖️", "ارزیابی کفایت"),
    "refresh":            ("🌐", "جمع‌آوری دانش تازه"),
    "reason":             ("💡", "استدلال"),
    "validate":           ("✔️", "اعتبارسنجی"),
    "answer":             ("🏁", "پاسخ نهایی"),
    "error":              ("❌", "خطا"),
}

_STAGE_FA = {
    "classify":       "🧭 طبقه‌بندی",
    "plan":           "🗺️ برنامه",
    "shortcut":       "⚡ میان‌بر",
    "model_thinking": "💭 فکر مدل",
    "file":           "📎 فایل",
    "answer":         "💭 فکر مدل",
}


def _fmt_dur(sec: float) -> str:
    if sec >= 60:
        return f"{int(sec // 60)}m{sec % 60:.0f}s"
    if sec >= 1:
        return f"{sec:.1f}s"
    if sec >= 0.05:
        return f"{sec * 1000:.0f}ms"
    return ""


def compose_thinking(state: Any, running: bool | None = None) -> str:
    """markdown فرایند فکر از روی state (flow_trace + thinking_log + reasoning)"""
    if state is None:
        return "…"

    if running is None:
        try:
            from agents.state import FlowStep
            running = state.current_step not in (FlowStep.ANSWER, FlowStep.ERROR)
        except Exception:
            running = False

    lines: list[str] = []

    # ── مراحل طی‌شده ──
    for e in getattr(state, "flow_trace", []) or []:
        step = e.get("step", "")
        if step == "start":
            continue
        icon, fa = _STEP_FA.get(step, ("•", step))
        dur = _fmt_dur(float(e.get("duration") or 0))
        detail = str(e.get("detail") or "")
        mark = "❌" if e.get("status") == "error" else "✅"
        seg = f"{mark} {icon} **{fa}**"
        if dur:
            seg += f" ({dur})"
        if detail:
            seg += f" — {detail}"
        lines.append(seg)

    # ── مرحله‌ی در حال اجرا ──
    if running:
        cur = getattr(getattr(state, "current_step", None), "value", "")
        done_steps = {e.get("step") for e in getattr(state, "flow_trace", [])}
        if cur and cur not in ("answer", "error") and cur not in done_steps:
            icon, fa = _STEP_FA.get(cur, ("•", cur))
            lines.append(f"⏳ {icon} **{fa}** — در حال اجرا…")

    # ── تصمیم‌ها و فکر مدل‌ها ──
    thoughts: list[str] = []
    for t in getattr(state, "thinking_log", []) or []:
        stage = _STAGE_FA.get(t.get("stage", ""), t.get("stage", ""))
        text = str(t.get("text", "")).strip()
        if text:
            thoughts.append(f"**{stage}:** {text}")

    # reasoning trace از ReasonerAgent (chain-of-thought مدل)
    for r in getattr(state, "reasoning_trace", []) or []:
        thought = str(r.get("thought", "")).strip()
        if thought:
            thoughts.append(f"**💡 گام استدلال {r.get('step', '')}:** {thought[:700]}")

    parts = []
    if lines:
        parts.append("\n".join(lines))
    if thoughts:
        parts.append("\n\n".join(thoughts))
    return "\n\n---\n\n".join(parts) if parts else "…"


def thinking_title(state: Any, running: bool) -> str:
    total = sum(
        float(e.get("duration") or 0)
        for e in getattr(state, "flow_trace", []) or []
    )
    dur = _fmt_dur(total)
    if running:
        return f"🧠 در حال فکر کردن… ({dur})" if dur else "🧠 در حال فکر کردن…"
    return f"🧠 فرایند فکر ({dur})" if dur else "🧠 فرایند فکر"
