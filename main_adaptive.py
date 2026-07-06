from pathlib import Path
import logging
import sys

# logging باید قبل از import ماژول‌هایی که در import-time لاگ می‌زنند تنظیم شود
from core.logger import setup_logging
setup_logging()

logger = logging.getLogger(__name__)

import gradio as gr
from agents.adaptive_orchestrator import AdaptiveOrchestrator
from configs.main_config import CONFIG
from core.model_manager import get_model_manager, ModelConfig
from core.model_gate import configure_model_gate
from core.model_paths import resolve_model_path
from core.memory_monitor import get_memory_monitor


_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import fitz  # PyMuPDF برای PDF

# نکته: اگر از PyMuPDF برای PDF استفاده می‌کنید، مطمئن شوید نصب است.
# برای فایل‌های ساده متنی نیازی به کتابخانه خاصی نیست.

_DEFAULT_SCHEMA_HINT = """Return STRICT JSON with keys:
{
  "chosen_intent": string,
  "entities": [{"type": string, "value": string, "confidence": number}],
  "constraints": {"cpu_first": bool, "max_latency_ms": number|null, "max_tokens_in": number|null, "max_tokens_out": number|null, "language": string|null, "safety": string},
  "need_context": {"vector": bool, "keyword": bool, "graph": bool, "rerank": bool},
  "ambiguity": {"is_ambiguous": bool, "questions": [string]}
}
No extra keys. JSON only.
"""

def read_pdf_file(file_path):
    """محتوای فایل PDF را بر اساس مسیر فایل می‌خواند."""
    try:
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return f"\n[محتوای فایل PDF: {text}]\n"
    except Exception as e:
        return f"\n[خطا در خواندن PDF {Path(file_path).name}: {str(e)}]\n"

def read_text_file(file_path):
    """محتوای فایل متنی را می‌خواند."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return f"\n[محتوای فایل متنی: {content}]\n"
    except Exception as e:
        return f"\n[خطا در خواندن فایل متنی {Path(file_path).name}: {str(e)}]\n"

# ── نگاشت پسوند فایل → skill پردازش ─────────────────────────────────
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif", ".tiff"}
_AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}
_DOC_SKILL_BY_EXT = {
    ".pdf":  "read_pdf_with_docling",
    ".docx": "read_docx_file",
    ".xlsx": "read_excel_file",
    ".xls":  "read_excel_file",
    ".csv":  "read_csv_file",
    ".md":   "read_markdown_file",
}
_TEXT_EXTS = {".txt", ".json", ".py", ".js"}

# «این pdf را به متن تبدیل کن» و مشابه‌ها → خودِ متن استخراج‌شده پاسخ است
import re as _re
_EXTRACT_TEXT_RE = _re.compile(
    r"(تبدیل.{0,20}(متن|تکست|text))|(استخراج.{0,10}متن)|"
    r"(convert.{0,20}(text|txt))|(extract.{0,10}text)|(to\s+text)",
    _re.IGNORECASE,
)


async def analyze_uploaded_file(file_path: str) -> str:
    """
    پردازش فایل آپلودشده با skill مناسب:
    تصویر → YOLO + طبقه‌بندی صحنه | صوت → Whisper | سند → docling/…
    خروجی: متن context برای مدل پاسخ‌دهنده.
    """
    path = Path(file_path)
    ext = path.suffix.lower()
    executor = GLOBAL_ADAPTIVE_ORCH.skill_executor

    if ext in _IMAGE_EXTS:
        parts = [f"[تصویر: {path.name}]"]
        det = await executor.execute_skill(
            "detect_objects_in_image", {"image_path": str(path)}
        )
        out = det.output if det.success and isinstance(det.output, dict) else {}
        if out.get("success"):
            objs = out.get("objects", [])
            if objs:
                labels = []
                for o in objs[:15]:
                    if isinstance(o, dict):
                        label = o.get("label") or o.get("class") or o.get("name") or "?"
                        conf = o.get("confidence") or o.get("conf")
                        labels.append(
                            f"{label} ({conf:.0%})" if isinstance(conf, float) else str(label)
                        )
                    else:
                        labels.append(str(o))
                parts.append(f"اشیاء تشخیص‌داده‌شده با YOLO: {', '.join(labels)}")
            else:
                parts.append("YOLO شیء مشخصی در تصویر پیدا نکرد.")
        else:
            parts.append(f"(تشخیص اشیاء ناموفق: {out.get('error') or det.error})")

        scn = await executor.execute_skill(
            "classify_image_scene", {"image_path": str(path)}
        )
        sout = scn.output if scn.success and isinstance(scn.output, dict) else {}
        if sout.get("success") and sout.get("scene") not in (None, "", "unknown"):
            conf = sout.get("confidence") or 0.0
            parts.append(f"صحنه تصویر: {sout['scene']} (اطمینان {conf:.0%})")
        return "\n".join(parts) + "\n"

    if ext in _AUDIO_EXTS:
        tr = await executor.execute_skill(
            "transcribe_audio_file", {"audio_path": str(path)}
        )
        out = tr.output if tr.success and isinstance(tr.output, dict) else {}
        if out.get("success") and out.get("text"):
            return (
                f"[فایل صوتی: {path.name}]\n"
                f"متن پیاده‌شده با Whisper (زبان={out.get('language', '?')}):\n"
                f"{out['text']}\n"
            )
        return f"[فایل صوتی: {path.name}] رونویسی ناموفق: {out.get('error') or tr.error}\n"

    if ext in _DOC_SKILL_BY_EXT:
        res = await executor.execute_skill(
            _DOC_SKILL_BY_EXT[ext], {"file_path": str(path)}
        )
        out = res.output if res.success and isinstance(res.output, dict) else {}
        if out.get("success"):
            text = out.get("text") or out.get("content") or ""
            if not text and out.get("data") is not None:
                text = str(out["data"])[:4000]
            tables = out.get("tables") or []
            extra = f" (+{len(tables)} جدول)" if tables else ""
            method = out.get("method", _DOC_SKILL_BY_EXT[ext])
            return f"[سند: {path.name} | روش: {method}]{extra}\n{text[:8000]}\n"
        if ext == ".pdf":
            return read_pdf_file(str(path))  # fallback ساده با PyMuPDF
        return f"[خطا در پردازش سند {path.name}: {out.get('error') or res.error}]\n"

    if ext in _TEXT_EXTS:
        return read_text_file(str(path))

    return f"\n[نوع فایل پشتیبانی نشده: {ext}]\n"


# ── نمایش «فرایند فکر» در چت (مثل ChatGPT/DeepSeek) ─────────────────
try:
    from gradio import ChatMessage as GrChatMessage
except Exception:
    GrChatMessage = None

# اگر gradio از type="messages" پشتیبانی نکند، در پایین False می‌شود
_CHATMSG_MODE = GrChatMessage is not None

from api.thinking_view import compose_thinking, thinking_title


def _thinking_bubble(text: str, title: str):
    """پیام «در حال فکر کردن» — حباب جمع‌شونده یا fallback متنی"""
    if _CHATMSG_MODE:
        return GrChatMessage(role="assistant", content=text, metadata={"title": title})
    return f"**{title}**\n\n{text}"


def _final_messages(thinking_text: str, title: str, answer: str):
    """پاسخ نهایی: حباب فکر (بسته) + پیام پاسخ"""
    if _CHATMSG_MODE:
        return [
            GrChatMessage(role="assistant", content=thinking_text,
                          metadata={"title": title}),
            GrChatMessage(role="assistant", content=answer),
        ]
    return (
        f"<details><summary>{title}</summary>\n\n{thinking_text}\n\n</details>\n\n"
        f"{answer}"
    )


async def chatbot(message, history):
    """
    هندلر چت — async generator: در حین اجرا «فرایند فکر» را زنده استریم
    می‌کند (مراحل فلو + فکر مدل‌ها) و در پایان حباب فکر + پاسخ را می‌دهد.
    """
    import asyncio

    context_text = ""
    user_command = ""
    file_paths = []

    # ۱. تشخیص ساختار ورودی
    if isinstance(message, dict):
        # ساختار: {'text': '...', 'files': ['/path/to/file1', ...]}
        user_command = message.get('text', '')
        file_paths = message.get('files', []) or []
    elif isinstance(message, str):
        user_command = message
    else:
        yield "فرمت ورودی ناشناخته است."
        return

    try:
        # بررسی وضعیت حافظه قبل از اجرا
        memory_monitor = get_memory_monitor()
        if not memory_monitor.check_memory()["ok"]:
            memory_monitor.force_cleanup()
            get_model_manager().auto_cleanup(threshold_mb=3072)

        # ۲. مسیر فایل: پردازش با skill ها (YOLO/Whisper/docling) + پاسخ مستقیم
        #    — با فلوی دستی تا در نوار زنده، tab «Flow» و سشن‌ها هم ثبت شود
        if file_paths:
            import time as _t

            question = user_command.strip() or "محتوای این فایل(ها) را توصیف کن."
            state = GLOBAL_ADAPTIVE_ORCH.begin_manual_flow(question)

            for file_path in file_paths:
                name = Path(file_path).name
                logger.info("📎 Processing uploaded file: %s", name)
                yield _thinking_bubble(
                    compose_thinking(state, running=True)
                    + f"\n\n⏳ 📎 در حال پردازش **{name}** …",
                    "🧠 در حال فکر کردن…",
                )
                t0 = _t.perf_counter()
                report = await analyze_uploaded_file(file_path)
                context_text += report
                GLOBAL_ADAPTIVE_ORCH.note_file_processed(
                    state, name, report, _t.perf_counter() - t0
                )

            # درخواست‌های «تبدیل به متن / استخراج متن» نیاز به LLM ندارند —
            # همان متن استخراج‌شده پاسخ است (سریع و بدون تحریف)
            if _EXTRACT_TEXT_RE.search(user_command or ""):
                answer = context_text.strip()
            else:
                yield _thinking_bubble(
                    compose_thinking(state, running=True)
                    + "\n\n⏳ 💬 تولید پاسخ روی محتوای استخراج‌شده…",
                    "🧠 در حال فکر کردن…",
                )
                answer = await GLOBAL_ADAPTIVE_ORCH.direct_answer(
                    question, context=context_text, state=state
                )

            await GLOBAL_ADAPTIVE_ORCH.finish_manual_flow(state, answer)
            yield _final_messages(
                compose_thinking(state, running=False),
                thinking_title(state, running=False),
                answer,
            )
            return

        # ۳. مسیر متن: orchestrator در پس‌زمینه + استریم زنده‌ی فکر
        task = asyncio.create_task(
            GLOBAL_ADAPTIVE_ORCH.run(user_command, "adaptive-session-001")
        )
        while not task.done():
            st = GLOBAL_ADAPTIVE_ORCH.active_state
            if st is not None and st.query == user_command:
                yield _thinking_bubble(
                    compose_thinking(st, running=True),
                    thinking_title(st, running=True),
                )
            await asyncio.wait({task}, timeout=0.8)

        state = task.result()
        yield _final_messages(
            compose_thinking(state, running=False),
            thinking_title(state, running=False),
            _format_answer(state),
        )

    except Exception as e:
        logger.error("chatbot failed: %s", e, exc_info=True)
        get_memory_monitor().force_cleanup()
        yield f"خطا در پردازش: {str(e)}"

# راه‌اندازی Model Manager با تنظیمات مدل‌ها
def initialize_models():
    """ثبت تنظیمات مدل‌ها در Model Manager"""
    model_manager = get_model_manager()
    
    # ثبت مدل Embedding
    model_manager.register_model(
        "embedding",
        ModelConfig(
            model_path=resolve_model_path("Qwen3-Embedding-0.6B", "/Users/dbk/Desktop/agentic-graph-RAG/models/Qwen3-Embedding-0.6B"),
            model_type="embedding",
            device="mps",  # برای Apple Silicon
            use_fp16=True,
            auto_unload=False,  # این مدل همیشه نیاز است
            cache_size=500,
        )
    )
    
    # ثبت مدل LLM اصلی
    model_manager.register_model(
        "llm_main",
        ModelConfig(
            model_path=resolve_model_path("phi3_mini", "/Users/dbk/Desktop/RAG/models/phi3_mini"),
            model_type="reasoning",
            device="mps",
            use_fp16=True,
            auto_unload=True,  # این مدل را می‌توان آزاد کرد
        )
    )
    
    # ثبت مدل Granite برای پاسخ‌دهی
    model_manager.register_model(
        "granite_answer",
        ModelConfig(
            model_path=resolve_model_path("granite4_3b", "/Users/dbk/Desktop/RAG/models/granite4_3b"),
            model_type="llm",
            device="mps",
            use_fp16=True,
            auto_unload=True,
        )
    )

# راه‌اندازی مدل‌ها
initialize_models()

# اعمال حالت اجرای مدل‌ها (serial/concurrent) روی ModelGate
configure_model_gate(CONFIG.get("model_execution"))

# ایجاد AdaptiveOrchestrator با تنظیمات
GLOBAL_ADAPTIVE_ORCH = AdaptiveOrchestrator(CONFIG)

def _format_answer(result) -> str:
    """استخراج پاسخ قابل‌نمایش از خروجی orchestrator (state یا متن)"""
    if isinstance(result, str):
        return result
    if getattr(result, 'final_answer', None):
        return result.final_answer
    if getattr(result, 'tool_results', None):
        parts = []
        for tr in result.tool_results:
            if tr.get("success") and isinstance(tr.get("output"), dict):
                out = tr["output"]
                if "result" in out:
                    expr = out.get("expression", "")
                    parts.append(f"{expr} = {out['result']}" if expr else str(out["result"]))
                elif "summary" in out and out["summary"]:
                    parts.append(out["summary"])
        if parts:
            return "\n".join(parts)

    # هرگز state خام را به کاربر نشان نده
    errs = "\n".join(f"- {e}" for e in getattr(result, "errors", [])) or "- (بدون جزئیات)"
    step = getattr(result, "current_step", None)
    step_name = step.value if step is not None else "?"
    return f"پاسخی تولید نشد (مرحله: {step_name}).\nخطاها:\n{errs}"


async def startAdaptiveOrchestrator(message, iface=None):
    """اجرای AdaptiveOrchestrator و برگرداندن پاسخ متنی (بدون استریم فکر)"""
    try:
        result = await GLOBAL_ADAPTIVE_ORCH.run(message, "adaptive-session-001")
        return _format_answer(result)
    except Exception as e:
        import traceback
        return f"خطا در AdaptiveOrchestrator: {str(e)}\n\nجزئیات:\n{traceback.format_exc()}"

# ساخت رابط کاربری — چت + گراف فلو + dashboard مدیریتی در یک Blocks
try:
    from api.dashboard import build_dashboard_tabs
except ImportError:
    build_dashboard_tabs = None

from api.flow_visualizer import build_flow_tab, build_chat_flow_strip

with gr.Blocks(title="Agentic Graph RAG", fill_height=True, fill_width=True) as iface:
    with gr.Tabs():
        with gr.Tab("💬 Chat"):
            # نوار زنده‌ی مسیر اجرای پرامپت — بالای چت
            build_chat_flow_strip(GLOBAL_ADAPTIVE_ORCH)
            _chat_kwargs = dict(
                fn=chatbot,
                multimodal=True,  # فایل‌ها به صورت دیکشنری با کلید 'files' ارسال می‌شوند
                title="دستیار هوشمند تحلیل فایل (Adaptive Mode)",
                description="فایل (PDF, TXT, CSV, تصویر, صوت) آپلود کنید و دستور خود را بنویسید. فرایند فکر مدل زنده نمایش داده می‌شود.",
                save_history=True,
                fill_height=True,
            )
            if _CHATMSG_MODE:
                try:
                    # type="messages" برای نمایش حباب «🧠 فرایند فکر»
                    gr.ChatInterface(type="messages", **_chat_kwargs)
                except TypeError:
                    # gradio قدیمی — fallback به حالت متنی (فکر داخل <details>)
                    _CHATMSG_MODE = False
                    gr.ChatInterface(**_chat_kwargs)
            else:
                gr.ChatInterface(**_chat_kwargs)
        build_flow_tab(GLOBAL_ADAPTIVE_ORCH)
        if build_dashboard_tabs is not None:
            build_dashboard_tabs(GLOBAL_ADAPTIVE_ORCH)

if __name__ == "__main__":
    from core.model_gate import get_model_gate

    print("\n" + "="*80)
    print("🚀 Starting Adaptive Agentic Graph RAG System")
    print("="*80)
    print("\nFeatures:")
    print("  ✅ Query Classification (7 types)")
    print("  ✅ Dynamic Planning")
    print("  ✅ 42 Skills Available")
    print("  ✅ Semantic Graph Integration")
    print("  ✅ Adaptive Flow Control")
    print(f"  ⚙️ Model Execution Mode: {get_model_gate().mode}"
          "  (MODEL_EXECUTION_MODE=serial|concurrent)")
    print("\n" + "="*80 + "\n")

    iface.launch()

