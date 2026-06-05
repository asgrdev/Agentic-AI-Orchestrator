import gradio as gr
from pathlib import Path
import sys
from agents.orchestrator import Orchestrator
from configs.main_config import CONFIG


_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import fitz  # PyMuPDF برای PDF
from llm.mlx_granite import MlxGraniteAnswerGenerator, MlxGraniteConfig

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

async def chatbot(message, iface):
    """
    پردازش ورودی که یک دیکشنری شامل 'text' و 'files' است.
    """
    
    context_text = ""
    user_command = ""
    
    # ۱. تشخیص ساختار ورودی
    if isinstance(message, dict):
        # ساختار: {'text': '...', 'files': ['/path/to/file1', '/path/to/file2']}
        user_command = message.get('text', '')
        file_paths = message.get('files', [])
        
        # ۲. پردازش لیست فایل‌ها
        if file_paths:
            for file_path in file_paths:
                ext = Path(file_path).suffix.lower()
                
                if ext == '.pdf':
                    context_text += read_pdf_file(file_path)
                elif ext in ['.txt', '.csv', '.json', '.md', '.py', '.js']:
                    context_text += read_text_file(file_path)
                elif ext in ['.png', '.jpg', '.jpeg', '.bmp', '.webp']:
                    # برای تصاویر، مدل Granite نمی‌تواند مستقیماً ببیند.
                    # ما فقط نام فایل را گزارش می‌دهیم (یا می‌توانید OCR اضافه کنید).
                    context_text += f"\n[تصویر آپلود شد: {Path(file_path).name} (توجه: مدل متنی، متن داخل تصویر را نمی‌بیند مایل به OCR هستید؟)]\n"
                else:
                    context_text += f"\n[نوع فایل پشتیبانی شده نیست: {ext}]\n"
        else:
            user_command = message.get('text', '')
            
    elif isinstance(message, str):
        # حالت عادی (بدون فایل)
        user_command = message
    else:
        return "فرمت ورودی ناشناخته است."

    # ۳. ساخت پرامپت نهایی
    if context_text:
        # اول محتوا، بعد دستور
        full_prompt = f"اطلاعات زیر از فایل‌های آپلود شده استخراج شده است:\n{context_text}\n\nالان دستور زیر را اجرا کن:\n{user_command}"
    else:
        full_prompt = user_command

    # ۴. اجرای مدل
    try:
        # استراتژی مدیریت حافظه:
        # اگر می‌خواهید سرعت بالا باشد و حافظه کافی دارید، مدل را خارج از تابع لود کنید.
        # اگر حافظه کم است و نگران OOM هستید، مدل را اینجا لود و پس از پایان (به کمک gc) آزاد کنید.
        # در اینجا برای اطمینان از رفع خطاهای حافظه، مدل را در هر بار اجرا لود می‌کنیم (یا کش می‌کنیم).
        
        # نسخه کش شده (بهترین تعادل):
        # if not hasattr(chatbot, 'extractor'):
        #     print("Loading model (first time)...")
        #     chatbot.extractor = MlxGraniteAnswerGenerator(
        #         MlxGraniteConfig(model_path="/Users/dbk/Desktop/RAG/models/granite4-7b")
        #     )
        
        # # اگر کش نداریم یا خراب شد (برای امنیت بیشتر)
        # if not chatbot.extractor:
        #      chatbot.extractor = MlxGraniteAnswerGenerator(
        #         MlxGraniteConfig(model_path="/Users/dbk/Desktop/RAG/models/granite4-7b")
        #     )
            
        output = await startOrchestrator(user_command, iface) #chatbot.extractor.extract_answer(prompt=full_prompt)
        return output
        
    except Exception as e:
        # در صورت خطا، حافظه را پاک کنیم تا بعدی اجراها مشکلی نداشته باشند
        import gc
        gc.collect()
        return f"خطا در پردازش: {str(e)}"

GLOBAL_ORCH = Orchestrator(CONFIG)

async def startOrchestrator(message, iface):
    return await GLOBAL_ORCH.run(message,"111-00001")

# async def startOrchestrator(message, iface):
#     orchestrator = Orchestrator(CONFIG)
#     return await orchestrator.run(message,"111-00001")
# ساخت رابط کاربری
# نیازی به تعریف دستی inputs نیست چون Gradio خودش ورودی متن و فایل را مدیریت می‌کند
iface = gr.ChatInterface(
    fn=chatbot,
    multimodal=True, # این گزینه باعث می‌شود فایل‌ها به صورت دیکشنری با کلید 'files' به تابع ارسال شوند
    title="دستیار هوشمند تحلیل فایل",
    description="فایل (PDF, TXT, CSV) آپلود کنید و دستور خود را بنویسید.",
    # نکته: در حالت multimodal=True، ورودی به صورت خودکار ترکیب می‌شود و در message دیکشنری شده می‌آید.
    save_history=True,
    fill_height=True,
    fill_width=True,

)

if __name__ == "__main__":
    iface.launch()
