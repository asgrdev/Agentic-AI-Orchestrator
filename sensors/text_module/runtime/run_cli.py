from __future__ import annotations
import argparse, logging, time
from ..config import RuntimeConfig
from ..sources.push_source import PushTextSource
from ..sources.file_source import FileTextSource
from ..sinks.jsonl_sink import JsonlSink
from ..runtime.pipeline import TextSegPipeline, ModelConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("text_module.cli")

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="TextSeg: command parsing + BERT embeddings (local).")
    ap.add_argument("--input", required=True, help="prompt:'text' OR file:/path.txt")
    ap.add_argument("--bert", default="bert-base-cased", help="Local path or HF id for bert-base-cased")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out_jsonl", default="outputs/text_segments.jsonl")
    ap.add_argument("--emb_dir", default="outputs/text_embeddings")
    ap.add_argument("--asr_conf", type=float, default=None)
    ap.add_argument("--source_type", default="user_prompt", choices=["user_prompt","asr","file","system"])
    return ap.parse_args()

def main() -> None:
    args = parse_args()
    runtime = RuntimeConfig()
    pipeline = TextSegPipeline(runtime=runtime, models=ModelConfig(bert_name_or_path=args.bert), device=args.device, embedding_store_dir=args.emb_dir)
    sink = JsonlSink(args.out_jsonl)

    if args.input.startswith("file:"):
        src = FileTextSource(file_path=args.input.split(":",1)[1])
        while True:
            pkt = src.read()
            if pkt is None:
                break
            seg = pipeline.process_packet(pkt)
            sink.write(seg)
            logger.info("Wrote TextSeg %s accepted=%s", seg.seg_id, seg.accepted)
        src.close()
        return

    txt = args.input.split(":",1)[1] if args.input.startswith("prompt:") else args.input
    push = PushTextSource(source_id="cli", source_type=args.source_type)
    push.push(txt, confidence=args.asr_conf)
    pkt = None
    while pkt is None:
        pkt = push.read()
    seg = pipeline.process_packet(pkt)
    sink.write(seg)
    logger.info("Wrote TextSeg %s accepted=%s intents=%d", seg.seg_id, seg.accepted, len(seg.intents))
    push.close()

if __name__ == "__main__":
    main()
