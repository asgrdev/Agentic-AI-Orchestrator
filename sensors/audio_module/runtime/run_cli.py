from __future__ import annotations
import argparse
import logging
import os

from ..config import RuntimeConfig
from ..sources.file_source import FileAudioSource
from ..sources.mic_source import MicrophoneAudioSource
from ..sinks.jsonl_sink import JsonlSink
from ..runtime.pipeline import AudioSegPipeline
from ..processors.efficientat import PANNsConfig
from ..processors.whisper_asr import WhisperConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("audio_module.cli")

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="AudioSeg pipeline (EfficientAT + Whisper) local/offline.")
    ap.add_argument("--input", required=True, help="file:/path.wav OR mic")
    ap.add_argument("--out_jsonl", default="outputs/audio_segments.jsonl")
    ap.add_argument("--emb_dir", default="outputs/embeddings")

    ap.add_argument("--sample_rate", type=int, default=16000)
    ap.add_argument("--segment_sec", type=float, default=1.0)
    ap.add_argument("--hop_sec", type=float, default=1.0)

    ap.add_argument("--efficientat_model", required=True, help="Local EfficientAT model path (.pt/.ts)")
    ap.add_argument("--efficientat_labels", required=True, help="Label file, one per line")
    ap.add_argument("--efficientat_device", default="cpu")
    ap.add_argument("--efficientat_topk", type=int, default=8)

    ap.add_argument("--whisper_model", default="medium")
    ap.add_argument("--whisper_device", default="cpu")
    ap.add_argument("--whisper_language", default=None)

    ap.add_argument("--always_asr", action="store_true", help="Force ASR when possible (CPU heavy).")
    return ap.parse_args()

def main() -> None:
    args = parse_args()

    runtime = RuntimeConfig(
        sample_rate_hz=args.sample_rate,
        segment_duration_sec=args.segment_sec,
        segment_hop_sec=args.hop_sec,
    )

    efficientat_cfg = PANNsConfig(
        model_name="EfficientAT-medium",
        model_path=args.efficientat_model,
        labels_path=args.efficientat_labels,
        device=args.efficientat_device,
        topk=args.efficientat_topk,
    )
    whisper_cfg = WhisperConfig(
        model_name=args.whisper_model,
        device=args.whisper_device,
        language=args.whisper_language,
    )

    pipeline = AudioSegPipeline(
        runtime=runtime,
        efficientat_cfg=efficientat_cfg,
        whisper_cfg=whisper_cfg,
        embedding_store_dir=args.emb_dir,
    )
    sink = JsonlSink(args.out_jsonl)

    if args.input.startswith("file:"):
        path = args.input.split(":", 1)[1]
        source = FileAudioSource(file_path=path, source_id=f"file:{os.path.basename(path)}", target_sample_rate_hz=args.sample_rate)
    elif args.input == "mic":
        source = MicrophoneAudioSource(source_id="mic0", sample_rate_hz=args.sample_rate)
    else:
        raise ValueError("Unsupported input. Use file:/path.wav or mic")

    logger.info("Writing JSONL: %s", args.out_jsonl)
    try:
        while True:
            chunk = source.read()
            if chunk is None:
                break
            segments = pipeline.process_chunk(chunk, user_requested_asr=args.always_asr)
            for seg in segments:
                sink.write(seg)
    finally:
        source.close()

    logger.info("Done.")

if __name__ == "__main__":
    main()
