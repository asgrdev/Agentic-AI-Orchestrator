from __future__ import annotations
import argparse, logging, os, time
from ..config import RuntimeConfig
from ..sources.file_source import FileVideoSource
from ..sources.camera_source import CameraVideoSource
from ..sources.rtsp_source import RtspVideoSource
from ..sinks.jsonl_sink import JsonlSink
from ..runtime.pipeline import VisionSegPipeline, ModelPaths

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vision_module.cli")

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="file:/path.mp4 OR cam:0 OR rtsp:URL")
    ap.add_argument("--out_jsonl", default="outputs/vision_segments.jsonl")
    ap.add_argument("--emb_dir", default="outputs/vision_embeddings")
    ap.add_argument("--snap_dir", default="outputs/vision_snapshots")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--window_sec", type=float, default=15.0)
    ap.add_argument("--fps_infer", type=float, default=3.0)
    ap.add_argument("--snapshot_every_sec", type=float, default=5.0)
    ap.add_argument("--yolo_det", required=True)
    ap.add_argument("--yolo_seg", default=None)
    ap.add_argument("--yolo_pose", default=None)
    ap.add_argument("--disable_seg", action="store_true")
    ap.add_argument("--disable_pose", action="store_true")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    return ap.parse_args()

def main() -> None:
    args = parse_args()
    runtime = RuntimeConfig(window_duration_sec=args.window_sec, fps_infer=args.fps_infer,
                           snapshot_every_sec=args.snapshot_every_sec,
                           detection_conf_threshold=args.conf, detection_iou_threshold=args.iou,
                           enable_segmentation=(not args.disable_seg),
                           enable_pose=(not args.disable_pose))
    models = ModelPaths(detector_path=args.yolo_det, segmenter_path=args.yolo_seg, pose_path=args.yolo_pose)
    pipeline = VisionSegPipeline(runtime=runtime, model_paths=models, device=args.device,
                                 embedding_store_dir=args.emb_dir, snapshot_store_dir=args.snap_dir)
    sink = JsonlSink(args.out_jsonl)

    if args.input.startswith("file:"):
        path = args.input.split(":",1)[1]
        source = FileVideoSource(file_path=path, source_id=f"file:{os.path.basename(path)}")
    elif args.input.startswith("cam:"):
        idx = int(args.input.split(":",1)[1])
        source = CameraVideoSource(camera_index=idx, source_id=f"cam{idx}")
    elif args.input.startswith("rtsp:"):
        url = args.input.split(":",1)[1]
        source = RtspVideoSource(rtsp_url=url, source_id="rtsp0")
    else:
        raise ValueError("Unsupported input")

    try:
        while True:
            frame = source.read()
            if frame is None:
                break
            seg = pipeline.process_frame(frame)
            if seg is not None:
                sink.write(seg)
                logger.info("Emitted VisionSeg %s frames=%d", seg.seg_id, seg.frames_used)
    finally:
        try:
            seg = pipeline.flush(source_id=getattr(source,"source_id","unknown"), t_ms=int(time.time()*1000))
            if seg is not None:
                sink.write(seg)
        except Exception as e:
          print(f">>{e}")
          pass
        source.close()

if __name__ == "__main__":
    main()
