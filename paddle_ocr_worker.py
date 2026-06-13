import argparse
import importlib
import json
import os
import sys

import cv2
# Important on Windows: importing torch before paddleocr avoids a DLL load failure
# seen when albumentations' pytorch module imports torch during paddleocr import.
import torch  # noqa: F401


def _looks_like_quad(points):
    if not isinstance(points, (list, tuple)) or len(points) != 4:
        return False
    for p in points:
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            return False
        try:
            float(p[0])
            float(p[1])
        except (TypeError, ValueError):
            return False
    return True


def _rec_text_to_str(rec) -> str:
    if rec is None:
        return ""
    if isinstance(rec, (list, tuple)) and len(rec) > 0:
        return str(rec[0] or "").strip()
    return str(rec).strip()


def _poly_to_quad_list(poly):
    """PaddleX ``rec_polys`` entry → four ``[x, y]`` corners (JSON-serializable)."""
    if poly is None:
        return None
    if hasattr(poly, "tolist"):
        try:
            poly = poly.tolist()
        except Exception:
            return None
    if not isinstance(poly, (list, tuple)) or len(poly) < 4:
        return None
    corners = list(poly[:4]) if len(poly) > 4 else list(poly)
    pts = []
    for p in corners:
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            return None
        try:
            pts.append([float(p[0]), float(p[1])])
        except (TypeError, ValueError):
            return None
    return pts


def _pages_from_paddle_output(output):
    """Normalize ``ocr()`` / ``predict()`` value to a list of per-image dicts."""
    if output is None:
        return []
    if isinstance(output, dict) and "rec_texts" in output:
        return [output]
    if isinstance(output, list):
        return output
    return []


def _normalize_paddlex_dict_pages(pages):
    """
    PaddleOCR 3.x uses PaddleX ``OCRResult`` dicts: ``rec_polys``, ``rec_texts``, ``rec_scores``.
    The legacy nested walker never enters these dicts, so lines (especially Chinese) were dropped.
    """
    out = []
    for page in pages:
        if not isinstance(page, dict):
            continue
        texts = page.get("rec_texts")
        polys = page.get("rec_polys")
        scores = page.get("rec_scores")
        if texts is None or polys is None:
            continue
        if hasattr(texts, "tolist"):
            try:
                texts = texts.tolist()
            except Exception:
                texts = list(texts)
        if hasattr(polys, "tolist"):
            try:
                polys = polys.tolist()
            except Exception:
                polys = list(polys)
        if hasattr(scores, "tolist"):
            try:
                scores = scores.tolist()
            except Exception:
                scores = list(scores) if scores is not None else []
        if not isinstance(texts, (list, tuple)) or not isinstance(polys, (list, tuple)):
            continue
        scores = scores if isinstance(scores, (list, tuple)) else []
        n = min(len(texts), len(polys))
        for i in range(n):
            txt = _rec_text_to_str(texts[i])
            if not txt:
                continue
            quad = _poly_to_quad_list(polys[i])
            if quad is None or not _looks_like_quad(quad):
                continue
            try:
                conf = float(scores[i]) if i < len(scores) else 0.0
            except (TypeError, ValueError):
                conf = 0.0
            out.append([quad, txt, conf])
    return out


def _normalize_legacy_walk(output):
    out = []

    def walk(node):
        if not isinstance(node, (list, tuple)):
            return
        if len(node) == 2 and _looks_like_quad(node[0]):
            rec = node[1]
            text = ""
            conf = 0.0
            if isinstance(rec, (list, tuple)) and len(rec) >= 2:
                text = str(rec[0] or "")
                try:
                    conf = float(rec[1] or 0.0)
                except (TypeError, ValueError):
                    conf = 0.0
            elif isinstance(rec, str):
                text = rec
            if text.strip():
                out.append([node[0], text, conf])
            return
        for child in node:
            walk(child)

    walk(output)
    return out


def _normalize(output):
    pages = _pages_from_paddle_output(output)
    if pages and isinstance(pages[0], dict) and "rec_texts" in pages[0]:
        px = _normalize_paddlex_dict_pages(pages)
        if px:
            return px
    return _normalize_legacy_walk(output)


def _load_paddle_ocr_class():
    """Load PaddleOCR lazily so environments without paddleocr can fail gracefully at runtime."""
    module = importlib.import_module("paddleocr")
    return getattr(module, "PaddleOCR")


def _paddle_extra_init_kwargs() -> dict:
    """
    Optional tuning via environment (PaddleOCR / PaddleX pipeline).

    - OCR_PADDLE_TEXT_DET_THRESH — lower (~0.2–0.3) → more text proposals (more recall, more noise).
    - OCR_PADDLE_TEXT_DET_BOX_THRESH — lower (~0.3–0.45) → keep weaker boxes.
    - OCR_PADDLE_TEXT_DET_UNCLIP_RATIO — higher (~1.8–2.2) → larger expanded boxes.
    - OCR_PADDLE_TEXT_REC_SCORE_THRESH — lower (~0.3–0.5) → keep weaker recognition scores.
    - OCR_PADDLE_DET_LIMIT_SIDE_LEN — larger (e.g. 2048) → upscale limit before detection.

    Bilingual Arabic/English signage often improves when Arabic (`lang=ar`) runs before English in the app merge order,
    and when detector thresholds are tuned (above); PP-OCR line granularity still differs from EasyOCR CRAFT.
    """
    kw = {}
    mapping = (
        ("text_det_thresh", "OCR_PADDLE_TEXT_DET_THRESH", float),
        ("text_det_box_thresh", "OCR_PADDLE_TEXT_DET_BOX_THRESH", float),
        ("text_det_unclip_ratio", "OCR_PADDLE_TEXT_DET_UNCLIP_RATIO", float),
        ("text_rec_score_thresh", "OCR_PADDLE_TEXT_REC_SCORE_THRESH", float),
        ("text_det_limit_side_len", "OCR_PADDLE_DET_LIMIT_SIDE_LEN", int),
    )
    for param, env_name, caster in mapping:
        raw = (os.environ.get(env_name) or "").strip()
        if not raw:
            continue
        try:
            kw[param] = caster(raw)
        except (TypeError, ValueError):
            continue
    return kw


def _valid_image_bgr(img) -> bool:
    return (
        img is not None
        and getattr(img, "size", 0) > 0
        and len(getattr(img, "shape", ())) >= 2
        and int(img.shape[0]) > 0
        and int(img.shape[1]) > 0
    )


def run_oneshot(image_path: str, lang: str, use_gpu: int) -> None:
    """Single image: load Paddle once, OCR once, print JSON, exit."""
    try:
        PaddleOCR = _load_paddle_ocr_class()
    except Exception as e:
        sys.stderr.write(f"Import error: {e}\n")
        print("[]")
        sys.exit(2)

    img = cv2.imread(image_path)
    if not _valid_image_bgr(img):
        print("[]")
        return

    try:
        extra = _paddle_extra_init_kwargs()
        ocr = PaddleOCR(
            use_angle_cls=True,
            lang=lang,
            use_gpu=bool(use_gpu),
            show_log=False,
            **extra,
        )
        raw = ocr.ocr(img, cls=True)
        print(json.dumps(_normalize(raw), ensure_ascii=False))
    except Exception as e:
        sys.stderr.write(f"OCR error: {e}\n")
        print("[]")
        sys.exit(1)


def run_daemon(lang: str, use_gpu: int) -> None:
    """Keep PaddleOCR loaded; read JSON lines {\"path\": \"...\"} from stdin; reply with one JSON array per line."""
    try:
        PaddleOCR = _load_paddle_ocr_class()
    except Exception as e:
        sys.stderr.write(f"Import error: {e}\n")
        print("INIT_FAIL", flush=True)
        sys.exit(2)

    try:
        extra = _paddle_extra_init_kwargs()
        ocr = PaddleOCR(
            use_angle_cls=True,
            lang=lang,
            use_gpu=bool(use_gpu),
            show_log=False,
            **extra,
        )
    except Exception as e:
        sys.stderr.write(f"PaddleOCR init error: {e}\n")
        print("INIT_FAIL", flush=True)
        sys.exit(2)

    print("READY", flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line == "__EXIT__":
            break
        try:
            obj = json.loads(line)
            path = obj.get("path") if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            print(json.dumps([]), flush=True)
            continue
        if not path or not isinstance(path, str):
            print(json.dumps([]), flush=True)
            continue
        img = cv2.imread(path)
        if not _valid_image_bgr(img):
            print(json.dumps([]), flush=True)
            continue
        try:
            raw = ocr.ocr(img, cls=True)
            print(json.dumps(_normalize(raw), ensure_ascii=False), flush=True)
        except Exception as e:
            sys.stderr.write(f"OCR error: {e}\n")
            print("[]", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--daemon", action="store_true", help="Stay alive; stdin protocol for repeated OCR.")
    ap.add_argument("--image", default=None)
    ap.add_argument("--lang", default="en")
    ap.add_argument(
        "--use-gpu",
        type=int,
        default=1,
        choices=(0, 1),
        help="1 = use GPU (requires paddlepaddle-gpu + CUDA), 0 = CPU",
    )
    args = ap.parse_args()

    if args.daemon:
        run_daemon(args.lang, args.use_gpu)
        return

    if not args.image:
        ap.error("--image is required unless --daemon")
    run_oneshot(args.image, args.lang, args.use_gpu)


if __name__ == "__main__":
    main()
