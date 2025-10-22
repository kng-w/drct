import argparse
import os
import sys
import time
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from drct.archs.DRCT_arch import DRCT
from tqdm import tqdm


# ------------------
# Model construction
# ------------------

def _build_model_cfg(arch: str) -> dict:
    arch = arch.upper()
    if arch == "XL":
        return dict(
            upscale=4,
            in_chans=3,
            img_size=64,
            window_size=16,
            compress_ratio=3,
            squeeze_factor=30,
            conv_scale=0.01,
            overlap_ratio=0.5,
            img_range=1.0,
            depths=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
            gc=32,
            mlp_ratio=2,
            upsampler="pixelshuffle",
            resi_connection="1conv",
        )
    elif arch == "L":
        return dict(
            upscale=4,
            in_chans=3,
            img_size=64,
            window_size=16,
            compress_ratio=3,
            squeeze_factor=30,
            conv_scale=0.01,
            overlap_ratio=0.5,
            img_range=1.0,
            depths=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
            gc=32,
            mlp_ratio=2,
            upsampler="pixelshuffle",
            resi_connection="1conv",
        )
    else:
        raise ValueError(f"Unknown arch: {arch}")


def build_model_for_weights(model_path: str, prefer_arch: Optional[str] = None) -> Tuple[torch.nn.Module, str]:
    """
    Build a DRCT model that matches provided weights. Tries preferred arch first,
    then falls back to the other known arch if strict loading fails.
    Returns: (model, resolved_arch)
    """
    ckpt = torch.load(model_path, map_location="cpu")
    params = ckpt.get("params", ckpt)

    tried: List[str] = []
    candidates: List[str] = []
    if prefer_arch:
        candidates.append(prefer_arch.upper())
    # heuristic from filename
    name = os.path.basename(model_path).lower()
    if "xl" in name and "XL" not in candidates:
        candidates.append("XL")
    if "-l" in name or name.endswith("_l.pth"):
        if "L" not in candidates:
            candidates.append("L")
    # default fallbacks
    for arch in ["XL", "L"]:
        if arch not in candidates:
            candidates.append(arch)

    last_err: Optional[Exception] = None
    for arch in candidates:
        tried.append(arch)
        cfg = _build_model_cfg(arch)
        model = DRCT(**cfg)
        try:
            model.load_state_dict(params, strict=True)
            return model, arch
        except Exception as e:  # size mismatch
            last_err = e
            continue

    raise RuntimeError(
        f"Unable to load weights strictly with any arch {tried}. Last error: {last_err}"
    )


# ------------------
# Inference helpers
# ------------------

def _to_tensor_bgr_uint8(img_path: str) -> Tuple[torch.Tensor, Tuple[int, int]]:
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")
    h, w = img.shape[:2]
    img = img.astype(np.float32) / 255.0
    # BGR to RGB and HWC->CHW
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    return img.unsqueeze(0), (h, w)


def _tile_test(img_lq: torch.Tensor, model: torch.nn.Module, scale: int, tile: int, tile_overlap: int, window_size: int) -> torch.Tensor:
    b, c, h, w = img_lq.size()
    t = min(tile, h, w)
    assert t % window_size == 0, "tile size should be a multiple of window_size"
    stride = t - tile_overlap
    h_idx_list = list(range(0, h - t, stride)) + [h - t]
    w_idx_list = list(range(0, w - t, stride)) + [w - t]
    E = torch.zeros(b, c, h * scale, w * scale, dtype=img_lq.dtype, device=img_lq.device)
    W = torch.zeros_like(E)

    for hi in h_idx_list:
        for wi in w_idx_list:
            in_patch = img_lq[..., hi : hi + t, wi : wi + t]
            out_patch = model(in_patch)
            out_mask = torch.ones_like(out_patch)
            E[..., hi * scale : (hi + t) * scale, wi * scale : (wi + t) * scale].add_(out_patch)
            W[..., hi * scale : (hi + t) * scale, wi * scale : (wi + t) * scale].add_(out_mask)
    output = E.div_(W)
    return output


def _inference_one(
    src_path: str,
    dst_path: str,
    model: torch.nn.Module,
    device: torch.device,
    scale: int = 4,
    tile: Optional[int] = None,
    tile_overlap: int = 32,
    window_size: int = 16,
    use_half: bool = False,
) -> None:
    img, (h_old, w_old) = _to_tensor_bgr_uint8(src_path)
    if use_half and device.type == "cuda":
        img = img.half()
    img = img.to(device)

    with torch.no_grad():
        # pad to multiples of window size
        _, _, h, w = img.size()
        h_pad = (h // window_size + 1) * window_size - h
        w_pad = (w // window_size + 1) * window_size - w
        img_pad = torch.cat([img, torch.flip(img, [2])], 2)[:, :, : h + h_pad, :]
        img_pad = torch.cat([img_pad, torch.flip(img_pad, [3])], 3)[:, :, :, : w + w_pad]

        if tile is None:
            out = model(img_pad)
        else:
            out = _tile_test(img_pad, model, scale=scale, tile=tile, tile_overlap=tile_overlap, window_size=window_size)

        out = out[..., : h_old * scale, : w_old * scale]

    out = out.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    out = np.transpose(out[[2, 1, 0], :, :], (1, 2, 0))  # RGB->BGR
    out = (out * 255.0).round().astype(np.uint8)

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    cv2.imwrite(dst_path, out)


# ------------------
# Progress helpers
# ------------------

def _format_eta(elapsed: float, done: int, total: int) -> str:
    if done == 0:
        return "--:--"
    rate = elapsed / done
    remain = rate * (total - done)
    return time.strftime("%M:%S", time.gmtime(remain))


def _print_progress(done: int, total: int, prefix: str = "") -> None:
    # Kept for backward compatibility; not used when tqdm is available
    pct = (done / total * 100) if total else 100.0
    bar_len = 20
    filled = int(bar_len * pct / 100)
    bar = "#" * filled + "-" * (bar_len - filled)
    sys.stdout.write(f"\r{prefix}[{bar}] {done}/{total} ({pct:5.1f}%)")
    sys.stdout.flush()


# ------------------
# Main entry
# ------------------

def list_images(root: str, exts: Sequence[str]) -> List[str]:
    root = os.path.abspath(root)
    out: List[str] = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if os.path.splitext(fn)[1].lower() in exts:
                out.append(os.path.join(dp, fn))
    out.sort()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Recursive directory inference for DRCT")
    parser.add_argument("--input_dir", required=True, type=str, help="Root directory containing input images")
    parser.add_argument("--model_path", type=str, default="models/DRCT_X4.pth")
    parser.add_argument("--scale", type=int, default=4, help="Upscale factor (default: 4)")
    parser.add_argument("--tile", type=int, default=None, help="Tile size, None to process whole image")
    parser.add_argument("--tile_overlap", type=int, default=32, help="Overlap between tiles")
    parser.add_argument("--arch", type=str, default=None, choices=["L", "XL", None], help="Force model arch (L/XL). Default: auto")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 on CUDA for faster inference")
    parser.add_argument("--workers", type=int, default=1, help="CPU-only: number of parallel workers")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for GPU inference (tile=None)")
    parser.add_argument("--output_root", type=str, default="output", help="Output root directory")
    parser.add_argument("--suffix_mode", type=str, default="auto", choices=["auto", "x4", "xl"], help="Suffix for output dir name")
    parser.add_argument("--exts", type=str, default=".jpg,.jpeg,.png,.bmp,.webp", help="Comma-separated image extensions to include")
    parser.add_argument("--skip_existing", action="store_true", help="Skip processing files that already exist in output")

    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    if not os.path.isdir(input_dir):
        raise SystemExit(f"Input directory not found: {input_dir}")

    # Determine suffix for output directory name
    base = os.path.basename(os.path.normpath(input_dir))
    suffix: str
    if args.suffix_mode == "x4":
        suffix = "x4"
    elif args.suffix_mode == "xl":
        suffix = "xl"
    else:
        mp = os.path.basename(args.model_path).lower()
        suffix = "xl" if "xl" in mp else f"x{args.scale}"

    out_root = os.path.join(args.output_root, f"{base}-sr_{suffix}")
    os.makedirs(out_root, exist_ok=True)

    exts = tuple(e.strip().lower() for e in args.exts.split(",") if e.strip())
    srcs = list_images(input_dir, exts)
    if not srcs:
        print("No images found to process.")
        return

    # Collect (src, dst)
    pairs: List[Tuple[str, str]] = []
    for sp in srcs:
        rel = os.path.relpath(sp, input_dir)
        dp = os.path.join(out_root, rel)
        if args.skip_existing and os.path.exists(dp):
            continue
        pairs.append((sp, dp))

    # Setup device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prefer_arch = args.arch
    model, used_arch = build_model_for_weights(args.model_path, prefer_arch=prefer_arch)
    model.eval()
    model = model.to(device)
    window_size = 16
    use_half = bool(args.fp16 and device.type == "cuda")
    if use_half:
        model = model.half()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    total = len(pairs)
    start = time.time()
    print(f"Found {total} images. Device={device}, Arch={used_arch}, FP16={use_half}")

    # Parallelism policy: only enable multi-process on CPU to avoid GPU OOM/contention.
    cpu_parallel = device.type == "cpu" and args.workers and args.workers > 1

    processed = 0

    if not cpu_parallel:
        with tqdm(total=total, desc="Processing", unit="img", dynamic_ncols=True, miniters=1) as pbar:
            # If batch_size > 1 and no tile mode, run batched inference for throughput
            if args.batch_size and args.batch_size > 1 and args.tile is None:
                bs = int(args.batch_size)
                # local helper to pad to target h,w with reflection like single-image path
                def _pad_to(img: torch.Tensor, th: int, tw: int) -> torch.Tensor:
                    b, c, h, w = img.size()
                    if h < th:
                        img = torch.cat([img, torch.flip(img, [2])], 2)[:, :, :th, :]
                    if w < tw:
                        img = torch.cat([img, torch.flip(img, [3])], 3)[:, :, :, :tw]
                    return img

                for i in range(0, total, bs):
                    batch_pairs = pairs[i : i + bs]
                    images: List[torch.Tensor] = []
                    metas: List[Tuple[str, str, int, int]] = []
                    # load CPU tensors
                    for sp, dp in batch_pairs:
                        img, (h_old, w_old) = _to_tensor_bgr_uint8(sp)
                        images.append(img)
                        metas.append((sp, dp, h_old, w_old))

                    # first pad each to window multiple
                    init_padded: List[torch.Tensor] = []
                    sizes: List[Tuple[int, int]] = []
                    for img, (_, _, h_old, w_old) in zip(images, metas):
                        h1 = ((h_old + window_size - 1) // window_size) * window_size
                        w1 = ((w_old + window_size - 1) // window_size) * window_size
                        img1 = _pad_to(img, h1, w1)
                        init_padded.append(img1)
                        sizes.append((h1, w1))

                    max_h = max(h for h, _ in sizes)
                    max_w = max(w for _, w in sizes)
                    final: List[torch.Tensor] = [
                        _pad_to(t, max_h, max_w) for t in init_padded
                    ]

                    batch = torch.cat(final, dim=0)
                    if use_half:
                        batch = batch.half()
                    batch = batch.to(device)

                    with torch.no_grad():
                        out = model(batch)

                    # save each
                    for b_idx, (sp, dp, h_old, w_old) in enumerate(metas):
                        out_i = out[b_idx:b_idx + 1, ...]
                        out_i = out_i[..., : h_old * args.scale, : w_old * args.scale]
                        out_i = out_i.data.squeeze().float().cpu().clamp_(0, 1).numpy()
                        out_i = np.transpose(out_i[[2, 1, 0], :, :], (1, 2, 0))
                        out_i = (out_i * 255.0).round().astype(np.uint8)
                        os.makedirs(os.path.dirname(dp), exist_ok=True)
                        cv2.imwrite(dp, out_i)
                    processed += len(batch_pairs)
                    pbar.update(len(batch_pairs))
            else:
                # single-image path
                for sp, dp in pairs:
                    _inference_one(
                        sp,
                        dp,
                        model=model,
                        device=device,
                        scale=args.scale,
                        tile=args.tile,
                        tile_overlap=args.tile_overlap,
                        window_size=window_size,
                        use_half=use_half,
                    )
                    processed += 1
                    pbar.update(1)
    else:
        # CPU-only parallel processing: each worker builds its own model to avoid sharing issues.
        # Lazily import here to keep startup time small when not used.
        import concurrent.futures as _fut

        # Prepare args needed by worker
        worker_args = dict(
            model_path=args.model_path,
            prefer_arch=prefer_arch,
            scale=args.scale,
            tile=args.tile,
            tile_overlap=args.tile_overlap,
            window_size=window_size,
        )

        def _worker(task: Tuple[str, str]) -> str:
            src, dst = task
            # Build model per worker on CPU
            m, _ = build_model_for_weights(worker_args["model_path"], prefer_arch=worker_args["prefer_arch"]) 
            m.eval()
            dev = torch.device("cpu")
            _inference_one(
                src,
                dst,
                model=m,
                device=dev,
                scale=worker_args["scale"],
                tile=worker_args["tile"],
                tile_overlap=worker_args["tile_overlap"],
                window_size=worker_args["window_size"],
                use_half=False,
            )
            return dst

        with _fut.ProcessPoolExecutor(max_workers=args.workers) as ex, tqdm(total=total, desc="Processing", unit="img", dynamic_ncols=True, miniters=1) as pbar:
            for _ in ex.map(_worker, pairs):
                processed += 1
                pbar.update(1)

    elapsed = time.time() - start
    sys.stdout.write("\n")
    print(f"Done. Processed {total} images in {elapsed:.1f}s. Output: {out_root}")


if __name__ == "__main__":
    main()
