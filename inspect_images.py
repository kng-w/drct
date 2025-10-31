#!/usr/bin/env python3
"""
指定した画像ファイルのサイズ（幅・高さ、バイトサイズ、形式）を調べて表示する簡単なCLIツール。

使い方:
  python inspect_images.py path1 [path2 ...]
  python inspect_images.py images/**/*.png --recursive

オプション:
  --recursive        ディレクトリ指定時に再帰的に探索
  --json             JSON で出力
  --details          Pillow が利用可能な場合、モードや DPI 等の追加情報も表示/出力

備考:
  - Pillow(PIL) が導入済みならそれを使って画像情報を取得します。
  - Pillow が無い場合でも、PNG/JPEG/GIF/BMP はヘッダ解析で幅・高さを取得できます。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Dict, Any


def try_import_pillow():
    try:
        from PIL import Image, ExifTags  # type: ignore
        return Image, ExifTags
    except Exception:
        return None, None


def human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(n)
    for u in units:
        if size < 1024.0 or u == units[-1]:
            return f"{size:.1f}{u}"
        size /= 1024.0


def detect_png_size(fp) -> Optional[Tuple[int, int]]:
    sig = fp.read(8)
    if sig != b"\x89PNG\r\n\x1a\n":
        return None
    # IHDR チャンクを読む
    # 4B length(13), 4B type('IHDR'), 13B data
    data = fp.read(8 + 13)
    if len(data) < 21 or data[4:8] != b"IHDR":
        return None
    w = int.from_bytes(data[8:12], "big")
    h = int.from_bytes(data[12:16], "big")
    return w, h


def detect_gif_size(fp) -> Optional[Tuple[int, int]]:
    hdr = fp.read(6)
    if hdr not in (b"GIF87a", b"GIF89a"):
        return None
    dims = fp.read(4)
    if len(dims) < 4:
        return None
    w = int.from_bytes(dims[0:2], "little")
    h = int.from_bytes(dims[2:4], "little")
    return w, h


def detect_bmp_size(fp) -> Optional[Tuple[int, int]]:
    hdr = fp.read(2)
    if hdr != b"BM":
        return None
    # DIB ヘッダ内の幅・高さ（標準的な BITMAPINFOHEADER 前提）
    # ファイル先頭から 18 バイト目に幅(4B little)、22 バイト目に高さ(4B little)
    try:
        fp.seek(18)
        data = fp.read(8)
        if len(data) < 8:
            return None
        w = int.from_bytes(data[0:4], "little", signed=True)
        h = int.from_bytes(data[4:8], "little", signed=True)
        return abs(w), abs(h)
    except Exception:
        return None


def detect_jpeg_size(fp) -> Optional[Tuple[int, int]]:
    # JPEG は SOI(FFD8) の後、SOF セグメントまでスキャン
    soi = fp.read(2)
    if soi != b"\xFF\xD8":
        return None
    def _read_marker() -> Optional[int]:
        # 0xFF の連続の後の 1 バイトがマーカー
        b = fp.read(1)
        while b == b"\xFF":
            b = fp.read(1)
        if not b:
            return None
        return b[0]

    SOF_MARKERS = set([
        0xC0, 0xC1, 0xC2, 0xC3,
        0xC5, 0xC6, 0xC7,
        0xC9, 0xCA, 0xCB,
        0xCD, 0xCE, 0xCF,
    ])

    while True:
        marker = _read_marker()
        if marker is None:
            return None
        if marker in (0xD8, 0xD9):  # SOI/EOI
            continue
        if marker == 0x01 or (0xD0 <= marker <= 0xD7):
            # Tem/RSTn は長さフィールド無し
            continue
        # 長さ（この2バイト自身を含む）
        seg_len_bytes = fp.read(2)
        if len(seg_len_bytes) != 2:
            return None
        seg_len = int.from_bytes(seg_len_bytes, "big")
        if seg_len < 2:
            return None
        if marker in SOF_MARKERS:
            # precision(1) + height(2) + width(2) + ...
            rest = fp.read(5)
            if len(rest) < 5:
                return None
            h = int.from_bytes(rest[1:3], "big")
            w = int.from_bytes(rest[3:5], "big")
            return w, h
        # それ以外はスキップ
        to_skip = seg_len - 2
        fp.seek(to_skip, os.SEEK_CUR)


def sniff_size(path: Path) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    """拡張ライブラリ無しで画像種別とサイズを推定（PNG/JPEG/GIF/BMP）。"""
    try:
        with path.open("rb") as fp:
            # 先頭を見てフォーマットを判定し、それぞれのリーダで幅・高さ取得
            fp.seek(0)
            for fmt, reader in (
                ("PNG", detect_png_size),
                ("GIF", detect_gif_size),
                ("BMP", detect_bmp_size),
                ("JPEG", detect_jpeg_size),
            ):
                fp.seek(0)
                size = reader(fp)
                if size:
                    return fmt, size[0], size[1]
    except Exception:
        pass
    return None, None, None


def gather_paths(inputs: Iterable[str], recursive: bool) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp"}
    result: List[Path] = []
    for raw in inputs:
        p = Path(raw)
        # グロブ表現は Path.glob でなく glob モジュールの方が直感的だが、
        # ここでは簡単のため * を含むかで分岐
        if any(ch in raw for ch in "*?["):
            for m in Path().glob(raw):
                if m.is_file():
                    result.append(m)
                elif m.is_dir() and recursive:
                    for f in m.rglob("*"):
                        if f.is_file() and f.suffix.lower() in exts:
                            result.append(f)
        elif p.is_file():
            result.append(p)
        elif p.is_dir():
            it = p.rglob("*") if recursive else p.glob("*")
            for f in it:
                if f.is_file() and (recursive or f.suffix.lower() in exts):
                    # 非再帰のときは拡張子で軽くフィルタ
                    result.append(f)
    # 重複排除、安定化
    seen = set()
    uniq: List[Path] = []
    for p in result:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    return uniq


def inspect_path(path: Path, use_details: bool, Image=None, ExifTags=None) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "path": str(path),
        "size_bytes": None,
        "size_human": None,
        "format": None,
        "width": None,
        "height": None,
    }
    try:
        sz = path.stat().st_size
        info["size_bytes"] = sz
        info["size_human"] = human_bytes(sz)
    except Exception:
        pass

    pil_ok = Image is not None
    if pil_ok:
        try:
            with Image.open(path) as img:
                w, h = img.size
                info.update({
                    "format": img.format,
                    "width": w,
                    "height": h,
                })
                if use_details:
                    info["mode"] = img.mode
                    dpi = None
                    if "dpi" in img.info:
                        dpi = img.info.get("dpi")
                    else:
                        # EXIF から DPI を試みる
                        try:
                            exif = getattr(img, "_getexif", None)
                            if callable(exif):
                                exif_data = exif() or {}
                                tag_by_name = {v: k for k, v in (ExifTags.TAGS or {}).items()}
                                x_tag = tag_by_name.get("XResolution")
                                y_tag = tag_by_name.get("YResolution")
                                if exif_data and x_tag in exif_data and y_tag in exif_data:
                                    x = exif_data[x_tag]
                                    y = exif_data[y_tag]
                                    # 比率型 (num, den) の可能性
                                    def _to_float(v):
                                        try:
                                            if isinstance(v, tuple) and len(v) == 2:
                                                return float(v[0]) / float(v[1] or 1)
                                            return float(v)
                                        except Exception:
                                            return None
                                    dpi = (_to_float(x), _to_float(y))
                        except Exception:
                            pass
                    if dpi is not None:
                        info["dpi"] = dpi
        except Exception:
            pil_ok = False

    if not pil_ok:
        fmt, w, h = sniff_size(path)
        if fmt:
            info.update({"format": fmt, "width": w, "height": h})

    return info


def print_table(rows: List[Dict[str, Any]], show_details: bool) -> None:
    # 簡易テーブル: Path, Format, WxH, Size
    # 幅揃えのために列幅を計算
    def fmt_wh(r):
        if r.get("width") is None or r.get("height") is None:
            return "-"
        return f"{r['width']}x{r['height']}"

    headers = ["Path", "Format", "Size", "Bytes", "WxH"]
    if show_details:
        headers += ["Mode", "DPI"]

    path_w = max([len(headers[0])] + [len(str(r.get("path", ""))) for r in rows])
    fmt_w = max([len(headers[1])] + [len(str(r.get("format", "-"))) for r in rows])
    sz_w = max([len(headers[2])] + [len(str(r.get("size_human", "-"))) for r in rows])
    byt_w = max([len(headers[3])] + [len(str(r.get("size_bytes", "-"))) for r in rows])
    wh_w = max([len(headers[4])] + [len(fmt_wh(r)) for r in rows])

    def line(hdr=False, r=None):
        if hdr:
            vals = headers
        else:
            vals = [
                str(r.get("path", "")),
                str(r.get("format", "-")),
                str(r.get("size_human", "-")),
                str(r.get("size_bytes", "-")),
                fmt_wh(r),
            ]
            if show_details:
                vals.append(str(r.get("mode", "-")))
                vals.append(str(r.get("dpi", "-")))
        print(
            f"{vals[0]:<{path_w}}  {vals[1]:<{fmt_w}}  {vals[2]:>{sz_w}}  {vals[3]:>{byt_w}}  {vals[4]:>{wh_w}}"
            + ("  " + f"{vals[5]}  {vals[6]}" if show_details else "")
        )

    line(hdr=True)
    for r in rows:
        line(r=r)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="画像サイズ調査ツール")
    p.add_argument("paths", nargs="+", help="調査するファイル/ディレクトリ/グロブパターン")
    p.add_argument("--recursive", action="store_true", help="ディレクトリを再帰的に探索")
    p.add_argument("--json", action="store_true", help="JSON で出力")
    p.add_argument("--details", action="store_true", help="詳細情報 (Pillow 利用時) を出力")
    args = p.parse_args(argv)

    Image, ExifTags = try_import_pillow()

    targets = gather_paths(args.paths, recursive=args.recursive)
    rows = [inspect_path(p, args.details, Image=Image, ExifTags=ExifTags) for p in targets]
    # 見つかった順で安定化（すでに順序は gather_paths 順）

    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        print_table(rows, show_details=args.details)
        print(f"\nFound: {len(rows)} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

