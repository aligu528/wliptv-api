#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
iptv_check.py

对直播源 URL 做可用性检测（适用于你有权限使用的源）：
- 支持输入：TVBox/影视仓 txt、m3u
- 支持输出：过滤后的 txt + m3u + CSV 报告
- HLS(m3u8) 会进一步探测首段/子播放列表，减少“只回 200 但不可播”的误判

用法示例：
  python iptv_check.py -i output.txt -o output --limit-per-channel 3
  python iptv_check.py -i output.m3u -o live --timeout 8 --workers 40
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen


UA = "Mozilla/5.0 (iptv_check.py) Python-urllib"


@dataclasses.dataclass
class Channel:
    name: str
    group: str = ""
    urls: List[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class ProbeResult:
    url: str
    ok: bool
    elapsed_ms: int
    http_code: int
    note: str


def _strip_bom(s: str) -> str:
    return s.lstrip("\ufeff")


def is_http_url(u: str) -> bool:
    try:
        p = urlparse(u.strip())
        return p.scheme in ("http", "https")
    except Exception:
        return False


def smart_decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return data.decode(enc, errors="replace")
        except Exception:
            pass
    return data.decode("utf-8", errors="replace")


def read_text(path: str) -> str:
    with open(path, "rb") as f:
        return smart_decode(f.read())


def parse_tvbox_txt(text: str) -> List[Channel]:
    """
    央视频道,#genre#
    CCTV-1 综合,http://a
    CCTV-1 综合,http://b
    """
    channels: List[Channel] = []
    index: Dict[Tuple[str, str], Channel] = {}
    group = ""
    for raw in text.splitlines():
        line = _strip_bom(raw).strip()
        if not line or line.startswith("#") or line.startswith(";"):
            continue

        if line.lower().endswith(",#genre#"):
            group = line.split(",", 1)[0].strip()
            continue

        if "," not in line:
            continue

        name, urls_part = [x.strip() for x in line.split(",", 1)]
        if not name or not urls_part:
            continue

        urls = [u.strip() for u in urls_part.split("#") if u.strip()]
        if not urls:
            continue

        key = (group, name)
        if key not in index:
            index[key] = Channel(name=name, group=group, urls=[])
            channels.append(index[key])
        ch = index[key]
        for u in urls:
            if u not in ch.urls:
                ch.urls.append(u)
    return channels


def parse_m3u(text: str) -> List[Channel]:
    """
    简单 m3u 解析：抓 EXTINF 的 display name + 下一行 URL。
    group-title 若没有就留空。
    """
    channels: List[Channel] = []
    group_comment = ""

    # group-title="xxx"
    grp_re = re.compile(r'group-title="([^"]*)"')

    pending_name: Optional[str] = None
    pending_group: str = ""

    for raw in text.splitlines():
        line = _strip_bom(raw).strip()
        if not line:
            continue

        # 组注释
        if line.startswith("#") and not line.startswith("#EXT"):
            group_comment = line.lstrip("#").strip()
            continue

        if line.startswith("#EXTINF"):
            # name 在逗号后
            parts = line.split(",", 1)
            pending_name = (parts[1].strip() if len(parts) > 1 else "").strip() or "Unknown"

            m = grp_re.search(line)
            pending_group = (m.group(1).strip() if m else "") or group_comment
            continue

        # url 行
        if pending_name is not None and not line.startswith("#"):
            channels.append(Channel(name=pending_name, group=pending_group, urls=[line.strip()]))
            pending_name = None
            pending_group = ""
            continue

    return channels


def parse_input_file(path: str) -> List[Channel]:
    text = read_text(path).lstrip()
    if text.startswith("#EXTM3U") or "#EXTINF" in text[:8000]:
        return parse_m3u(text)
    return parse_tvbox_txt(text)


def http_get_bytes(url: str, timeout: int, byte_limit: int = 2048, use_range: bool = True) -> Tuple[int, bytes, str]:
    """
    返回 (http_code, data, content_type)
    """
    headers = {"User-Agent": UA, "Connection": "close"}
    if use_range:
        headers["Range"] = f"bytes=0-{byte_limit-1}"

    req = Request(url, headers=headers, method="GET")
    with urlopen(req, timeout=timeout) as resp:
        code = int(getattr(resp, "status", resp.getcode()))
        ctype = resp.headers.get("Content-Type", "") or ""
        data = resp.read(byte_limit)
        return code, data, ctype


def http_get_text(url: str, timeout: int, use_range: bool = False) -> Tuple[int, str, str]:
    code, data, ctype = http_get_bytes(url, timeout=timeout, byte_limit=200_000, use_range=use_range)
    return code, smart_decode(data), ctype


def pick_first_m3u8_child_or_segment(m3u8_text: str, base_url: str) -> Optional[str]:
    """
    - 若是 master playlist：优先取第一个 #EXT-X-STREAM-INF 后的下一行（子 m3u8）
    - 否则取第一个非注释行（segment 或子 m3u8）
    """
    lines = [ln.strip() for ln in m3u8_text.splitlines() if ln.strip()]

    # master playlist
    for i, ln in enumerate(lines):
        if ln.startswith("#EXT-X-STREAM-INF") and i + 1 < len(lines):
            nxt = lines[i + 1].strip()
            if nxt and not nxt.startswith("#"):
                return urljoin(base_url, nxt)

    # media playlist / fallback
    for ln in lines:
        if ln.startswith("#"):
            continue
        return urljoin(base_url, ln)

    return None


def probe_url(url: str, timeout: int, use_range: bool, hls_deep: bool) -> ProbeResult:
    t0 = time.monotonic()
    url = url.strip()
    if not url:
        return ProbeResult(url=url, ok=False, elapsed_ms=0, http_code=0, note="empty")

    # 仅实现 HTTP/HTTPS；其他协议可在报告里标记 unsupported
    if not is_http_url(url):
        return ProbeResult(url=url, ok=False, elapsed_ms=0, http_code=0, note="unsupported scheme (non-http)")

    try:
        lower = url.lower()
        # HLS 深度检测
        if hls_deep and ("m3u8" in lower):
            code, txt, ctype = http_get_text(url, timeout=timeout, use_range=False)
            if code not in (200, 206):
                ms = int((time.monotonic() - t0) * 1000)
                return ProbeResult(url=url, ok=False, elapsed_ms=ms, http_code=code, note=f"m3u8 http {code}")

            if "#EXTM3U" not in txt[:2000]:
                # 有些会省略，但通常应有；这里给个弱失败
                ms = int((time.monotonic() - t0) * 1000)
                return ProbeResult(url=url, ok=False, elapsed_ms=ms, http_code=code, note="m3u8 missing #EXTM3U")

            child = pick_first_m3u8_child_or_segment(txt, base_url=url)
            if not child:
                ms = int((time.monotonic() - t0) * 1000)
                return ProbeResult(url=url, ok=False, elapsed_ms=ms, http_code=code, note="m3u8 no child/segment")

            # 如果 child 还是 m3u8，继续拉一次取分片
            if "m3u8" in child.lower():
                code2, txt2, _ = http_get_text(child, timeout=timeout, use_range=False)
                if code2 not in (200, 206) or "#EXTM3U" not in txt2[:2000]:
                    ms = int((time.monotonic() - t0) * 1000)
                    return ProbeResult(url=url, ok=False, elapsed_ms=ms, http_code=code2, note="child m3u8 bad")

                seg = pick_first_m3u8_child_or_segment(txt2, base_url=child)
                if not seg:
                    ms = int((time.monotonic() - t0) * 1000)
                    return ProbeResult(url=url, ok=False, elapsed_ms=ms, http_code=code2, note="child m3u8 no segment")

                code3, data3, _ = http_get_bytes(seg, timeout=timeout, byte_limit=2048, use_range=use_range)
                ok = code3 in (200, 206) and len(data3) > 0
                ms = int((time.monotonic() - t0) * 1000)
                return ProbeResult(url=url, ok=ok, elapsed_ms=ms, http_code=code3,
                                   note=("hls ok (segment fetched)" if ok else "hls fail (segment empty/bad)"))

            # child 是分片，直接探测
            code3, data3, _ = http_get_bytes(child, timeout=timeout, byte_limit=2048, use_range=use_range)
            ok = code3 in (200, 206) and len(data3) > 0
            ms = int((time.monotonic() - t0) * 1000)
            return ProbeResult(url=url, ok=ok, elapsed_ms=ms, http_code=code3,
                               note=("hls ok (first segment fetched)" if ok else "hls fail (segment empty/bad)"))

        # 普通 HTTP 流：读一点点字节即可
        code, data, ctype = http_get_bytes(url, timeout=timeout, byte_limit=2048, use_range=use_range)
        ok = code in (200, 206) and len(data) > 0
        ms = int((time.monotonic() - t0) * 1000)
        note = "ok" if ok else f"bad (code={code}, bytes={len(data)})"
        return ProbeResult(url=url, ok=ok, elapsed_ms=ms, http_code=code, note=note)

    except Exception as e:
        ms = int((time.monotonic() - t0) * 1000)
        return ProbeResult(url=url, ok=False, elapsed_ms=ms, http_code=0, note=f"error: {type(e).__name__}: {e}")


def write_tvbox_txt(channels: List[Channel], out_path: str, limit_per_channel: int) -> None:
    lines: List[str] = []
    last_group = None
    for ch in channels:
        urls = [u for u in ch.urls if u][:limit_per_channel]
        if not urls:
            continue
        g = ch.group or "其他"
        if g != last_group:
            lines.append(f"{g},#genre#")
            last_group = g
        for u in urls:
            lines.append(f"{ch.name},{u}")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def write_m3u(channels: List[Channel], out_path: str) -> None:
    lines: List[str] = ["#EXTM3U"]
    last_group = None
    for ch in channels:
        if not ch.urls:
            continue
        g = ch.group or "其他"
        if g != last_group:
            lines.append(f"# {g}")
            last_group = g
        # m3u 每个条目只能放一个 URL：这里取第一条可用线路
        url = ch.urls[0]
        lines.append(f'#EXTINF:-1 group-title="{g}",{ch.name}')
        lines.append(url)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="输入文件：txt 或 m3u")
    ap.add_argument("-o", "--out", default="output", help="输出前缀（默认 output）")
    ap.add_argument("--timeout", type=int, default=8, help="每次请求超时秒数")
    ap.add_argument("--workers", type=int, default=40, help="并发线程数")
    ap.add_argument("--limit-per-channel", type=int, default=3, help="每个频道最多保留几条可用线路")
    ap.add_argument("--no-range", action="store_true", help="不使用 Range 请求（某些源不支持 Range）")
    ap.add_argument("--no-hls-deep", action="store_true", help="不做 m3u8 深度检测（只测 m3u8 本身是否可拉取）")
    args = ap.parse_args()

    channels = parse_input_file(args.input)
    if not channels:
        print("[ERR] 没解析到频道，请检查输入格式。", file=sys.stderr)
        return 2

    # 收集唯一 URL
    all_urls: List[str] = []
    seen = set()
    for ch in channels:
        for u in ch.urls:
            u = u.strip()
            if not u or u in seen:
                continue
            seen.add(u)
            all_urls.append(u)

    if not all_urls:
        print("[ERR] 没有可检测的 URL。", file=sys.stderr)
        return 2

    print(f"[INFO] 频道数: {len(channels)}，唯一 URL 数: {len(all_urls)}")
    use_range = not args.no_range
    hls_deep = not args.no_hls_deep

    results: Dict[str, ProbeResult] = {}

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = {ex.submit(probe_url, u, args.timeout, use_range, hls_deep): u for u in all_urls}
        done = 0
        for fut in as_completed(futs):
            r = fut.result()
            results[r.url] = r
            done += 1
            if done % 50 == 0 or done == len(all_urls):
                print(f"[INFO] progress: {done}/{len(all_urls)}")

    # 过滤频道：保留可用 URL
    checked_channels: List[Channel] = []
    channel_rows: List[Tuple[str, str, int, int, str]] = []

    for ch in channels:
        ok_urls = []
        for u in ch.urls:
            pr = results.get(u.strip())
            if pr and pr.ok:
                ok_urls.append(u.strip())

        ok_urls = ok_urls[: max(1, args.limit_per_channel)]
        total = len(ch.urls)
        okc = len(ok_urls)
        channel_rows.append((ch.group or "其他", ch.name, okc, total, "#".join(ok_urls)))

        if ok_urls:
            checked_channels.append(Channel(name=ch.name, group=ch.group, urls=ok_urls))

    # 输出文件
    out_txt = f"{args.out}_checked.txt"
    out_m3u = f"{args.out}_checked.m3u"
    out_urls_csv = f"{args.out}_report_urls.csv"
    out_ch_csv = f"{args.out}_report_channels.csv"

    write_tvbox_txt(checked_channels, out_txt, limit_per_channel=max(1, args.limit_per_channel))
    write_m3u(checked_channels, out_m3u)

    # URL 报告
    with open(out_urls_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["url", "ok", "elapsed_ms", "http_code", "note"])
        for u in all_urls:
            r = results.get(u)
            if r:
                w.writerow([r.url, int(r.ok), r.elapsed_ms, r.http_code, r.note])
            else:
                w.writerow([u, 0, 0, 0, "no result"])

    # Channel 报告
    with open(out_ch_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "channel", "ok_count", "total_count", "kept_urls"])
        for row in channel_rows:
            w.writerow(list(row))

    ok_url_count = sum(1 for r in results.values() if r.ok)
    print(f"[OK] 可用 URL: {ok_url_count}/{len(all_urls)}")
    print(f"[OK] 输出：{out_txt} / {out_m3u}")
    print(f"[OK] 报告：{out_urls_csv} / {out_ch_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
