#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iptv_builder.py

把你“已合法获取/有权限使用”的直播源（m3u/txt）聚合、去重，并按自定义分组模板导出：
- output.m3u  (EXTM3U)
- output.txt  (影视仓/TVBox 常见：分组行以 ,#genre# 标识；频道行：名称,URL)

⚠️ 说明：
本脚本不包含“自动在 GitHub 上搜集/抓取疑似侵权 IPTV 源”的功能；
请仅在你确认拥有播放/分发权限的前提下使用你在 source.txt 中提供的源地址。
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import dataclasses
import os
import re
import sys
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse
from urllib.request import Request, urlopen


UA = "Mozilla/5.0 (iptv_builder.py; +https://example.com) Python-urllib"


@dataclasses.dataclass
class Channel:
    name: str
    urls: List[str] = dataclasses.field(default_factory=list)
    group: str = ""
    tvg_name: str = ""
    tvg_logo: str = ""

    def add_url(self, url: str) -> None:
        u = url.strip()
        if not u:
            return
        if u not in self.urls:
            self.urls.append(u)


@dataclasses.dataclass
class TemplateItem:
    group: str
    display_name: str
    aliases: List[str] = dataclasses.field(default_factory=list)
    fixed_urls: List[str] = dataclasses.field(default_factory=list)


def _strip_bom(s: str) -> str:
    return s.lstrip("\ufeff")


def normalize_name(name: str) -> str:
    """
    用于匹配：去掉空格/标点/连接符，保留中文+字母数字并转大写。
    """
    name = name.strip()
    name = re.sub(r"\s+", "", name)
    name = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", name)
    return name.upper()


def smart_decode(data: bytes, charset_hint: Optional[str] = None) -> str:
    """
    尽量把各种源文件解码成 str。
    """
    if charset_hint:
        try:
            return data.decode(charset_hint, errors="replace")
        except Exception:
            pass
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return data.decode(enc, errors="replace")
        except Exception:
            continue
    return data.decode("utf-8", errors="replace")


def read_local_text(path: str) -> str:
    with open(path, "rb") as f:
        data = f.read()
    return smart_decode(data)


def http_get_text(url: str, timeout: int = 15) -> str:
    req = Request(url, headers={"User-Agent": UA})
    with urlopen(req, timeout=timeout) as resp:
        data = resp.read()
        charset = None
        try:
            charset = resp.headers.get_content_charset()
        except Exception:
            charset = None
    return smart_decode(data, charset)


def is_url(s: str) -> bool:
    try:
        p = urlparse(s.strip())
        return p.scheme in ("http", "https")
    except Exception:
        return False


def iter_source_lines(source_file: str) -> List[Tuple[str, str]]:
    """
    读取 source.txt：
    - 支持：URL / 本地文件路径
    - 支持：name|url 形式（name 仅用于日志）
    - # 开头为注释
    返回 [(name, ref), ...]
    """
    text = read_local_text(source_file)
    out: List[Tuple[str, str]] = []
    for raw in text.splitlines():
        line = _strip_bom(raw).strip()
        if not line or line.startswith("#") or line.startswith(";"):
            continue
        # 去掉行内注释（简单处理）
        if " #" in line:
            line = line.split(" #", 1)[0].strip()
        if "|" in line:
            name, ref = [x.strip() for x in line.split("|", 1)]
        else:
            name, ref = "", line
        if ref:
            out.append((name, ref))
    return out


_ATTR_RE = re.compile(r'([A-Za-z0-9\-_]+)="([^"]*)"')


def parse_m3u(text: str) -> List[Channel]:
    channels: List[Channel] = []
    current_group_comment: str = ""
    extgrp_next: str = ""
    pending: Optional[Channel] = None

    for raw in text.splitlines():
        line = _strip_bom(raw).strip()
        if not line:
            continue

        # 组标题注释：#央视频道
        if line.startswith("#") and not line.startswith("#EXT"):
            current_group_comment = line.lstrip("#").strip()
            continue

        # #EXTGRP:xxx
        if line.startswith("#EXTGRP:"):
            extgrp_next = line.split(":", 1)[1].strip()
            continue

        if line.startswith("#EXTINF"):
            pending = Channel(name="")
            # 解析属性
            # EXTINF 行通常有逗号：...,DisplayName
            left, right = (line.split(",", 1) + [""])[:2]
            attrs = dict(_ATTR_RE.findall(left))
            pending.tvg_name = attrs.get("tvg-name", "").strip()
            pending.tvg_logo = attrs.get("tvg-logo", "").strip()
            pending.group = (attrs.get("group-title", "") or "").strip()

            # Display name
            pending.name = right.strip() or pending.tvg_name or "Unknown"

            # 如果 EXTGRP/注释组存在，补上
            if not pending.group:
                if extgrp_next:
                    pending.group = extgrp_next
                elif current_group_comment:
                    pending.group = current_group_comment

            extgrp_next = ""
            continue

        # URL 行
        if pending is not None:
            if line.startswith("#"):
                # 其他注释/指令，忽略
                continue
            pending.add_url(line)
            if pending.urls:
                channels.append(pending)
            pending = None

    return channels


def parse_tvbox_txt(text: str) -> List[Channel]:
    """
    解析常见 TVBox/影视仓 txt：
    央视频道,#genre#
    CCTV-1 综合,http://...
    """
    channels: List[Channel] = []
    current_group = ""
    for raw in text.splitlines():
        line = _strip_bom(raw).strip()
        if not line:
            continue
        if line.startswith("#") or line.startswith(";"):
            continue

        # 分组行
        lower = line.lower()
        if lower.endswith(",#genre#"):
            current_group = line.split(",", 1)[0].strip()
            continue

        # 频道行：name,url1#url2...
        if "," in line:
            name, urls = [x.strip() for x in line.split(",", 1)]
            if not name:
                continue
            ch = Channel(name=name, group=current_group)
            # urls 可能包含多线路，用 # 分隔
            for u in [u.strip() for u in urls.split("#")]:
                if u:
                    ch.add_url(u)
            if ch.urls:
                channels.append(ch)
            continue

        # 无逗号：可能是模板/纯名字，跳过
    return channels


def parse_playlist(text: str) -> List[Channel]:
    """
    根据内容判断并解析。
    """
    t = text.lstrip()
    if t.startswith("#EXTM3U") or "#EXTINF" in text[:5000]:
        return parse_m3u(text)
    if "#genre#" in text:
        return parse_tvbox_txt(text)
    # 兜底：尝试按 name,url 解析
    return parse_tvbox_txt(text)


def merge_channels(all_channels: Dict[str, Channel], incoming: Iterable[Channel]) -> None:
    for ch in incoming:
        key = normalize_name(ch.name)
        if not key:
            continue
        if key not in all_channels:
            all_channels[key] = Channel(
                name=ch.name.strip(),
                group=ch.group.strip(),
                tvg_name=ch.tvg_name.strip(),
                tvg_logo=ch.tvg_logo.strip(),
                urls=[],
            )
        dst = all_channels[key]
        # 补全元信息
        if not dst.group and ch.group:
            dst.group = ch.group
        if not dst.tvg_name and ch.tvg_name:
            dst.tvg_name = ch.tvg_name
        if not dst.tvg_logo and ch.tvg_logo:
            dst.tvg_logo = ch.tvg_logo
        for u in ch.urls:
            dst.add_url(u)


def load_sources(source_items: List[Tuple[str, str]], workers: int = 8, timeout: int = 15) -> Dict[str, Channel]:
    """
    并发加载 source.txt 里列出的多个源，汇总成 {norm_name: Channel}
    """
    all_channels: Dict[str, Channel] = {}

    def _load_one(name: str, ref: str) -> Tuple[str, str, Optional[str], Optional[Exception]]:
        try:
            if is_url(ref):
                text = http_get_text(ref, timeout=timeout)
            else:
                text = read_local_text(ref)
            return name, ref, text, None
        except Exception as e:
            return name, ref, None, e

    with futures.ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        fs = [ex.submit(_load_one, name, ref) for (name, ref) in source_items]
        for f in futures.as_completed(fs):
            name, ref, text, err = f.result()
            if err is not None:
                print(f"[WARN] 加载失败: {name or ''} {ref} -> {err}", file=sys.stderr)
                continue
            if not text:
                continue
            chans = parse_playlist(text)
            merge_channels(all_channels, chans)

    return all_channels


def guess_group_rules_fallback(name: str) -> str:
    """
    没有模板时的粗略分组（可按需扩展）。
    """
    n = name
    if "CCTV" in n.upper() or "央视" in n:
        return "央视频道"
    if "卫视" in n:
        return "卫视频道"
    if "新闻" in n:
        return "新闻频道"
    if "少儿" in n or "儿童" in n:
        return "少儿频道"
    return "其他"


def parse_group_template(path: str) -> List[TemplateItem]:
    """
    解析 goup.txt / group.txt（模板）：
    - 央视频道,#genre#               -> 分组开始
    - CCTV-1 综合|CCTV1|央视一套      -> 频道匹配（| 后为别名/关键词）
    - 也支持直接写死 URL：CCTV-1,http://a#http://b
    """
    text = read_local_text(path)
    items: List[TemplateItem] = []
    current_group: str = "默认分组"

    for raw in text.splitlines():
        line = _strip_bom(raw).strip()
        if not line or line.startswith("#") or line.startswith(";"):
            continue

        low = line.lower()
        if low.endswith(",#genre#"):
            current_group = line.split(",", 1)[0].strip() or "默认分组"
            continue

        # 允许在模板里直接写死 URL（覆盖/补充）
        if "," in line and "://" in line.split(",", 1)[1]:
            display, url_part = [x.strip() for x in line.split(",", 1)]
            fixed_urls = [u.strip() for u in url_part.split("#") if u.strip()]
            aliases = [display]
            items.append(TemplateItem(group=current_group, display_name=display, aliases=aliases, fixed_urls=fixed_urls))
            continue

        # 模板匹配：Display|alias1|alias2...
        parts = [p.strip() for p in line.split("|") if p.strip()]
        if not parts:
            continue
        display = parts[0]
        aliases = parts[:]  # display 也算 alias
        items.append(TemplateItem(group=current_group, display_name=display, aliases=aliases))
    return items


def pick_best_channel(all_channels: Dict[str, Channel], aliases: List[str]) -> Optional[Channel]:
    """
    用 aliases 在 all_channels 里找最匹配的频道：
    优先：归一化后完全一致；其次：包含关系；最后：正则（re:xxx）
    """
    # 1) exact
    for a in aliases:
        if a.startswith("re:"):
            continue
        k = normalize_name(a)
        if k in all_channels:
            return all_channels[k]

    # 2) substring / contains
    norm_aliases = [normalize_name(a) for a in aliases if a and not a.startswith("re:")]
    if norm_aliases:
        best: Tuple[int, Optional[Channel]] = (0, None)
        for key, ch in all_channels.items():
            for na in norm_aliases:
                if not na:
                    continue
                if key == na:
                    return ch
                if na in key or key in na:
                    # 越长越像
                    score = 50 + min(len(na), len(key))
                    if score > best[0]:
                        best = (score, ch)
        if best[1] is not None:
            return best[1]

    # 3) regex (re:...)
    for a in aliases:
        if not a.startswith("re:"):
            continue
        pat = a[3:].strip()
        if not pat:
            continue
        try:
            r = re.compile(pat, flags=re.IGNORECASE)
        except re.error:
            continue
        for ch in all_channels.values():
            if r.search(ch.name):
                return ch

    return None


def build_ordered_playlist(
    all_channels: Dict[str, Channel],
    template: Optional[List[TemplateItem]],
    limit_per_channel: int = 3,
    append_unlisted: bool = True,
) -> List[Channel]:
    """
    按模板输出；模板缺失时则按原 group/规则分组排序。
    """
    out: List[Channel] = []
    used_keys: set[str] = set()

    if template:
        for item in template:
            # 模板行写死 URL：直接用
            if item.fixed_urls:
                ch = Channel(name=item.display_name, group=item.group)
                for u in item.fixed_urls[:limit_per_channel]:
                    ch.add_url(u)
                out.append(ch)
                continue

            found = pick_best_channel(all_channels, item.aliases)
            if found is None or not found.urls:
                continue

            key = normalize_name(found.name)
            used_keys.add(key)

            ch = Channel(
                name=item.display_name or found.name,
                group=item.group or found.group or guess_group_rules_fallback(found.name),
                tvg_name=found.tvg_name,
                tvg_logo=found.tvg_logo,
                urls=found.urls[:limit_per_channel],
            )
            out.append(ch)

        if append_unlisted:
            # 把未在模板中出现的频道追加到最后（按 group 排序）
            remain = []
            for k, ch in all_channels.items():
                if k in used_keys:
                    continue
                remain.append(ch)
            remain.sort(key=lambda c: (c.group or guess_group_rules_fallback(c.name), normalize_name(c.name)))
            for ch0 in remain:
                ch = Channel(
                    name=ch0.name,
                    group=ch0.group or guess_group_rules_fallback(ch0.name),
                    tvg_name=ch0.tvg_name,
                    tvg_logo=ch0.tvg_logo,
                    urls=ch0.urls[:limit_per_channel],
                )
                out.append(ch)
        return out

    # 没有模板：按 group / fallback 分类
    tmp = []
    for ch in all_channels.values():
        g = ch.group or guess_group_rules_fallback(ch.name)
        tmp.append((g, ch))
    tmp.sort(key=lambda x: (x[0], normalize_name(x[1].name)))
    for g, ch0 in tmp:
        out.append(Channel(
            name=ch0.name,
            group=g,
            tvg_name=ch0.tvg_name,
            tvg_logo=ch0.tvg_logo,
            urls=ch0.urls[:limit_per_channel],
        ))
    return out


def write_txt_tvbox(channels: List[Channel], out_path: str) -> None:
    """
    写出 TVBox/影视仓 常见 txt：
    分组行：Group,#genre#
    频道行：Name,url（多线路拆成多行）
    """
    lines: List[str] = [f"# 更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"]
    last_group = None
    for ch in channels:
        g = ch.group or "其他"
        if g != last_group:
            lines.append(f"{g},#genre#")
            last_group = g
        for u in ch.urls:
            if u:
                lines.append(f"{ch.name},{u}")
    data = "\n".join(lines).rstrip() + "\n"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(data)


def write_m3u(channels: List[Channel], out_path: str) -> None:
    lines: List[str] = ["#EXTM3U"]
    last_group = None
    for ch in channels:
        g = ch.group or "其他"
        if g != last_group:
            # 分组注释（可选）
            lines.append(f"# {g}")
            last_group = g

        url = ch.urls[0] if ch.urls else ""
        if not url:
            continue

        attrs = []
        # tvg-name / logo 可选
        tvg_name = ch.tvg_name.strip() or ch.name
        if tvg_name:
            attrs.append(f'tvg-name="{tvg_name}"')
        if ch.tvg_logo.strip():
            attrs.append(f'tvg-logo="{ch.tvg_logo.strip()}"')
        if g:
            attrs.append(f'group-title="{g}"')

        attr_str = " ".join(attrs)
        lines.append(f"#EXTINF:-1 {attr_str},{ch.name}")
        lines.append(url)

    data = "\n".join(lines).rstrip() + "\n"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(data)


def main() -> int:
    ap = argparse.ArgumentParser(description="Merge IPTV playlists and export m3u/txt with custom grouping template.")
    ap.add_argument("-s", "--source", default="source.txt", help="source.txt 路径（每行一个源：URL 或本地文件）")
    ap.add_argument("-g", "--group", default="", help="分组模板文件（group.txt / goup.txt），可选")
    ap.add_argument("-o", "--out", default="output", help="输出文件前缀（不含扩展名），默认 output")
    ap.add_argument("--out-dir", default="", help="输出目录，默认当前目录")
    ap.add_argument("--workers", type=int, default=8, help="并发加载源的线程数")
    ap.add_argument("--timeout", type=int, default=15, help="下载超时时间(秒)")
    ap.add_argument("--limit-per-channel", type=int, default=3, help="每个频道最多保留几条线路（txt 每行一条线路）")
    ap.add_argument("--no-append-unlisted", action="store_true", help="不追加模板外的频道")
    args = ap.parse_args()

    if not os.path.exists(args.source):
        print(f"[ERR] 找不到 source 文件：{args.source}", file=sys.stderr)
        return 2

    # group 文件默认兼容：没传则依次尝试 group.txt / goup.txt
    group_path = args.group.strip()
    if not group_path:
        for cand in ("group.txt", "goup.txt"):
            if os.path.exists(cand):
                group_path = cand
                break

    source_items = iter_source_lines(args.source)
    if not source_items:
        print("[ERR] source.txt 为空或没有可用行。", file=sys.stderr)
        return 2

    all_channels = load_sources(source_items, workers=args.workers, timeout=args.timeout)
    if not all_channels:
        print("[ERR] 没有解析到任何频道条目。请检查源格式或网络。", file=sys.stderr)
        return 2

    template = None
    if group_path and os.path.exists(group_path):
        template = parse_group_template(group_path)

    append_unlisted = False if template else True
    if args.no_append_unlisted:
        append_unlisted = False

    ordered = build_ordered_playlist(
        all_channels,
        template=template,
        limit_per_channel=max(1, args.limit_per_channel),
        append_unlisted=append_unlisted,
    )

    out_dir = args.out_dir.strip()
    out_prefix = args.out
    if out_dir:
        try:
            os.makedirs(out_dir, exist_ok=True)
        except OSError as e:
            print(f"[ERR] 无法创建输出目录：{out_dir} ({e})", file=sys.stderr)
            return 2
        out_prefix = os.path.join(out_dir, os.path.basename(out_prefix))

    out_txt = out_prefix + ".txt"
    out_m3u = out_prefix + ".m3u"
    write_txt_tvbox(ordered, out_txt)
    write_m3u(ordered, out_m3u)

    print(f"[OK] 输出完成：{out_txt}  /  {out_m3u}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
