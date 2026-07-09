#!/usr/bin/env python3
"""Record 4 parallel live vLLM streams, then render a real-time 30fps demo video.

No speedup: frame N shows wall-clock state at t = N/30 seconds.
"""
from __future__ import annotations

import asyncio
import json
import math
import textwrap
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import httpx
from PIL import Image, ImageDraw, ImageFont

# ── config ──────────────────────────────────────────────────────────────────
ENDPOINT = "http://localhost:8000/v1/chat/completions"
MODEL = "nemotron-puzzle-75b-nvfp4"
MODEL_LABEL = "Nemotron Puzzle 75B-A9B  ·  NVFP4  ·  MTP×3"
HARDWARE = "1× NVIDIA DGX Spark  ·  GB10  ·  max_num_seqs=4"
OUT_DIR = Path(__file__).resolve().parent / "out"
FPS = 30
W, H = 1920, 1080
MAX_TOKENS = 220
INTRO_S = 1.4
HOLD_S = 3.0

PROMPTS = [
    (
        "CODE",
        "Write a clean Python LRU cache class with get/put. Include type hints and a short docstring. Code only.",
    ),
    (
        "SYSTEMS",
        "In 8–10 sentences, explain how speculative decoding (MTP) increases tokens/sec on a single GPU without changing model weights.",
    ),
    (
        "STRUCTURED",
        "Emit a JSON object only (no markdown) with keys: service, endpoints (array of 4 objects with path and method), sla_ms, notes. Realistic values for a local LLM API.",
    ),
    (
        "ENGINEERING",
        "Give a tight checklist (8 bullets) for serving an NVFP4 MoE model on unified-memory GB10: memory, KV cache, concurrency, and prefix caching.",
    ),
]

SYSTEM = (
    "You are a precise technical assistant. Follow the user request exactly. "
    "No preambles. No closing questions."
)

# ── palette (muted, professional) ───────────────────────────────────────────
BG = (9, 11, 14)
PANEL = (15, 18, 24)
PANEL_EDGE = (36, 44, 56)
HEADER_LINE = (46, 120, 168)
TEXT = (226, 232, 240)
MUTED = (132, 144, 160)
DIM = (88, 98, 112)
ACCENT = (78, 168, 214)
GREEN = (92, 196, 140)
AMBER = (210, 168, 86)
STREAM_ACCENT = [
    (78, 168, 214),
    (92, 196, 140),
    (168, 140, 214),
    (210, 168, 86),
]


def font(path_candidates, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for p in path_candidates:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


SANS = [
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
SANS_B = [
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]
MONO = [
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
]
MONO_B = [
    "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
]

F_TITLE = font(SANS_B, 34)
F_SUB = font(SANS, 20)
F_BIG = font(SANS_B, 56)
F_MED = font(SANS_B, 22)
F_SMALL = font(SANS, 16)
F_TINY = font(SANS, 14)
F_MONO = font(MONO, 15)
F_MONO_SM = font(MONO, 13)
F_LABEL = font(MONO_B, 14)


@dataclass
class TokenEvent:
    t: float  # seconds since global t0
    text: str


@dataclass
class StreamRec:
    idx: int
    label: str
    prompt: str
    events: list[TokenEvent] = field(default_factory=list)
    start_t: float | None = None
    end_t: float | None = None
    ttft: float | None = None
    completion_tokens: int = 0
    prompt_tokens: int = 0
    final_text: str = ""
    error: str | None = None

    def text_at(self, t: float) -> str:
        if self.start_t is None or t < self.start_t:
            return ""
        out = []
        for ev in self.events:
            if ev.t <= t:
                out.append(ev.text)
            else:
                break
        return "".join(out)

    def tokens_est_at(self, t: float) -> float:
        """Scale char progress to final completion_tokens for honest display."""
        if not self.final_text or self.completion_tokens <= 0:
            txt = self.text_at(t)
            return max(0.0, len(txt) / 4.0)
        frac = min(1.0, len(self.text_at(t)) / max(1, len(self.final_text)))
        return self.completion_tokens * frac


async def run_one(
    client: httpx.AsyncClient,
    rec: StreamRec,
    t0: float,
    max_tokens: int,
) -> None:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": rec.prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.35,
        "stream": True,
        "stream_options": {"include_usage": True},
        # Nemotron-v3: keep reasoning off so panes show final tokens (demo-ready)
        "chat_template_kwargs": {"enable_thinking": False},
    }
    rec.start_t = time.time() - t0
    try:
        async with client.stream("POST", ENDPOINT, json=payload, timeout=300.0) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if data == "[DONE]":
                    break
                try:
                    j = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if j.get("usage"):
                    u = j["usage"]
                    rec.completion_tokens = int(u.get("completion_tokens") or 0)
                    rec.prompt_tokens = int(u.get("prompt_tokens") or 0)
                ch = j.get("choices") or []
                if not ch:
                    continue
                d = ch[0].get("delta") or {}
                # Final answer tokens only — never show hidden reasoning in the video
                piece = d.get("content") or ""
                if not piece:
                    continue
                now = time.time() - t0
                if rec.ttft is None:
                    rec.ttft = now - rec.start_t
                rec.events.append(TokenEvent(t=now, text=piece))
                rec.final_text += piece
    except Exception as e:
        rec.error = str(e)
    finally:
        rec.end_t = time.time() - t0
        if rec.completion_tokens <= 0 and rec.final_text:
            # fallback estimate if usage missing
            rec.completion_tokens = max(1, len(rec.final_text) // 4)


async def record_all() -> tuple[list[StreamRec], float]:
    streams = [
        StreamRec(idx=i, label=label, prompt=prompt)
        for i, (label, prompt) in enumerate(PROMPTS)
    ]
    # Warmup single short request so first-token is clean
    async with httpx.AsyncClient(http2=False, timeout=60.0) as client:
        warm = {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Reply: ok"}],
            "max_tokens": 4,
            "temperature": 0,
        }
        await client.post(ENDPOINT, json=warm)
        await asyncio.sleep(0.4)

        t0 = time.time()
        await asyncio.gather(*[run_one(client, s, t0, MAX_TOKENS) for s in streams])
        wall = time.time() - t0
    return streams, wall


def rounded_rect(draw: ImageDraw.ImageDraw, xy, r: int, fill, outline=None, width=1):
    draw.rounded_rectangle(xy, radius=r, fill=fill, outline=outline, width=width)


def wrap_mono(text: str, width: int = 52) -> list[str]:
    lines: list[str] = []
    for para in text.splitlines() or [""]:
        if not para:
            lines.append("")
            continue
        lines.extend(textwrap.wrap(para, width=width, replace_whitespace=False) or [""])
    return lines


def draw_header(draw: ImageDraw.ImageDraw, agg_tps: float, active: int, elapsed: float, phase: str):
    # top accent bar
    draw.rectangle((0, 0, W, 3), fill=HEADER_LINE)
    draw.text((48, 28), MODEL_LABEL, fill=TEXT, font=F_TITLE)
    draw.text((48, 72), HARDWARE, fill=MUTED, font=F_SUB)

    # right metrics cluster
    box_x0, box_y0, box_x1, box_y1 = 1180, 20, 1872, 118
    rounded_rect(draw, (box_x0, box_y0, box_x1, box_y1), 10, PANEL, PANEL_EDGE, 1)
    draw.text((box_x0 + 24, box_y0 + 14), "AGGREGATE THROUGHPUT", fill=DIM, font=F_TINY)
    tps_str = f"{agg_tps:5.1f}"
    draw.text((box_x0 + 24, box_y0 + 36), tps_str, fill=GREEN, font=F_BIG)
    draw.text((box_x0 + 230, box_y0 + 58), "tok/s", fill=MUTED, font=F_MED)
    draw.text(
        (box_x0 + 360, box_y0 + 28),
        f"streams  {active}/4\nelapsed  {elapsed:5.1f}s\nphase    {phase}",
        fill=MUTED,
        font=F_MONO_SM,
    )


def draw_stream_panel(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    rec: StreamRec,
    t: float,
    accent,
):
    x0, y0, x1, y1 = xy
    rounded_rect(draw, xy, 12, PANEL, PANEL_EDGE, 1)
    # left accent strip
    draw.rectangle((x0, y0 + 12, x0 + 4, y1 - 12), fill=accent)

    running = rec.start_t is not None and t >= rec.start_t and (
        rec.end_t is None or t < rec.end_t
    )
    done = rec.end_t is not None and t >= rec.end_t
    waiting = rec.start_t is None or t < (rec.start_t or 0)

    status = "WAITING"
    status_c = DIM
    if running:
        status = "GENERATING"
        status_c = accent
    elif done:
        status = "COMPLETE"
        status_c = GREEN
    if rec.error and done:
        status = "ERROR"
        status_c = (220, 90, 90)

    # title row
    draw.text((x0 + 20, y0 + 14), f"STREAM {rec.idx + 1:02d}  ·  {rec.label}", fill=TEXT, font=F_LABEL)
    draw.text((x1 - 160, y0 + 14), status, fill=status_c, font=F_LABEL)

    # prompt one-liner
    prompt_line = rec.prompt.replace("\n", " ")
    if len(prompt_line) > 78:
        prompt_line = prompt_line[:75] + "…"
    draw.text((x0 + 20, y0 + 38), prompt_line, fill=DIM, font=F_TINY)

    # separator
    draw.line((x0 + 16, y0 + 62, x1 - 16, y0 + 62), fill=PANEL_EDGE, width=1)

    # body text
    body = rec.text_at(t)
    if waiting and not body:
        body = "▸ waiting for first token…"
        body_color = DIM
    elif rec.error and done:
        body = rec.error
        body_color = (220, 120, 120)
    else:
        body_color = TEXT

    lines = wrap_mono(body, width=54)
    # show last N lines (terminal scrollback feel)
    max_lines = 16
    view = lines[-max_lines:]
    ty = y0 + 74
    for line in view:
        draw.text((x0 + 20, ty), line, fill=body_color, font=F_MONO)
        ty += 18

    # footer metrics
    draw.line((x0 + 16, y1 - 42, x1 - 16, y1 - 42), fill=PANEL_EDGE, width=1)
    toks = rec.tokens_est_at(t)
    if rec.start_t is not None and t > rec.start_t:
        dt = max(1e-3, min(t, rec.end_t or t) - rec.start_t)
        tps = toks / dt
    else:
        tps = 0.0
    ttft_s = rec.ttft if (rec.ttft is not None and (done or running)) else None
    ttft_str = f"{ttft_s * 1000:.0f} ms" if ttft_s is not None else "—"
    draw.text(
        (x0 + 20, y1 - 30),
        f"{tps:5.1f} tok/s    tokens {toks:6.0f}    ttft {ttft_str}",
        fill=MUTED,
        font=F_MONO_SM,
    )


def state_at(streams: list[StreamRec], t_abs: float, gen_start: float) -> tuple[float, int, float, str]:
    """Return agg_tps, active, elapsed_gen, phase."""
    if t_abs < gen_start:
        return 0.0, 0, 0.0, "ready"
    t = t_abs  # stream times are absolute from record t0; we shift in render
    # In render we use t_rel where gen starts at gen_start
    return 0.0, 0, 0.0, "run"  # filled by caller


def render_frame(
    streams: list[StreamRec],
    t_stream: float | None,
    phase: str,
    intro_elapsed: float,
    wall_total: float,
) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    if phase == "intro":
        agg = 0.0
        active = 0
        elapsed = 0.0
        # still draw empty panels
        t_query = -1.0
    else:
        t_query = t_stream if t_stream is not None else wall_total
        total_tok = sum(s.tokens_est_at(t_query) for s in streams)
        # elapsed from first stream start
        starts = [s.start_t for s in streams if s.start_t is not None]
        start0 = min(starts) if starts else 0.0
        elapsed = max(0.0, t_query - start0)
        agg = total_tok / elapsed if elapsed > 0.05 else 0.0
        active = sum(
            1
            for s in streams
            if s.start_t is not None
            and t_query >= s.start_t
            and (s.end_t is None or t_query < s.end_t)
        )
        if phase == "hold":
            # final numbers
            total_tok = sum(s.completion_tokens for s in streams)
            ends = [s.end_t for s in streams if s.end_t is not None]
            end0 = max(ends) if ends else wall_total
            elapsed = max(0.05, end0 - start0)
            agg = total_tok / elapsed
            active = 0

    draw_header(draw, agg, active if phase != "hold" else 4, elapsed, phase)

    # 2×2 grid
    margin_x = 40
    top = 140
    gap = 18
    pane_w = (W - margin_x * 2 - gap) // 2
    pane_h = (H - top - 56 - gap) // 2
    positions = [
        (margin_x, top, margin_x + pane_w, top + pane_h),
        (margin_x + pane_w + gap, top, margin_x + 2 * pane_w + gap, top + pane_h),
        (margin_x, top + pane_h + gap, margin_x + pane_w, top + 2 * pane_h + gap),
        (
            margin_x + pane_w + gap,
            top + pane_h + gap,
            margin_x + 2 * pane_w + gap,
            top + 2 * pane_h + gap,
        ),
    ]

    for i, rec in enumerate(streams):
        t_for = t_query if phase != "intro" else -1.0
        if phase == "hold":
            t_for = 1e9  # show final text
        draw_stream_panel(draw, positions[i], rec, t_for, STREAM_ACCENT[i])

    # footer
    footer = (
        "Live inference  ·  no speed-up  ·  wall-clock playback @ 30 fps  ·  "
        "OpenAI-compatible /v1/chat/completions"
    )
    draw.text((48, H - 36), footer, fill=DIM, font=F_TINY)
    draw.text((W - 320, H - 36), "local · offline", fill=DIM, font=F_TINY)
    return img


def render_video(streams: list[StreamRec], wall: float) -> Path:
    frames_dir = OUT_DIR / "frames"
    if frames_dir.exists():
        for p in frames_dir.glob("*.png"):
            p.unlink()
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Stream timestamps are relative to record t0 (first request launch).
    duration = INTRO_S + wall + HOLD_S
    n_frames = int(math.ceil(duration * FPS))
    print(f"Rendering {n_frames} frames ({duration:.2f}s @ {FPS} fps)…")

    for i in range(n_frames):
        t_abs = i / FPS
        if t_abs < INTRO_S:
            phase = "intro"
            t_stream = None
            img = render_frame(streams, t_stream, phase, t_abs, wall)
        elif t_abs < INTRO_S + wall:
            phase = "live"
            t_stream = t_abs - INTRO_S
            img = render_frame(streams, t_stream, phase, t_abs, wall)
        else:
            phase = "hold"
            img = render_frame(streams, wall, phase, t_abs, wall)

        img.save(frames_dir / f"frame_{i:05d}.png")
        if i % (FPS * 2) == 0 or i == n_frames - 1:
            print(f"  frame {i+1}/{n_frames}")

    # poster = mid-live or last live
    mid = min(n_frames - 1, int((INTRO_S + wall * 0.55) * FPS))
    poster = OUT_DIR / "poster.png"
    Image.open(frames_dir / f"frame_{mid:05d}.png").save(poster)

    out_mp4 = OUT_DIR / "nemotron-puzzle-4stream-30fps.mp4"
    # Real-time encode, yuv420p, high quality for social
    import subprocess

    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(FPS),
        "-i",
        str(frames_dir / "frame_%05d.png"),
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "16",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out_mp4),
    ]
    print("Encoding", out_mp4)
    subprocess.run(cmd, check=True)
    return out_mp4


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Recording 4 parallel live streams…")
    streams, wall = asyncio.run(record_all())

    total_tok = sum(s.completion_tokens for s in streams)
    starts = [s.start_t for s in streams if s.start_t is not None]
    ends = [s.end_t for s in streams if s.end_t is not None]
    span = (max(ends) - min(starts)) if starts and ends else wall
    agg = total_tok / span if span > 0 else 0

    summary = {
        "model": MODEL,
        "wall_s": wall,
        "span_s": span,
        "total_completion_tokens": total_tok,
        "aggregate_tok_s": agg,
        "streams": [
            {
                "idx": s.idx,
                "label": s.label,
                "prompt": s.prompt,
                "completion_tokens": s.completion_tokens,
                "prompt_tokens": s.prompt_tokens,
                "ttft_s": s.ttft,
                "start_t": s.start_t,
                "end_t": s.end_t,
                "events": len(s.events),
                "error": s.error,
                "text_preview": s.final_text[:240],
            }
            for s in streams
        ],
    }
    (OUT_DIR / "recording.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    # Save full texts
    for s in streams:
        (OUT_DIR / f"stream_{s.idx+1}_{s.label.lower()}.txt").write_text(s.final_text)

    # Persist events for exact replay
    events_blob = {
        "wall_s": wall,
        "streams": [
            {
                **{k: getattr(s, k) for k in ("idx", "label", "prompt", "start_t", "end_t", "ttft", "completion_tokens", "prompt_tokens", "final_text", "error")},
                "events": [asdict(e) for e in s.events],
            }
            for s in streams
        ],
    }
    (OUT_DIR / "events.json").write_text(json.dumps(events_blob))

    out = render_video(streams, wall)
    print("DONE:", out)
    print(f"Aggregate: {agg:.1f} tok/s  ({total_tok} tokens / {span:.2f}s)")


if __name__ == "__main__":
    main()
