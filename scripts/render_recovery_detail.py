"""Build the annotated partner-motion film from the native replay and socket-frame sections.

Run with Python 3, Pillow (DejaVu Sans fonts), NumPy, and ffmpeg from any directory.
Regenerate recovery-geometry.json with replay_recovery_geometry.py if required.
"""

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

MEDIA = Path(__file__).resolve().parents[1] / "paper/media"
WIDTH, HEIGHT, HALF = 840, 620, 420
BG = "#111a22"
TEXT, MUTED = "#f3f6f8", "#c6d2dd"
WALL, BORE = "#4271a2", "#101a23"
SHAFT, EDGE = "#318953", "#a2e7aa"
FAIL, PASS = "#ffb398", "#9aebc4"
ANGLE = np.linspace(0, 2 * np.pi, 241)
CIRCLE = np.array((np.cos(ANGLE), np.sin(ANGLE)))
FONTS = {
    size: ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    for size in (17, 20, 22, 26, 30, 32)
}
NORMALS = {"-x": (-1, 0), "+x": (1, 0), "-y": (0, -1), "+y": (0, 1)}


def text(d, xy, message, size, fill=TEXT, anchor="lm"):
    d.text(xy, message, font=FONTS[size], fill=fill, anchor=anchor)


def draw_section(image, x, data, opening):
    """Both views show the true ellipse at the face nearest a wall."""
    d = ImageDraw.Draw(image)
    center = np.array(data["centers_mm"][data["limiting_face"]])
    basis = np.array(data["basis_mm"])
    edge = (center[:, None] + basis @ CIRCLE).T
    normal = np.array(NORMALS[data["limiting_wall"]])
    tangent = np.array([-normal[1], normal[0]])
    # Support point of the oblique cylinder cross-section along the wall normal.
    support = basis.T @ normal
    point = center + basis @ (support / np.linalg.norm(support))
    gap = opening - np.dot(point, normal)
    if abs(gap - data["min_clearance_mm"]) > 0.0001:
        raise ValueError("Drawn shaft edge disagrees with native clearance")

    text(d, (x + 36, 317), "Socket section", 20)
    text(d, (x + 245, 317), "Wall detail", 20)
    scale, cx, cy = 4.95, x + 124, 426
    lo, hi = opening * scale, opening * scale + 6
    d.rectangle((cx - hi, cy - hi, cx + hi, cy + hi), fill=WALL)
    d.rectangle((cx - lo, cy - lo, cx + lo, cy + lo), fill=BORE)
    outline = [(cx + scale * px, cy - scale * py) for px, py in edge]
    d.polygon(outline, fill=SHAFT)
    d.line(outline, fill=EDGE, width=2)
    # Outline the region enlarged in the neighboring, wall-aligned view.
    along = np.dot(point, tangent)
    window = [
        normal * (opening + u) + tangent * (along + v)
        for u, v in [
            (-3.45, -2.85),
            (1.15, -2.85),
            (1.15, 2.85),
            (-3.45, 2.85),
            (-3.45, -2.85),
        ]
    ]
    d.line([(cx + scale * p[0], cy - scale * p[1]) for p in window], fill=TEXT, width=2)

    # Always place the nearest wall on the right of the magnification.
    zoom = Image.new("RGB", (140, 172), BORE)
    zd = ImageDraw.Draw(zoom)
    z, wall_x, mid_y = 30, 104, 86
    zd.rectangle((wall_x, 0, zoom.width, zoom.height), fill=WALL)
    coords = [
        (wall_x + z * (p @ normal - opening), mid_y - z * (p @ tangent - along))
        for p in edge
    ]
    zd.polygon(coords, fill=SHAFT)
    zd.line(coords, fill=EDGE, width=2)
    zd.line((wall_x, 0, wall_x, zoom.height), fill="#b5d2f0", width=2)
    a = wall_x - z * gap
    zd.line((a, mid_y, wall_x, mid_y), fill="#fff1a7", width=3)
    for xp in (a, wall_x):
        zd.line((xp, mid_y - 5, xp, mid_y + 5), fill="#fff1a7", width=2)
    image.paste(zoom, (x + 245, 340))


def draw_frame(source, geom, receipt, i):
    canvas = Image.new("RGB", (WIDTH, HEIGHT), BG)
    d = ImageDraw.Draw(canvas)
    phases = [
        (0, "Same starting grasp"),
        (22, "Pin arm moves left"),
        (54, "Socket follow-up begins"),
        (88, "Insertion check 1 of 2"),
        (89, "Settling"),
        (100, "Insertion check 2 of 2"),
    ]
    stage = next(label for start, label in reversed(phases) if i >= start)
    text(d, (19, 25), stage, 26)
    text(d, (WIDTH - 18, 25), f"{geom['time_s'][i]:.2f} s", 26, MUTED, "rm")

    for col, (branch, title) in enumerate(
        (("hold", "Hold"), ("ppo", "PPO · move left"))
    ):
        x = col * HALF
        # Crop the pre-existing text and the far edge of each native camera view.
        view = source.crop((600 * col + 74, 48, 600 * col + 526, 282))
        canvas.paste(view.resize((HALF, 218), Image.Resampling.LANCZOS), (x, 52))
        if i >= 54:
            width = d.textlength(title, font=FONTS[26])
            d.rounded_rectangle((x + 12, 60, x + 32 + width, 98), radius=5, fill=BG)
            text(d, (x + 22, 79), title, 26)
        data = geom["branches"][branch][i]
        draw_section(canvas, x, data, geom["half_opening_mm"])
        gap = data["min_clearance_mm"]
        check = {88: 0, 100: 1}.get(i)
        color = TEXT
        if check is not None:
            end = receipt["endpoints"][branch][check]
            if abs(end["clearance_mm"] - gap) > 0.0001:
                raise ValueError(
                    "Endpoint measurement differs from the released replay"
                )
            color = PASS if end["ok"] else FAIL
            text(
                d,
                (x + HALF / 2, 578),
                "PASS" if end["ok"] else "FAIL · wall clearance",
                22,
                color,
                "mm",
            )
        else:
            text(d, (x + HALF / 2, 578), "Wall clearance", 20, MUTED, "mm")
        text(
            d, (x + HALF / 2, 543), f"{gap:+.3f} mm".replace("-", "−"), 32, color, "mm"
        )
    d.line((HALF, 52, HALF, 270), fill=BG, width=3)
    d.line((HALF, 307, HALF, HEIGHT - 20), fill="#374651", width=1)
    text(
        d,
        (WIDTH / 2, 285),
        "Minimum wall clearance   ≥ +0.10 mm at both checks",
        22,
        TEXT,
        "mm",
    )
    return canvas


def main():
    geometry = json.loads((MEDIA / "recovery-geometry.json").read_text())
    receipt = json.loads((MEDIA / "follow-up-comparison.json").read_text())
    if (geometry["family_id"], geometry["matched_state_sha256"]) != (
        receipt["family_id"],
        receipt["matched_state_sha256"],
    ):
        raise ValueError("Geometry and footage belong to different decision states")
    footage = MEDIA / "follow-up-comparison.mp4"
    if hashlib.sha256(footage.read_bytes()).hexdigest() != receipt["video_sha256"]:
        raise ValueError("Unexpected source MP4")
    if len(geometry["time_s"]) != 101 or not np.allclose(
        geometry["time_s"], np.arange(101) / 100
    ):
        raise ValueError("Unexpected geometry timeline")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(footage),
                "-threads",
                "2",
                "-start_number",
                "0",
                str(tmp / "input-%03d.png"),
            ],
            check=True,
        )
        if len(list(tmp.glob("input-*.png"))) != 101:
            raise ValueError("Unexpected native film length")
        movie = MEDIA / "recovery-wall-detail.mp4"
        cmd = [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{WIDTH}x{HEIGHT}",
            "-framerate",
            "25",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-threads",
            "3",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(movie),
        ]
        with subprocess.Popen(cmd, stdin=subprocess.PIPE) as encoder:
            for i in range(101):
                with Image.open(tmp / f"input-{i:03d}.png") as source:
                    frame = draw_frame(source, geometry, receipt, i)
                # Keep key frames readable. The clock always shows simulation time.
                repeats = {0: 8, 54: 24, 88: 32, 100: 52}.get(i, 1)
                encoder.stdin.write(frame.tobytes() * repeats)
                if i == 100:
                    frame.save(MEDIA / "recovery-wall-detail.png", optimize=True)
            encoder.stdin.close()
            if encoder.wait():
                raise RuntimeError("MP4 encoding failed")
        palette_filter = (
            "fps=12,scale=630:-2:flags=lanczos,split[a][b];"
            "[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4"
        )
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-i",
                str(movie),
                "-filter_complex_threads",
                "1",
                "-filter_complex",
                palette_filter,
                "-loop",
                "0",
                str(MEDIA / "recovery-wall-detail.gif"),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
