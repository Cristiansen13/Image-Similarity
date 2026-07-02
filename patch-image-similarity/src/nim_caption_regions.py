"""Region captioning via NVIDIA NIM — the spec's actual recipe: one call per
image, a numbered grid overlay, structured JSON with one description per cell.

Requires NVIDIA_API_KEY in .env (see src/config.py). Uses NIM's
OpenAI-compatible API with a vision-instruction model.
"""
import base64
import io
import json
import os
import re
import time
from dataclasses import dataclass, asdict

from openai import APIConnectionError, APIStatusError, OpenAI
from PIL import Image, ImageDraw, ImageFont

from config import load_env

DEFAULT_MODEL = "meta/llama-3.2-11b-vision-instruct"
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
MAX_IMAGE_BYTES = 170_000  # stay under NIM's inline base64 image size limit


@dataclass
class Region:
    row: int
    col: int
    description: str


def _draw_grid_overlay(image: Image.Image, grid_size: int) -> Image.Image:
    image = image.convert("RGB").copy()
    w, h = image.size
    draw = ImageDraw.Draw(image)
    cell_w, cell_h = w / grid_size, h / grid_size

    for i in range(1, grid_size):
        draw.line([(i * cell_w, 0), (i * cell_w, h)], fill=(255, 0, 0), width=2)
        draw.line([(0, i * cell_h), (w, i * cell_h)], fill=(255, 0, 0), width=2)

    font = ImageFont.load_default(size=max(14, int(min(cell_w, cell_h) * 0.25)))
    index = 1
    for row in range(grid_size):
        for col in range(grid_size):
            x, y = col * cell_w + 4, row * cell_h + 2
            label = str(index)
            bbox = draw.textbbox((x, y), label, font=font)
            draw.rectangle(bbox, fill=(255, 255, 255))
            draw.text((x, y), label, fill=(255, 0, 0), font=font)
            index += 1
    return image


def _to_data_uri(image: Image.Image) -> str:
    data = None
    for quality in (85, 65, 45, 30):
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=quality)
        data = buf.getvalue()
        if len(data) <= MAX_IMAGE_BYTES:
            break
    b64 = base64.b64encode(data).decode()
    return f"data:image/jpeg;base64,{b64}"


def _extract_json(text: str) -> dict:
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)
    else:
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            text = brace_match.group(0)
    return json.loads(text)


class NimRegionCaptioner:
    def __init__(self, model_name: str = DEFAULT_MODEL, cache_path: str | None = None):
        load_env()
        api_key = os.environ.get("NVIDIA_API_KEY")
        if not api_key:
            raise RuntimeError("NVIDIA_API_KEY not set (expected in .env)")
        self.client = OpenAI(base_url=NIM_BASE_URL, api_key=api_key)
        self.model_name = model_name
        self.cache_path = cache_path
        self._cache = {}
        if cache_path and os.path.exists(cache_path):
            with open(cache_path) as f:
                self._cache = json.load(f)

    def _save_cache(self):
        if self.cache_path:
            with open(self.cache_path, "w") as f:
                json.dump(self._cache, f, indent=2)

    def caption_regions(self, image_path: str, grid_size: int = 4) -> list[Region]:
        key = f"{image_path}|{os.path.getmtime(image_path)}|{grid_size}|{self.model_name}"
        if key in self._cache:
            return [Region(**r) for r in self._cache[key]]

        image = Image.open(image_path).convert("RGB")
        overlay = _draw_grid_overlay(image, grid_size)
        data_uri = _to_data_uri(overlay)

        n = grid_size * grid_size
        prompt = (
            f"This image has a red {grid_size}x{grid_size} grid overlaid on it. Cells are "
            f"numbered 1 to {n}, left to right then top to bottom, with the number printed "
            f"in the top-left corner of each cell. For every cell number, write one short "
            f"phrase (3-8 words) describing what is in that region, using the whole image "
            f"for context. Respond with ONLY a JSON object mapping each cell number (as a "
            f'string) to its description, e.g. {{"1": "...", "2": "..."}}. Include all {n} '
            f"keys."
        )

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ],
        }]

        parsed = None
        last_raw = None
        for attempt in range(4):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=1536,
                )
            except (APIStatusError, APIConnectionError) as e:
                wait = 2 ** attempt
                print(f"  [warn] API error for {os.path.basename(image_path)} "
                      f"(attempt {attempt + 1}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
                continue
            raw = response.choices[0].message.content
            last_raw = raw
            try:
                parsed = _extract_json(raw)
                break
            except json.JSONDecodeError:
                print(f"  [warn] non-JSON response for {os.path.basename(image_path)} "
                      f"(attempt {attempt + 1}): {raw[:200]!r}")

        if parsed is None:
            print(f"  [warn] giving up on {os.path.basename(image_path)}, using empty captions. "
                  f"Last raw response: {(last_raw or '(no response -- all attempts errored)')[:300]!r}")
            parsed = {}

        regions = []
        index = 1
        for row in range(grid_size):
            for col in range(grid_size):
                description = str(parsed.get(str(index), "")).strip() or "(no description)"
                regions.append(Region(row=row, col=col, description=description))
                index += 1

        self._cache[key] = [asdict(r) for r in regions]
        self._save_cache()
        return regions
