from __future__ import annotations

import argparse
import hashlib
import json
import re
import textwrap
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


WOWHEAD_SCREENSHOT_RE = re.compile(
    r"(?:https?:)?//wow\.zamimg\.com/uploads/screenshots/normal/[A-Za-z0-9_.%/-]+?\.(?:jpg|jpeg|png)",
    re.IGNORECASE,
)
WOWHEAD_RELATIVE_SCREENSHOT_RE = re.compile(
    r"uploads/screenshots/normal/[A-Za-z0-9_.%/-]+?\.(?:jpg|jpeg|png)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class OreReferencePage:
    ore_type: str
    page_url: str


@dataclass(frozen=True)
class OreReferenceImage:
    ore_type: str
    page_url: str
    image_url: str
    local_path: Path
    source_name: str = "wowhead"


@dataclass(frozen=True)
class DirectOreReference:
    ore_type: str
    page_url: str
    image_url: str
    source_name: str


DEFAULT_ORE_REFERENCE_PAGES = [
    OreReferencePage("Copper Vein", "https://www.wowhead.com/wotlk/object=1731/copper-vein"),
    OreReferencePage("Tin Vein", "https://www.wowhead.com/wotlk/object=1732/tin-vein"),
    OreReferencePage("Silver Vein", "https://www.wowhead.com/wotlk/object=1733/silver-vein"),
    OreReferencePage("Gold Vein", "https://www.wowhead.com/wotlk/object=1734/gold-vein"),
    OreReferencePage("Iron Deposit", "https://www.wowhead.com/wotlk/object=1735/iron-deposit"),
    OreReferencePage("Mithril Deposit", "https://www.wowhead.com/wotlk/object=2040/mithril-deposit"),
    OreReferencePage("Truesilver Deposit", "https://www.wowhead.com/wotlk/object=2047/truesilver-deposit"),
    OreReferencePage("Small Thorium Vein", "https://www.wowhead.com/wotlk/object=324/small-thorium-vein"),
    OreReferencePage("Rich Thorium Vein", "https://www.wowhead.com/wotlk/object=175404/rich-thorium-vein"),
    OreReferencePage("Dark Iron Deposit", "https://www.wowhead.com/wotlk/object=165658/dark-iron-deposit"),
    OreReferencePage("Fel Iron Deposit", "https://www.wowhead.com/wotlk/object=181555/fel-iron-deposit"),
    OreReferencePage("Adamantite Deposit", "https://www.wowhead.com/wotlk/object=181556/adamantite-deposit"),
    OreReferencePage("Khorium Vein", "https://www.wowhead.com/wotlk/object=181557/khorium-vein"),
    OreReferencePage("Cobalt Deposit", "https://www.wowhead.com/wotlk/object=189978/cobalt-deposit"),
    OreReferencePage("Rich Cobalt Deposit", "https://www.wowhead.com/wotlk/object=189979/rich-cobalt-deposit"),
    OreReferencePage("Saronite Deposit", "https://www.wowhead.com/wotlk/object=189980/saronite-deposit"),
    OreReferencePage("Rich Saronite Deposit", "https://www.wowhead.com/wotlk/object=189981/rich-saronite-deposit"),
    OreReferencePage("Titanium Vein", "https://www.wowhead.com/wotlk/object=191133/titanium-vein"),
]

CORE_AZEROTH_ORE_REFERENCE_PAGES = [
    *DEFAULT_ORE_REFERENCE_PAGES[:10],
    OreReferencePage("Copper Vein", "https://www.wowhead.com/classic/object=1731/copper-vein"),
    OreReferencePage("Tin Vein", "https://www.wowhead.com/classic/object=1732/tin-vein"),
    OreReferencePage("Silver Vein", "https://www.wowhead.com/classic/object=1733/silver-vein"),
    OreReferencePage("Gold Vein", "https://www.wowhead.com/classic/object=1734/gold-vein"),
    OreReferencePage("Iron Deposit", "https://www.wowhead.com/classic/object=1735/iron-deposit"),
    OreReferencePage("Mithril Deposit", "https://www.wowhead.com/classic/object=2040/mithril-deposit"),
    OreReferencePage("Truesilver Deposit", "https://www.wowhead.com/classic/object=2047/truesilver-deposit"),
    OreReferencePage("Small Thorium Vein", "https://www.wowhead.com/classic/object=324/small-thorium-vein"),
    OreReferencePage("Rich Thorium Vein", "https://www.wowhead.com/classic/object=175404/rich-thorium-vein"),
    OreReferencePage("Dark Iron Deposit", "https://www.wowhead.com/classic/object=165658/dark-iron-deposit"),
]

WARCRAFT_TAVERN_CORE_ORE_REFERENCES = [
    DirectOreReference(
        "Copper Vein",
        "https://www.warcrafttavern.com/wow-classic/guides/mining-1-300/",
        "https://www.warcrafttavern.com/wp-content/uploads/2022/03/WoWScrnShot_022622_180836-1.jpg",
        "warcraft_tavern",
    ),
    DirectOreReference(
        "Tin Vein",
        "https://www.warcrafttavern.com/wow-classic/guides/mining-1-300/",
        "https://www.warcrafttavern.com/wp-content/uploads/2022/03/TIN-ORE-1.jpg",
        "warcraft_tavern",
    ),
    DirectOreReference(
        "Silver Vein",
        "https://www.warcrafttavern.com/wow-classic/guides/mining-1-300/",
        "https://www.warcrafttavern.com/wp-content/uploads/2022/03/WoWScrnShot_022822_015927-1.jpg",
        "warcraft_tavern",
    ),
    DirectOreReference(
        "Iron Deposit",
        "https://www.warcrafttavern.com/wow-classic/guides/mining-1-300/",
        "https://www.warcrafttavern.com/wp-content/uploads/2022/03/IRON-ORE-1.jpg",
        "warcraft_tavern",
    ),
    DirectOreReference(
        "Gold Vein",
        "https://www.warcrafttavern.com/wow-classic/guides/mining-1-300/",
        "https://www.warcrafttavern.com/wp-content/uploads/2022/03/gold-ore.jpg",
        "warcraft_tavern",
    ),
    DirectOreReference(
        "Mithril Deposit",
        "https://www.warcrafttavern.com/wow-classic/guides/mining-1-300/",
        "https://www.warcrafttavern.com/wp-content/uploads/2022/03/Mithrill-Ore.jpg",
        "warcraft_tavern",
    ),
    DirectOreReference(
        "Truesilver Deposit",
        "https://www.warcrafttavern.com/wow-classic/guides/mining-1-300/",
        "https://www.warcrafttavern.com/wp-content/uploads/2022/03/Truesilver-Ore.jpg",
        "warcraft_tavern",
    ),
    DirectOreReference(
        "Dark Iron Deposit",
        "https://www.warcrafttavern.com/wow-classic/guides/mining-1-300/",
        "https://www.warcrafttavern.com/wp-content/uploads/2022/03/Dark-Iron-Ore.jpg",
        "warcraft_tavern",
    ),
    DirectOreReference(
        "Thorium Vein",
        "https://www.warcrafttavern.com/wow-classic/guides/mining-1-300/",
        "https://www.warcrafttavern.com/wp-content/uploads/2022/03/Thorium-Ore.jpg",
        "warcraft_tavern",
    ),
]

COMMUNITY_CORE_ORE_REFERENCES = [
    DirectOreReference(
        "Small Thorium Vein",
        "https://www.wowhead.com/wotlk/object=324/small-thorium-vein",
        "https://wow.zamimg.com/uploads/screenshots/normal/275627-small-thorium-vein-blasted-lands.jpg",
        "wowhead_search",
    ),
    DirectOreReference(
        "Small Thorium Vein",
        "https://www.wowhead.com/wotlk/object=324/small-thorium-vein",
        "https://wow.zamimg.com/uploads/screenshots/normal/364800-small-thorium-vein.jpg",
        "wowhead_search",
    ),
    DirectOreReference(
        "Copper Vein",
        "https://www.wowhead.com/wotlk/object=1731/copper-vein",
        "https://wow.zamimg.com/uploads/screenshots/normal/461735-copper-ore.jpg",
        "wowhead_search",
    ),
    DirectOreReference(
        "Copper Vein",
        "https://frostmourne.stormforge.gg/en/news/wrath-of-the-lich-king-rebuffed",
        "https://frostmourne.stormforge.gg/cdn-cgi/image/width=3840,format=webp,quality=100/https:/content-manager-fm.stormforge.gg/uploads/mining_xp_crafting_20094fa940.webp",
        "stormforge",
    ),
    DirectOreReference(
        "Azeroth Ore Vein",
        "https://www.mmopixel.com/news/wow-classic-season-of-discovery-guide-to-best-mining-farming-routes",
        "https://files.mmopixel.com/cdn-cgi/image/format=auto/userArticleInfo/3c97a814-ab96-40ff-9d54-8d2ea6d1d4cd.jpeg",
        "mmopixel",
    ),
    DirectOreReference(
        "Copper Vein",
        "https://www.gamespot.com/articles/world-of-warcraft-walkthrough/1100-6114179/",
        "https://www.gamespot.com/a/uploads/original/gamespot/images/2004/reviews/624784-534914_20041130_050.jpg",
        "gamespot",
    ),
    DirectOreReference(
        "Ooze Covered Gold Vein",
        "https://forum.nostalrius.org/viewtopic.php?f=5&t=2459",
        "https://i.imgur.com/GsGYJFi.jpg",
        "nostalrius",
    ),
    DirectOreReference(
        "Azeroth Ore Vein",
        "https://boosting-ground.com/wow-classic/news/10-leveling-mistakes",
        "https://boosting-ground.com/uploads/images/news_body/wow-10-leveling-mistakes-mining.jpg",
        "boosting_ground",
    ),
    DirectOreReference(
        "Azeroth Ore Vein",
        "https://www.gamemeca.com/view.php?gid=36055",
        "https://cdn.gamemeca.com/gmdata/0000/036/055/goldmine-5060911.jpg",
        "gamemeca",
    ),
    DirectOreReference(
        "Azeroth Ore Vein",
        "https://gamewave.fr/world-of-warcraft-classic/wow-classic-la-mine-comment-monter-ce-metier/",
        "https://gamewave.fr/static/images/medias/upload/Amandine%20-%20Oct-Nov-2024/guide-mine-wow-classic.jpg",
        "gamewave",
    ),
    DirectOreReference(
        "Rich Thorium Vein",
        "https://gamewave.fr/world-of-warcraft-classic/wow-classic-la-mine-comment-monter-ce-metier/",
        "https://gamewave.fr/static/images/medias/upload/Amandine%20-%20Oct-Nov-2024/riches-filon-thorium.jpg",
        "gamewave",
    ),
    DirectOreReference(
        "Copper Vein",
        "https://ol.3dmgame.com/gl/203115.html",
        "https://olimg.3dmgame.com/uploads/images/raiders/2022/0923/1663897333343.jpg",
        "3dmgame",
    ),
]


def collect_ore_reference_images(
    output_dir: str | Path,
    *,
    pages: Iterable[OreReferencePage] | None = None,
    direct_images: Iterable[DirectOreReference] | None = None,
    limit_per_page: int = 3,
    timeout_seconds: float = 20.0,
) -> list[OreReferenceImage]:
    root = Path(output_dir)
    image_dir = root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    page_items = list(DEFAULT_ORE_REFERENCE_PAGES if pages is None else pages)
    direct_items = list(direct_images or [])
    collected: list[OreReferenceImage] = []
    seen_image_urls: set[str] = set()
    manifest_path = root / "manifest.jsonl"
    errors_path = root / "errors.jsonl"
    if errors_path.exists():
        errors_path.unlink()

    for page in page_items:
        try:
            html = _fetch_text(page.page_url, timeout_seconds=timeout_seconds)
        except RuntimeError as exc:
            _append_error(errors_path, page.ore_type, page.page_url, None, str(exc))
            continue
        urls = extract_wowhead_screenshot_urls(html)
        for image_url in urls[: max(1, limit_per_page)]:
            if image_url in seen_image_urls:
                continue
            local_path = image_dir / _reference_filename(page.ore_type, image_url)
            if not local_path.exists():
                try:
                    data = _fetch_bytes(image_url, timeout_seconds=timeout_seconds)
                except RuntimeError as exc:
                    _append_error(errors_path, page.ore_type, page.page_url, image_url, str(exc))
                    continue
                local_path.write_bytes(data)
            item = OreReferenceImage(page.ore_type, page.page_url, image_url, local_path)
            if _is_readable_image(local_path):
                collected.append(item)
                seen_image_urls.add(image_url)

    for direct in direct_items:
        if direct.image_url in seen_image_urls:
            continue
        local_path = image_dir / _reference_filename(direct.ore_type, direct.image_url)
        if not local_path.exists():
            try:
                local_path.write_bytes(
                    _fetch_bytes(direct.image_url, timeout_seconds=timeout_seconds)
                )
            except RuntimeError as exc:
                _append_error(
                    errors_path,
                    direct.ore_type,
                    direct.page_url,
                    direct.image_url,
                    str(exc),
                )
                continue
        item = OreReferenceImage(
            direct.ore_type,
            direct.page_url,
            direct.image_url,
            local_path,
            direct.source_name,
        )
        if _is_readable_image(local_path):
            collected.append(item)
            seen_image_urls.add(direct.image_url)

    _write_manifest(manifest_path, collected)
    return collected


def collect_core_azeroth_ore_reference_images(
    output_dir: str | Path,
    *,
    limit_per_page: int = 3,
    timeout_seconds: float = 20.0,
) -> list[OreReferenceImage]:
    return collect_ore_reference_images(
        output_dir,
        pages=CORE_AZEROTH_ORE_REFERENCE_PAGES,
        direct_images=[
            *WARCRAFT_TAVERN_CORE_ORE_REFERENCES,
            *COMMUNITY_CORE_ORE_REFERENCES,
        ],
        limit_per_page=limit_per_page,
        timeout_seconds=timeout_seconds,
    )


def extract_wowhead_screenshot_urls(html: str) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    for match in WOWHEAD_SCREENSHOT_RE.findall(html):
        url = "https:" + match if match.startswith("//") else match
        if url not in seen:
            urls.append(url)
            seen.add(url)
    for match in WOWHEAD_RELATIVE_SCREENSHOT_RE.findall(html):
        url = urllib.parse.urljoin("https://wow.zamimg.com/", match)
        if url not in seen:
            urls.append(url)
            seen.add(url)
    return urls


def draw_ore_reference_contact_sheet(
    references: Iterable[OreReferenceImage],
    output_path: str | Path,
    *,
    columns: int = 4,
    tile_width: int = 360,
    tile_height: int = 250,
) -> Path:
    items = list(references)
    if not items:
        raise ValueError("No ore reference images to draw")
    columns = max(1, columns)
    rows = int(np.ceil(len(items) / float(columns)))
    sheet = np.full((rows * tile_height, columns * tile_width, 3), 30, dtype=np.uint8)

    for index, item in enumerate(items):
        image = cv2.imread(str(item.local_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        row = index // columns
        col = index % columns
        x = col * tile_width
        y = row * tile_height
        thumb = _fit_image(image, tile_width, tile_height - 46)
        thumb_y = y + 34
        thumb_x = x + (tile_width - thumb.shape[1]) // 2
        sheet[thumb_y : thumb_y + thumb.shape[0], thumb_x : thumb_x + thumb.shape[1]] = thumb
        _draw_text(sheet, item.ore_type, x + 10, y + 23, color=(235, 235, 235), scale=0.58)
        _draw_text(
            sheet,
            Path(item.local_path).name[:42],
            x + 10,
            y + tile_height - 12,
            color=(175, 175, 175),
            scale=0.42,
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), sheet)
    return output


def _fetch_text(url: str, *, timeout_seconds: float) -> str:
    data = _fetch_bytes(url, timeout_seconds=timeout_seconds)
    return data.decode("utf-8", errors="replace")


def _fetch_bytes(url: str, *, timeout_seconds: float) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 vision-bot ore reference collector",
            "Accept": "text/html,image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return response.read()
    except (TimeoutError, OSError, urllib.error.URLError) as exc:
        raise RuntimeError(f"Failed to fetch {url}: {exc}") from exc


def _reference_filename(ore_type: str, image_url: str) -> str:
    parsed = urllib.parse.urlparse(image_url)
    suffix = Path(parsed.path).suffix.lower() or ".jpg"
    digest = hashlib.sha1(image_url.encode("utf-8")).hexdigest()[:10]
    slug = re.sub(r"[^a-z0-9]+", "_", ore_type.lower()).strip("_")
    return f"{slug}_{digest}{suffix}"


def _is_readable_image(path: Path) -> bool:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return image is not None and image.size > 0


def _write_manifest(path: Path, references: list[OreReferenceImage]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in references:
            image = cv2.imread(str(item.local_path), cv2.IMREAD_COLOR)
            height, width = image.shape[:2] if image is not None else (None, None)
            handle.write(
                json.dumps(
                    {
                        "ore_type": item.ore_type,
                        "source_name": item.source_name,
                        "page_url": item.page_url,
                        "image_url": item.image_url,
                        "local_path": item.local_path.as_posix(),
                        "sha256": hashlib.sha256(item.local_path.read_bytes()).hexdigest(),
                        "width": width,
                        "height": height,
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )


def _append_error(path: Path, ore_type: str, page_url: str, image_url: str | None, error: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "ore_type": ore_type,
                    "page_url": page_url,
                    "image_url": image_url,
                    "error": error,
                },
                ensure_ascii=True,
            )
            + "\n"
        )


def _fit_image(image: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(max_width / float(width), max_height / float(height))
    target_width = max(1, int(round(width * scale)))
    target_height = max(1, int(round(height * scale)))
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)


def _draw_text(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    *,
    color: tuple[int, int, int],
    scale: float,
) -> None:
    for line_index, line in enumerate(textwrap.wrap(text, width=36)[:2]):
        cv2.putText(
            image,
            line,
            (x, y + line_index * 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            1,
            cv2.LINE_AA,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Download visible ore vein reference screenshots")
    parser.add_argument("--output-dir", default="data/ore_references")
    parser.add_argument("--limit-per-page", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--no-contact-sheet", action="store_true")
    parser.add_argument("--core-azeroth", action="store_true")
    args = parser.parse_args()

    collector = (
        collect_core_azeroth_ore_reference_images
        if args.core_azeroth
        else collect_ore_reference_images
    )
    references = collector(
        args.output_dir,
        limit_per_page=args.limit_per_page,
        timeout_seconds=args.timeout,
    )
    print(f"Ore references: {len(references)}")
    print(f"Manifest: {Path(args.output_dir) / 'manifest.jsonl'}")
    if references and not args.no_contact_sheet:
        sheet_path = draw_ore_reference_contact_sheet(
            references,
            Path(args.output_dir) / "ore_reference_contact_sheet.jpg",
        )
        print(f"Contact sheet: {sheet_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
