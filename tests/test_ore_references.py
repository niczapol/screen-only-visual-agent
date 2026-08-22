import cv2
import numpy as np

from vision_bot.ore_references import (
    COMMUNITY_CORE_ORE_REFERENCES,
    DirectOreReference,
    OreReferenceImage,
    OreReferencePage,
    collect_ore_reference_images,
    draw_ore_reference_contact_sheet,
    extract_wowhead_screenshot_urls,
)


def test_community_reference_sources_are_curated_full_screenshots():
    assert len(COMMUNITY_CORE_ORE_REFERENCES) == 12
    assert {item.source_name for item in COMMUNITY_CORE_ORE_REFERENCES} == {
        "3dmgame",
        "boosting_ground",
        "gamespot",
        "gamemeca",
        "gamewave",
        "mmopixel",
        "nostalrius",
        "stormforge",
        "wowhead_search",
    }


def test_extract_wowhead_screenshot_urls_deduplicates_absolute_and_relative_urls():
    html = """
    <meta property="og:image" content="https://wow.zamimg.com/uploads/screenshots/normal/1214536-copper-vein.jpg">
    <img src="//wow.zamimg.com/uploads/screenshots/normal/1214536-copper-vein.jpg">
    <script>var s='uploads/screenshots/normal/999999-iron-deposit.jpg';</script>
    """

    urls = extract_wowhead_screenshot_urls(html)

    assert urls == [
        "https://wow.zamimg.com/uploads/screenshots/normal/1214536-copper-vein.jpg",
        "https://wow.zamimg.com/uploads/screenshots/normal/999999-iron-deposit.jpg",
    ]


def test_draw_ore_reference_contact_sheet_writes_image(tmp_path):
    image_path = tmp_path / "ore.jpg"
    cv2.imwrite(str(image_path), np.full((80, 120, 3), 120, dtype=np.uint8))
    output_path = tmp_path / "sheet.jpg"

    sheet = draw_ore_reference_contact_sheet(
        [
            OreReferenceImage(
                "Copper Vein",
                "https://example.test/page",
                "https://example.test/image.jpg",
                image_path,
            )
        ],
        output_path,
        columns=1,
        tile_width=160,
        tile_height=120,
    )

    assert sheet == output_path
    assert cv2.imread(str(output_path), cv2.IMREAD_COLOR) is not None


def test_collect_ore_references_combines_pages_and_direct_images(monkeypatch, tmp_path):
    image = np.full((32, 48, 3), 140, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok

    monkeypatch.setattr(
        "vision_bot.ore_references._fetch_text",
        lambda *_args, **_kwargs: (
            '<img src="https://wow.zamimg.com/uploads/screenshots/normal/1-copper.jpg">'
        ),
    )
    monkeypatch.setattr(
        "vision_bot.ore_references._fetch_bytes",
        lambda *_args, **_kwargs: encoded.tobytes(),
    )

    references = collect_ore_reference_images(
        tmp_path,
        pages=[OreReferencePage("Copper Vein", "https://example.test/copper")],
        direct_images=[
            DirectOreReference(
                "Tin Vein",
                "https://example.test/tin",
                "https://example.test/tin.jpg",
                "guide",
            )
        ],
    )

    assert [item.ore_type for item in references] == ["Copper Vein", "Tin Vein"]
    manifest = (tmp_path / "manifest.jsonl").read_text(encoding="utf-8")
    assert '"source_name": "guide"' in manifest
    assert '"width": 48' in manifest
