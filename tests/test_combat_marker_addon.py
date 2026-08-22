from pathlib import Path


ADDON_DIR = Path(__file__).parents[1] / "addons" / "ScreenVisionTelemetry"
ADDON_SOURCE = ADDON_DIR / "ScreenVisionTelemetry.lua"
ADDON_MANIFEST = ADDON_DIR / "ScreenVisionTelemetry.toc"


def test_addon_tracks_real_attackers_and_exposes_secure_cycle_binding():
    source = ADDON_SOURCE.read_text(encoding="utf-8")

    assert 'destGUID == playerGUID' in source
    assert 'activeAttackers[sourceGUID]' in source
    assert 'UnitGUID("target")' in source
    assert 'SetOverrideBindingClick(targetCycleOwner, true, "F8"' in source
    assert '"/targetenemy"' in source


def test_addon_exposes_capped_visible_unique_attacker_count():
    source = ADDON_SOURCE.read_text(encoding="utf-8")

    assert '"ScreenVisionAttackerCountMarkerFrame"' in source
    assert "for _guid, deadline in pairs(activeAttackers) do" in source
    assert "attackerCountMarker.markerTexture:SetTexture(1, 0.55, 0, 1)" in source
    assert 'command == "threat3"' in source
    assert "## Version: 0.4.5" in ADDON_MANIFEST.read_text(
        encoding="utf-8"
    )


def test_addon_keeps_hit_telemetry_but_does_not_hide_new_ui_errors():
    source = ADDON_SOURCE.read_text(encoding="utf-8")

    assert "local HIT_VISIBLE_SECONDS = 2.00" in source
    assert "if HitMarkerIsActive() then" not in source
    assert "## Version: 0.4.5" in ADDON_MANIFEST.read_text(encoding="utf-8")


def test_addon_exposes_player_heading_and_camera_reset_binding():
    source = ADDON_SOURCE.read_text(encoding="utf-8")

    assert "GetPlayerFacing" in source
    assert 'SetFormattedText("H %.1f", degrees)' in source
    assert 'SetOverrideBinding(targetCycleOwner, true, "F7", "SETVIEW5")' in source
    assert 'SLASH_SCREENVISIONCAMERA1 = "/svacamera"' in source


def test_addon_exposes_visible_coordinate_and_heading_telemetry_strip():
    source = ADDON_SOURCE.read_text(encoding="utf-8")

    assert '"ScreenVisionRuntimeTelemetryFrame"' in source
    assert 'GetPlayerMapPosition, "player"' in source
    assert "UpdateRuntimeTelemetry()" in source
    assert "for bitIndex = 0, 13 do" in source


def test_addon_relays_only_native_minimap_ore_tooltips_as_visible_telemetry():
    source = ADDON_SOURCE.read_text(encoding="utf-8")

    assert '"ScreenVisionMinimapOreTooltipTelemetryFrame"' in source
    assert '"ScreenVisionMinimapOreTooltipTextFrame"' in source
    assert "Minimap:IsMouseOver()" in source
    assert "GameTooltipTextLeft1" in source
    assert 'string.lower("Small Thorium Vein")' in source
    assert 'oreTooltipText:SetText("MINIMAP ORE: " .. tooltipName)' in source
    assert "SetOreTooltipBit(9, true, 0, 1, 0)" in source


def test_addon_marks_validated_underground_tanaris_subzones():
    source = ADDON_SOURCE.read_text(encoding="utf-8")

    assert 'GetSubZoneText()' in source
    assert 'subzone == "The Gaping Chasm"' in source
    assert 'subzone == "The Noxious Lair"' in source
    assert 'forbiddenSubzoneMarker:Show()' in source


def test_addon_marks_a_selected_dead_hostile_for_bounded_loot_flow():
    source = ADDON_SOURCE.read_text(encoding="utf-8")

    assert 'UnitIsDead("target")' in source
    assert 'UnitCanAttack("player", "target")' in source
    assert 'deadHostileTargetMarker:Show()' in source


def test_addon_tracks_target_switches_for_confirmed_in_combat_loot():
    source = ADDON_SOURCE.read_text(encoding="utf-8")

    assert 'eventType == "UNIT_DIED"' in source
    assert 'event == "PLAYER_TARGET_CHANGED"' in source
    assert 'pendingCombatLootGUID = destGUID' in source
    assert 'SetOverrideBindingClick(targetCycleOwner, true, "F9"' in source
    assert 'targetLastButton:SetAttribute("macrotext", "/targetlasttarget")' in source
    assert 'events:RegisterEvent("LOOT_OPENED")' in source
    assert 'UnitReaction("player", "target")' in source
    assert 'FlashMarker(lootOpenedMarker, LOOT_VISIBLE_SECONDS)' in source


def test_addon_exposes_exact_spirit_healer_state_and_follow_binding():
    source = ADDON_SOURCE.read_text(encoding="utf-8")

    assert 'string.lower(targetName or "") == "spirit healer"' in source
    assert 'GossipFrame:IsShown()' in source
    assert 'spiritHealerTargetMarker:Show()' in source
    assert 'spiritHealerDialogMarker:Show()' in source
    assert 'followTargetButton:SetAttribute("macrotext", "/follow target")' in source
    assert 'SetOverrideBindingClick(targetCycleOwner, true, "F6"' in source
