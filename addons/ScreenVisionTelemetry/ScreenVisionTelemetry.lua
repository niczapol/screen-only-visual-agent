local function CreateMarker(name, xOffset, red, green, blue)
    local frame = CreateFrame("Frame", name, UIParent)
    frame:SetWidth(56)
    frame:SetHeight(24)
    frame:SetPoint("TOP", UIParent, "TOP", xOffset, -8)
    frame:SetFrameStrata("TOOLTIP")

    local texture = frame:CreateTexture(nil, "OVERLAY")
    texture:SetAllPoints(frame)
    texture:SetTexture(red, green, blue, 1)
    frame.markerTexture = texture
    frame:Hide()
    return frame
end

local combatMarker = CreateMarker("ScreenVisionCombatMarkerFrame", 0, 1, 0, 1)
local hitMarker = CreateMarker("ScreenVisionCombatHitMarkerFrame", -72, 0, 1, 0)
local attackerTargetMarker = CreateMarker("ScreenVisionAttackerTargetMarkerFrame", -144, 0, 0, 1)
local facingMarker = CreateMarker("ScreenVisionCombatFacingMarkerFrame", 72, 0, 1, 1)
local rangeMarker = CreateMarker("ScreenVisionCombatRangeMarkerFrame", 144, 1, 0.45, 0)
local mountedMarker = CreateMarker("ScreenVisionMountedMarkerFrame", 216, 0.05, 0.45, 1)
local forbiddenSubzoneMarker = CreateMarker("ScreenVisionForbiddenSubzoneMarkerFrame", 288, 1, 0.85, 0)
local deadHostileTargetMarker = CreateMarker("ScreenVisionDeadHostileTargetMarkerFrame", 360, 0.30, 1, 0.10)
local spiritHealerTargetMarker = CreateMarker("ScreenVisionSpiritHealerTargetMarkerFrame", 432, 1, 0.15, 0.55)
local spiritHealerDialogMarker = CreateMarker("ScreenVisionSpiritHealerDialogMarkerFrame", 504, 0.35, 1, 0.65)
local lootOpenedMarker = CreateMarker("ScreenVisionLootOpenedMarkerFrame", 576, 1, 1, 1)
local combatLootPendingMarker = CreateMarker("ScreenVisionCombatLootPendingMarkerFrame", -216, 0.65, 0.20, 1)
-- Capped 1/2/3+ count of unique combat-log sources that have attacked the
-- player recently.  It remains an ordinary visible screen marker; the
-- external controller never reads the Lua table directly.
local attackerCountMarker = CreateMarker("ScreenVisionAttackerCountMarkerFrame", -288, 1, 0, 0)

local targetCycleOwner = CreateFrame("Frame", "ScreenVisionTargetCycleBindingOwner", UIParent)
local targetCycleButton = CreateFrame(
    "Button",
    "ScreenVisionTargetCycleButton",
    UIParent,
    "SecureActionButtonTemplate"
)
targetCycleButton:SetAttribute("type", "macro")
targetCycleButton:SetAttribute("macrotext", "/targetenemy")

local targetLastButton = CreateFrame(
    "Button",
    "ScreenVisionTargetLastButton",
    UIParent,
    "SecureActionButtonTemplate"
)
targetLastButton:SetAttribute("type", "macro")
targetLastButton:SetAttribute("macrotext", "/targetlasttarget")

local followTargetButton = CreateFrame(
    "Button",
    "ScreenVisionFollowTargetButton",
    UIParent,
    "SecureActionButtonTemplate"
)
followTargetButton:SetAttribute("type", "macro")
followTargetButton:SetAttribute("macrotext", "/follow target")

local function BindControlKeys()
    if type(SetOverrideBindingClick) ~= "function" then
        return
    end
    if type(InCombatLockdown) == "function" and InCombatLockdown() then
        return
    end
    ClearOverrideBindings(targetCycleOwner)
    SetOverrideBindingClick(targetCycleOwner, true, "F8", targetCycleButton:GetName())
    SetOverrideBindingClick(targetCycleOwner, true, "F9", targetLastButton:GetName())
    SetOverrideBindingClick(targetCycleOwner, true, "F6", followTargetButton:GetName())
    if type(SetOverrideBinding) == "function" then
        SetOverrideBinding(targetCycleOwner, true, "F7", "SETVIEW5")
    end
end

local coordinateFrame = CreateFrame("Frame", "ScreenVisionZoneCoordinateFrame", UIParent)
coordinateFrame:SetWidth(180)
coordinateFrame:SetHeight(40)
coordinateFrame:SetPoint("TOPRIGHT", UIParent, "TOPRIGHT", -30, -280)
coordinateFrame:SetFrameStrata("TOOLTIP")
coordinateFrame:SetBackdrop({
    bgFile = "Interface\\ChatFrame\\ChatFrameBackground",
    edgeFile = "Interface\\Tooltips\\UI-Tooltip-Border",
    edgeSize = 12,
    insets = {left = 3, right = 3, top = 3, bottom = 3},
})
coordinateFrame:SetBackdropColor(0.02, 0.02, 0.02, 0.96)
coordinateFrame:SetBackdropBorderColor(0.72, 0.55, 0.02, 1)

local coordinateText = coordinateFrame:CreateFontString(nil, "OVERLAY", "GameFontNormalLarge")
coordinateText:SetPoint("CENTER", coordinateFrame, "CENTER", 0, 0)
coordinateText:SetTextColor(1, 0.82, 0, 1)
coordinateText:SetText("--.--, --.--")

local armorMarker = CreateFrame("Frame", "ScreenVisionArmorCriticalMarkerFrame", UIParent)
armorMarker:SetWidth(180)
armorMarker:SetHeight(12)
armorMarker:SetPoint("TOPRIGHT", UIParent, "TOPRIGHT", -30, -326)
armorMarker:SetFrameStrata("TOOLTIP")
local armorTexture = armorMarker:CreateTexture(nil, "OVERLAY")
armorTexture:SetAllPoints(armorMarker)
armorTexture:SetTexture(1, 0, 0, 1)
armorMarker:Hide()

local headingFrame = CreateFrame("Frame", "ScreenVisionPlayerHeadingFrame", UIParent)
headingFrame:SetWidth(180)
headingFrame:SetHeight(28)
headingFrame:SetPoint("TOPRIGHT", UIParent, "TOPRIGHT", -30, -344)
headingFrame:SetFrameStrata("TOOLTIP")
headingFrame:SetBackdrop({
    bgFile = "Interface\\ChatFrame\\ChatFrameBackground",
    edgeFile = "Interface\\Tooltips\\UI-Tooltip-Border",
    edgeSize = 10,
    insets = {left = 3, right = 3, top = 3, bottom = 3},
})
headingFrame:SetBackdropColor(0.02, 0.02, 0.02, 0.96)
headingFrame:SetBackdropBorderColor(0.15, 0.55, 0.75, 1)

local headingText = headingFrame:CreateFontString(nil, "OVERLAY", "GameFontNormal")
headingText:SetPoint("CENTER", headingFrame, "CENTER", 0, 0)
headingText:SetTextColor(0.25, 0.80, 1.00, 1)
headingText:SetText("H --.-")

-- A fixed, visible binary strip avoids blocking OCR in the external controller.
-- It still crosses the same screen-only boundary as the text coordinates.
local telemetryFrame = CreateFrame("Frame", "ScreenVisionRuntimeTelemetryFrame", UIParent)
telemetryFrame:SetWidth(180)
telemetryFrame:SetHeight(16)
telemetryFrame:SetPoint("TOP", UIParent, "TOP", 0, -42)
telemetryFrame:SetFrameStrata("TOOLTIP")
telemetryFrame:SetBackdrop({bgFile = "Interface\\ChatFrame\\ChatFrameBackground"})
telemetryFrame:SetBackdropColor(0, 0, 0, 1)

local function CreateTelemetryTexture(xOffset, width, red, green, blue)
    local texture = telemetryFrame:CreateTexture(nil, "OVERLAY")
    texture:SetWidth(width)
    texture:SetHeight(12)
    texture:SetPoint("LEFT", telemetryFrame, "LEFT", xOffset, 0)
    texture:SetTexture(red, green, blue, 1)
    return texture
end

local telemetryLeftSentinel = CreateTelemetryTexture(4, 6, 1, 0, 1)
local telemetryRightSentinel = CreateTelemetryTexture(170, 6, 0, 1, 1)
local telemetryBits = {}
for index = 1, 37 do
    telemetryBits[index] = CreateTelemetryTexture(14 + (index - 1) * 4, 4, 0.04, 0.04, 0.04)
end
telemetryFrame:Hide()

-- The external controller deliberately hovers a CV-detected native minimap
-- blip.  This strip relays only a real GameTooltip owned by the minimap; it
-- does not expose GatherMate's historical database or hidden game state.
local oreTooltipFrame = CreateFrame("Frame", "ScreenVisionMinimapOreTooltipTelemetryFrame", UIParent)
oreTooltipFrame:SetWidth(66)
oreTooltipFrame:SetHeight(16)
oreTooltipFrame:SetPoint("TOP", UIParent, "TOP", 0, -72)
oreTooltipFrame:SetFrameStrata("TOOLTIP")
oreTooltipFrame:SetBackdrop({bgFile = "Interface\\ChatFrame\\ChatFrameBackground"})
oreTooltipFrame:SetBackdropColor(0, 0, 0, 1)

local function CreateOreTooltipTexture(xOffset, width, red, green, blue)
    local texture = oreTooltipFrame:CreateTexture(nil, "OVERLAY")
    texture:SetWidth(width)
    texture:SetHeight(12)
    texture:SetPoint("LEFT", oreTooltipFrame, "LEFT", xOffset, 0)
    texture:SetTexture(red, green, blue, 1)
    return texture
end

local oreTooltipLeftSentinel = CreateOreTooltipTexture(4, 6, 1, 1, 0)
local oreTooltipRightSentinel = CreateOreTooltipTexture(56, 6, 0, 0, 1)
local oreTooltipBits = {}
for index = 1, 9 do
    oreTooltipBits[index] = CreateOreTooltipTexture(14 + (index - 1) * 4, 4, 0.04, 0.04, 0.04)
end
oreTooltipFrame:Hide()

local oreTooltipTextFrame = CreateFrame("Frame", "ScreenVisionMinimapOreTooltipTextFrame", UIParent)
oreTooltipTextFrame:SetWidth(240)
oreTooltipTextFrame:SetHeight(28)
oreTooltipTextFrame:SetPoint("TOPRIGHT", UIParent, "TOPRIGHT", -30, -380)
oreTooltipTextFrame:SetFrameStrata("TOOLTIP")
oreTooltipTextFrame:SetBackdrop({
    bgFile = "Interface\\ChatFrame\\ChatFrameBackground",
    edgeFile = "Interface\\Tooltips\\UI-Tooltip-Border",
    edgeSize = 10,
    insets = {left = 3, right = 3, top = 3, bottom = 3},
})
oreTooltipTextFrame:SetBackdropColor(0.02, 0.02, 0.02, 0.96)
oreTooltipTextFrame:SetBackdropBorderColor(1, 0.82, 0, 1)
local oreTooltipText = oreTooltipTextFrame:CreateFontString(nil, "OVERLAY", "GameFontNormal")
oreTooltipText:SetPoint("CENTER", oreTooltipTextFrame, "CENTER", 0, 0)
oreTooltipText:SetTextColor(1, 0.82, 0, 1)
oreTooltipTextFrame:Hide()

local MINIMAP_ORE_BY_TOOLTIP = {
    [string.lower("Copper Vein")] = {1, "Copper"},
    [string.lower("Tin Vein")] = {2, "Tin"},
    [string.lower("Silver Vein")] = {3, "Silver"},
    [string.lower("Iron Deposit")] = {4, "Iron"},
    [string.lower("Gold Vein")] = {5, "Gold"},
    [string.lower("Mithril Deposit")] = {6, "Mithril"},
    [string.lower("Ooze Covered Mithril Deposit")] = {6, "Mithril"},
    [string.lower("Truesilver Deposit")] = {7, "Truesilver"},
    [string.lower("Ooze Covered Truesilver Deposit")] = {7, "Truesilver"},
    [string.lower("Small Thorium Vein")] = {8, "Small Thorium"},
    [string.lower("Ooze Covered Thorium Vein")] = {8, "Small Thorium"},
    [string.lower("Rich Thorium Vein")] = {9, "Rich Thorium"},
    [string.lower("Ooze Covered Rich Thorium Vein")] = {9, "Rich Thorium"},
    [string.lower("Fel Iron Deposit")] = {10, "Fel Iron"},
    [string.lower("Adamantite Deposit")] = {11, "Adamantite"},
    [string.lower("Rich Adamantite Deposit")] = {12, "Rich Adamantite"},
    [string.lower("Khorium Vein")] = {13, "Khorium"},
    [string.lower("Nethercite Deposit")] = {14, "Nethercite"},
    [string.lower("Cobalt Deposit")] = {15, "Cobalt"},
    [string.lower("Rich Cobalt Deposit")] = {16, "Rich Cobalt"},
    [string.lower("Saronite Deposit")] = {17, "Saronite"},
    [string.lower("Rich Saronite Deposit")] = {18, "Rich Saronite"},
    [string.lower("Titanium Vein")] = {19, "Titanium"},
}

local function SetOreTooltipBit(index, enabled, red, green, blue)
    if enabled then
        oreTooltipBits[index]:SetTexture(red, green, blue, 1)
    else
        oreTooltipBits[index]:SetTexture(0.04, 0.04, 0.04, 1)
    end
end

local function HideMinimapOreTooltip()
    oreTooltipFrame:Hide()
    oreTooltipTextFrame:Hide()
end

local function UpdateMinimapOreTooltip()
    if Minimap == nil or not Minimap:IsMouseOver() or GameTooltip == nil or not GameTooltip:IsShown() then
        HideMinimapOreTooltip()
        return
    end
    local firstLine = GameTooltipTextLeft1
    local tooltipName = firstLine ~= nil and firstLine:GetText() or nil
    local ore = tooltipName ~= nil and MINIMAP_ORE_BY_TOOLTIP[string.lower(tooltipName)] or nil
    if ore == nil then
        HideMinimapOreTooltip()
        return
    end

    local oreId = ore[1]
    for bitIndex = 0, 7 do
        SetOreTooltipBit(
            bitIndex + 1,
            math.floor(oreId / (2 ^ bitIndex)) % 2 == 1,
            1,
            0,
            0
        )
    end
    -- Explicit source bit: this frame is never valid for a world tooltip.
    SetOreTooltipBit(9, true, 0, 1, 0)
    oreTooltipText:SetText("MINIMAP ORE: " .. tooltipName)
    oreTooltipFrame:Show()
    oreTooltipTextFrame:Show()
end

local astrolabe = nil
local coordinateElapsed = 0
local runtimeMarkerElapsed = 0
local oreTooltipElapsed = 0
local nextZoneMapSyncAt = 0
local telemetryX = nil
local telemetryY = nil
local telemetryHeading = nil

local function ResolveAstrolabe()
    if astrolabe ~= nil then
        return astrolabe
    end
    if type(DongleStub) ~= "function" then
        return nil
    end
    local ok, library = pcall(DongleStub, "Astrolabe-0.4")
    if ok then
        astrolabe = library
    end
    return astrolabe
end

local function UpdateZoneCoordinates()
    if WorldMapFrame ~= nil and WorldMapFrame:IsShown() then
        telemetryX = nil
        telemetryY = nil
        return
    end

    local now = GetTime()
    if now >= nextZoneMapSyncAt and type(SetMapToCurrentZone) == "function" then
        pcall(SetMapToCurrentZone)
        nextZoneMapSyncAt = now + 0.25
    end

    if type(GetPlayerMapPosition) == "function" then
        local ok, x, y = pcall(GetPlayerMapPosition, "player")
        if ok and x ~= nil and y ~= nil and (x > 0 or y > 0) then
            coordinateText:SetFormattedText("%.2f, %.2f", x * 100, y * 100)
            telemetryX = x
            telemetryY = y
            return
        end
    end

    local library = ResolveAstrolabe()
    if library == nil then
        coordinateText:SetText("--.--, --.--")
        telemetryX = nil
        telemetryY = nil
        return
    end
    local ok, _continent, _zone, x, y = pcall(library.GetCurrentPlayerPosition, library)
    if not ok or x == nil or y == nil or (x <= 0 and y <= 0) then
        coordinateText:SetText("--.--, --.--")
        telemetryX = nil
        telemetryY = nil
        return
    end
    coordinateText:SetFormattedText("%.2f, %.2f", x * 100, y * 100)
    telemetryX = x
    telemetryY = y
end

local function UpdatePlayerHeading()
    if type(GetPlayerFacing) ~= "function" then
        headingText:SetText("H --.-")
        telemetryHeading = nil
        return
    end
    local ok, facing = pcall(GetPlayerFacing)
    if not ok or type(facing) ~= "number" then
        headingText:SetText("H --.-")
        telemetryHeading = nil
        return
    end
    local degrees = (facing * 180 / math.pi) % 360
    headingText:SetFormattedText("H %.1f", degrees)
    telemetryHeading = degrees
end

local function SetTelemetryBit(index, enabled, red, green, blue)
    local texture = telemetryBits[index]
    if enabled then
        texture:SetTexture(red, green, blue, 1)
    else
        texture:SetTexture(0.04, 0.04, 0.04, 1)
    end
end

local function UpdateRuntimeTelemetry()
    if telemetryX == nil or telemetryY == nil or telemetryHeading == nil then
        telemetryFrame:Hide()
        return
    end

    local xValue = math.max(0, math.min(10000, math.floor(telemetryX * 10000 + 0.5)))
    local yValue = math.max(0, math.min(10000, math.floor(telemetryY * 10000 + 0.5)))
    local headingValue = math.floor((telemetryHeading / 360) * 511 + 0.5) % 512

    for bitIndex = 0, 13 do
        SetTelemetryBit(bitIndex + 1, math.floor(xValue / (2 ^ bitIndex)) % 2 == 1, 1, 0, 0)
        SetTelemetryBit(bitIndex + 15, math.floor(yValue / (2 ^ bitIndex)) % 2 == 1, 0, 1, 0)
    end
    for bitIndex = 0, 8 do
        SetTelemetryBit(bitIndex + 29, math.floor(headingValue / (2 ^ bitIndex)) % 2 == 1, 0, 0, 1)
    end
    telemetryFrame:Show()
end

local function UpdateRuntimeMarkers()
    if type(IsMounted) == "function" and IsMounted() then
        mountedMarker:Show()
    else
        mountedMarker:Hide()
    end

    local currentTotal = 0
    local maximumTotal = 0
    for slot = 1, 18 do
        local current, maximum = GetInventoryItemDurability(slot)
        if current ~= nil and maximum ~= nil and maximum > 0 then
            currentTotal = currentTotal + current
            maximumTotal = maximumTotal + maximum
        end
    end
    if maximumTotal > 0 and currentTotal / maximumTotal <= 0.50 then
        armorMarker:Show()
    else
        armorMarker:Hide()
    end

    local subzone = type(GetSubZoneText) == "function" and GetSubZoneText() or ""
    if subzone == "The Gaping Chasm" or subzone == "The Noxious Lair" then
        forbiddenSubzoneMarker:Show()
    else
        forbiddenSubzoneMarker:Hide()
    end

    local targetReaction = UnitExists("target") and UnitReaction("player", "target") or nil
    local deadHostileTarget = UnitExists("target")
        and UnitIsDead("target")
        and (
            UnitCanAttack("player", "target") == 1
            or (targetReaction ~= nil and targetReaction <= 4)
        )
    if deadHostileTarget then
        deadHostileTargetMarker:Show()
    else
        deadHostileTargetMarker:Hide()
    end

    local targetName = UnitExists("target") and UnitName("target") or ""
    local spiritHealerTargeted = string.lower(targetName or "") == "spirit healer"
    if spiritHealerTargeted then
        spiritHealerTargetMarker:Show()
    else
        spiritHealerTargetMarker:Hide()
    end
    if spiritHealerTargeted and GossipFrame ~= nil and GossipFrame:IsShown() then
        spiritHealerDialogMarker:Show()
    else
        spiritHealerDialogMarker:Hide()
    end
end

local forcedState = nil
local forcedAttackerTargetState = nil
local forcedAttackerCountState = nil
local markerDeadlines = {}
local activeAttackers = {}
local currentTargetGUID = nil
local currentTargetHostile = false
local previousTargetGUID = nil
local previousTargetHostile = false
local pendingCombatLootGUID = nil
local pendingCombatLootDeadline = 0

local HIT_VISIBLE_SECONDS = 2.00
local ERROR_VISIBLE_SECONDS = 0.85
local ATTACKER_VISIBLE_SECONDS = 12.00
local LOOT_VISIBLE_SECONDS = 0.65
local COMBAT_LOOT_PENDING_SECONDS = 8.00

local function FlashMarker(frame, duration)
    frame:Show()
    markerDeadlines[frame] = GetTime() + duration
end

local function HideMarker(frame)
    frame:Hide()
    markerDeadlines[frame] = nil
end

local function FlashOutcome(frame, duration)
    HideMarker(hitMarker)
    HideMarker(facingMarker)
    HideMarker(rangeMarker)
    FlashMarker(frame, duration)
end

local function UpdateCombatMarker()
    local active = forcedState
    if active == nil then
        active = UnitAffectingCombat("player")
    end
    if active then
        combatMarker:Show()
    else
        combatMarker:Hide()
    end
end

local function UpdateAttackerTargetMarker()
    local active = forcedAttackerTargetState
    if active == nil then
        local targetGUID = UnitGUID("target")
        local deadline = targetGUID and activeAttackers[targetGUID] or nil
        active = deadline ~= nil and GetTime() < deadline and UnitAffectingCombat("player")
    end
    if active then
        attackerTargetMarker:Show()
    else
        attackerTargetMarker:Hide()
    end
end

local function UpdateAttackerCountMarker()
    local count = forcedAttackerCountState
    if count == nil then
        count = 0
        local now = GetTime()
        for _guid, deadline in pairs(activeAttackers) do
            if now < deadline then
                count = count + 1
            end
        end
        if not UnitAffectingCombat("player") then
            count = 0
        end
    end

    if count <= 0 then
        attackerCountMarker:Hide()
        return
    end
    if count == 1 then
        attackerCountMarker.markerTexture:SetTexture(1, 0, 0, 1)
    elseif count == 2 then
        attackerCountMarker.markerTexture:SetTexture(1, 0.55, 0, 1)
    else
        attackerCountMarker.markerTexture:SetTexture(1, 1, 1, 1)
    end
    attackerCountMarker:Show()
end

local function TrackTargetChange()
    local nextGUID = UnitGUID("target")
    if nextGUID ~= currentTargetGUID then
        previousTargetGUID = currentTargetGUID
        previousTargetHostile = currentTargetHostile
        currentTargetGUID = nextGUID
        currentTargetHostile = nextGUID ~= nil and UnitCanAttack("player", "target") == 1
    end
end

local function UpdateCombatLootPendingMarker()
    if pendingCombatLootGUID ~= nil
        and GetTime() < pendingCombatLootDeadline
        and UnitAffectingCombat("player")
        and UnitGUID("target") ~= pendingCombatLootGUID then
        combatLootPendingMarker:Show()
    else
        combatLootPendingMarker:Hide()
    end
end

local function IsDamageEvent(eventType)
    return eventType == "SWING_DAMAGE"
        or eventType == "RANGE_DAMAGE"
        or eventType == "SPELL_DAMAGE"
        or eventType == "SPELL_PERIODIC_DAMAGE"
        or eventType == "DAMAGE_SHIELD"
end

local function IsIncomingAttackEvent(eventType)
    return IsDamageEvent(eventType)
        or eventType == "SWING_MISSED"
        or eventType == "RANGE_MISSED"
        or eventType == "SPELL_MISSED"
end

local function HandleCombatLog(...)
    local _timestamp, eventType, sourceGUID, _sourceName, _sourceFlags, destGUID = ...
    local playerGUID = UnitGUID("player")
    if sourceGUID == UnitGUID("player") and IsDamageEvent(eventType) then
        FlashOutcome(hitMarker, HIT_VISIBLE_SECONDS)
    end
    if destGUID == playerGUID and sourceGUID ~= nil and IsIncomingAttackEvent(eventType) then
        activeAttackers[sourceGUID] = GetTime() + ATTACKER_VISIBLE_SECONDS
    elseif eventType == "UNIT_DIED" and destGUID ~= nil then
        activeAttackers[destGUID] = nil
        if (destGUID == currentTargetGUID and currentTargetHostile)
            or (destGUID == previousTargetGUID and previousTargetHostile) then
            pendingCombatLootGUID = destGUID
            pendingCombatLootDeadline = GetTime() + COMBAT_LOOT_PENDING_SECONDS
        end
    end
    UpdateAttackerTargetMarker()
    UpdateAttackerCountMarker()
    UpdateCombatLootPendingMarker()
end

local function HandleUiError(message)
    local normalized = string.lower(message or "")
    if string.find(normalized, "wrong way", 1, true)
        or string.find(normalized, "in front", 1, true)
        or string.find(normalized, "facing", 1, true) then
        FlashOutcome(facingMarker, ERROR_VISIBLE_SECONDS)
        return
    end
    if string.find(normalized, "out of range", 1, true)
        or string.find(normalized, "too far away", 1, true)
        or string.find(normalized, "closer", 1, true) then
        FlashOutcome(rangeMarker, ERROR_VISIBLE_SECONDS)
    end
end

local events = CreateFrame("Frame")
events:RegisterEvent("PLAYER_ENTERING_WORLD")
events:RegisterEvent("PLAYER_REGEN_DISABLED")
events:RegisterEvent("PLAYER_REGEN_ENABLED")
events:RegisterEvent("PLAYER_DEAD")
events:RegisterEvent("PLAYER_TARGET_CHANGED")
events:RegisterEvent("COMBAT_LOG_EVENT_UNFILTERED")
events:RegisterEvent("UI_ERROR_MESSAGE")
events:RegisterEvent("LOOT_OPENED")
events:SetScript("OnEvent", function(_self, event, ...)
    if event == "COMBAT_LOG_EVENT_UNFILTERED" then
        HandleCombatLog(...)
    elseif event == "UI_ERROR_MESSAGE" then
        HandleUiError(...)
    elseif event == "LOOT_OPENED" then
        FlashMarker(lootOpenedMarker, LOOT_VISIBLE_SECONDS)
        pendingCombatLootGUID = nil
        pendingCombatLootDeadline = 0
        UpdateCombatLootPendingMarker()
    else
        if event == "PLAYER_ENTERING_WORLD" then
            activeAttackers = {}
            currentTargetGUID = nil
            currentTargetHostile = false
            previousTargetGUID = nil
            previousTargetHostile = false
            pendingCombatLootGUID = nil
            pendingCombatLootDeadline = 0
        end
        if event == "PLAYER_TARGET_CHANGED" or event == "PLAYER_ENTERING_WORLD" then
            TrackTargetChange()
        end
        if event == "PLAYER_REGEN_ENABLED" then
            activeAttackers = {}
            pendingCombatLootGUID = nil
            pendingCombatLootDeadline = 0
            BindControlKeys()
        end
        UpdateCombatMarker()
        UpdateAttackerTargetMarker()
        UpdateAttackerCountMarker()
        UpdateCombatLootPendingMarker()
    end
end)
events:SetScript("OnUpdate", function(_self, elapsed)
    local now = GetTime()
    for frame, deadline in pairs(markerDeadlines) do
        if now >= deadline then
            frame:Hide()
            markerDeadlines[frame] = nil
        end
    end
    for guid, deadline in pairs(activeAttackers) do
        if now >= deadline then
            activeAttackers[guid] = nil
        end
    end
    UpdateAttackerTargetMarker()
    UpdateAttackerCountMarker()
    if pendingCombatLootGUID ~= nil and now >= pendingCombatLootDeadline then
        pendingCombatLootGUID = nil
        pendingCombatLootDeadline = 0
    end
    UpdateCombatLootPendingMarker()
    coordinateElapsed = coordinateElapsed + elapsed
    if coordinateElapsed >= 0.10 then
        coordinateElapsed = 0
        UpdateZoneCoordinates()
        UpdatePlayerHeading()
        UpdateRuntimeTelemetry()
    end
    runtimeMarkerElapsed = runtimeMarkerElapsed + elapsed
    if runtimeMarkerElapsed >= 0.25 then
        runtimeMarkerElapsed = 0
        UpdateRuntimeMarkers()
    end
    oreTooltipElapsed = oreTooltipElapsed + elapsed
    if oreTooltipElapsed >= 0.05 then
        oreTooltipElapsed = 0
        UpdateMinimapOreTooltip()
    end
end)

SLASH_SCREENVISIONCOMBATMARKER1 = "/svatest"
SlashCmdList.SCREENVISIONCOMBATMARKER = function(message)
    local command = string.lower(message or "")
    if command == "on" then
        forcedState = true
    elseif command == "off" then
        forcedState = false
    elseif command == "hit" then
        FlashOutcome(hitMarker, HIT_VISIBLE_SECONDS)
    elseif command == "attacker" then
        forcedAttackerTargetState = true
        UpdateAttackerTargetMarker()
    elseif command == "attacker-off" then
        forcedAttackerTargetState = false
        UpdateAttackerTargetMarker()
    elseif command == "threat1" then
        forcedAttackerCountState = 1
        UpdateAttackerCountMarker()
    elseif command == "threat2" then
        forcedAttackerCountState = 2
        UpdateAttackerCountMarker()
    elseif command == "threat3" then
        forcedAttackerCountState = 3
        UpdateAttackerCountMarker()
    elseif command == "threat-off" then
        forcedAttackerCountState = 0
        UpdateAttackerCountMarker()
    elseif command == "face" then
        FlashOutcome(facingMarker, ERROR_VISIBLE_SECONDS)
    elseif command == "range" then
        FlashOutcome(rangeMarker, ERROR_VISIBLE_SECONDS)
    elseif command == "all" then
        FlashMarker(hitMarker, 3.0)
        FlashMarker(facingMarker, 3.0)
        FlashMarker(rangeMarker, 3.0)
    else
        forcedState = nil
        forcedAttackerTargetState = nil
        forcedAttackerCountState = nil
    end
    UpdateCombatMarker()
end

SLASH_SCREENVISIONCAMERA1 = "/svacamera"
SlashCmdList.SCREENVISIONCAMERA = function(message)
    local command = string.lower(message or "")
    if command == "save" and type(SaveView) == "function" then
        SaveView(5)
    elseif command == "set" and type(SetView) == "function" then
        SetView(5)
    elseif command == "follow" and type(SetCVar) == "function" then
        SetCVar("cameraSmoothStyle", "2")
        SetCVar("cameraSmoothTrackingStyle", "2")
    end
end

UpdateCombatMarker()
UpdateAttackerTargetMarker()
UpdateAttackerCountMarker()
UpdateZoneCoordinates()
UpdatePlayerHeading()
UpdateRuntimeTelemetry()
UpdateRuntimeMarkers()
UpdateMinimapOreTooltip()
BindControlKeys()
