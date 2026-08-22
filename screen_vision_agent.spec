# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ["src\\vision_bot\\runner.py"],
    pathex=[],
    binaries=[],
    datas=[
        ("config.yaml", "."),
        ("data\\routes\\generated\\tanaris_terrain_coverage_cycle_v18_rail.json", "data\\routes\\generated"),
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="screen-vision-agent",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)

