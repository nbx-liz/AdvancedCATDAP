# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all, copy_metadata
import sys
import os

block_cipher = None

# AdvancedCATDAP source path
pkg_path = os.path.abspath("advanced_catdap")

datas = [
    # Include the entire package source to ensure re-imports works if needed
    # But usually PyInstaller collects code. We need non-code assets.
    # Frontend app.py is needed for 'streamlit run' which takes a path.
    ('advanced_catdap', 'advanced_catdap'), # Bundle entire package as data if needed, or rely on analysis.
    ('advanced_catdap/frontend/app.py', 'advanced_catdap/frontend'), 
]

binaries = []
hiddenimports = [
    'uvicorn.logging', 'uvicorn.loops', 'uvicorn.loops.auto', 'uvicorn.protocols', 
    'uvicorn.protocols.http', 'uvicorn.protocols.http.auto', 'uvicorn.protocols.websockets', 
    'uvicorn.protocols.websockets.auto', 'uvicorn.lifespan', 'uvicorn.lifespan.on',
    'fastapi', 'streamlit', 'pandas', 'sklearn', 'plotly', 'scipy',
    'httpx', 'tqdm', 'filelock', 'regex', 'joblib', 'webview', 'clr', 'System.Windows.Forms'
]

# Collect Streamlit
st_datas, st_binaries, st_hiddenimports = collect_all('streamlit')
datas += st_datas
binaries += st_binaries
hiddenimports += st_hiddenimports

# Metadata
datas += copy_metadata('streamlit')
datas += copy_metadata('joblib')
datas += copy_metadata('requests')
datas += copy_metadata('packaging')
datas += copy_metadata('numpy')
datas += copy_metadata('scipy')

a = Analysis(
    ['scripts/windows_main.py'],
    pathex=[os.path.abspath('.')],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='AdvancedCATDAP_Native312',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False, # Set to True for debugging
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None, # Add icon later if available
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='AdvancedCATDAP_Native312',
)
