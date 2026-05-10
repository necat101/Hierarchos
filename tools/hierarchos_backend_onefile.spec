# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
    copy_metadata,
)


ROOT = Path(SPECPATH).resolve().parent

hiddenimports = collect_submodules("hierarchos")
hiddenimports += [
    "huggingface_hub",
    "numpy",
    "safetensors",
    "tokenizers",
    "torch",
    "torch._C",
    "torch.backends.cuda",
    "torch.cuda",
    "torch.nn",
    "torch.nn.functional",
    "tqdm",
    "transformers",
    "transformers.models.auto.tokenization_auto",
]

datas = [
    (str(ROOT / "hierarchos"), "hierarchos"),
]

binaries = []

try:
    binaries += collect_dynamic_libs("torch")
except Exception:
    pass

try:
    datas += collect_data_files("transformers", include_py_files=False)
except Exception:
    pass

for package in [
    "torch",
    "transformers",
    "huggingface_hub",
    "tokenizers",
    "safetensors",
    "numpy",
    "tqdm",
]:
    try:
        datas += copy_metadata(package)
    except Exception:
        pass

excludes = [
    "accelerate",
    "aiohttp",
    "bitsandbytes",
    "IPython",
    "lxml",
    "mcp",
    "nltk",
    "optuna",
    "peft",
    "PIL",
    "PIL.ImageQt",
    "pyarrow",
    "redis",
    "soundfile",
    "sqlalchemy",
    "tiktoken",
    "tkinter",
    "torchaudio",
    "torchtext",
    "torchvision",
    "boto3",
    "botocore",
    "cv2",
    "datasets",
    "django",
    "fastapi",
    "jupyter",
    "lm_eval",
    "matplotlib",
    "notebook",
    "openai",
    "pandas",
    "pygame",
    "pytest",
    "scipy",
    "sklearn",
    "tensorflow",
    "uvicorn",
]

a = Analysis(
    [str(ROOT / "hierarchos_bridge_server.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=1,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="hierarchos-backend",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
)
