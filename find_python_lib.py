import sys
import sysconfig
import os
from pathlib import Path

print(f"Python Executable: {sys.executable}")
print(f"Prefix: {sys.prefix}")
print(f"Base Prefix: {sys.base_prefix}")
print(f"Exec Prefix: {sys.exec_prefix}")

print("\n--- Sysconfig Paths ---")
for scheme in sysconfig.get_scheme_names():
    try:
        paths = sysconfig.get_paths(scheme=scheme)
        print(f"Scheme '{scheme}':")
        if 'stdlib' in paths: print(f"  stdlib: {paths['stdlib']}")
        if 'data' in paths: print(f"  data: {paths['data']}")
        if 'include' in paths: print(f"  include: {paths['include']}")
    except:
        pass

print("\n--- Searching for python313.lib ---")
search_roots = [
    sys.prefix,
    sys.base_prefix,
    sys.exec_prefix,
    os.path.dirname(sys.executable),
    os.path.join(os.path.dirname(sys.executable), "libs"),
    os.path.join(sys.prefix, "libs"),
]

found = []
for root in search_roots:
    if not os.path.exists(root): continue
    print(f"Searching in: {root}")
    for path in Path(root).rglob("python313.lib"):
        print(f"  FOUND: {path}")
        found.append(str(path))

if not found:
    print("❌ python313.lib NOT FOUND in standard locations.")
else:
    print(f"✅ Found {len(found)} locations.")
