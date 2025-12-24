import os
import sys
import torch
import sysconfig
import ctypes

# --- Fix for Python Build Environment (Windows) ---
# 1. Fix spaces in include paths (monkey-patch sysconfig).
# 2. Fix missing python313.lib (add libs to LIB env var).

def _fix_python_build_environment_windows():
    if os.name != 'nt': return
    
    try:
        _GetShortPathNameW = ctypes.windll.kernel32.GetShortPathNameW
        _GetShortPathNameW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32]
        _GetShortPathNameW.restype = ctypes.c_uint32

        def get_short_path(path):
            if not path or not os.path.exists(path): return path
            buf_size = _GetShortPathNameW(path, None, 0)
            if buf_size == 0: return path
            buf = ctypes.create_unicode_buffer(buf_size)
            _GetShortPathNameW(path, buf, buf_size)
            return buf.value

        # --- 1. Patch sysconfig for include paths ---
        _original_get_path = sysconfig.get_path
        _original_get_paths = sysconfig.get_paths

        def _patched_get_path(name, scheme=sysconfig.get_default_scheme(), vars=None, expand=True):
            path = _original_get_path(name, scheme, vars, expand)
            if name in ('include', 'platinclude') and path:
                return get_short_path(path)
            return path

        def _patched_get_paths(scheme=sysconfig.get_default_scheme(), vars=None, expand=True):
            paths = _original_get_paths(scheme, vars, expand)
            for key in ('include', 'platinclude'):
                if key in paths and paths[key]:
                    paths[key] = get_short_path(paths[key])
            return paths

        sysconfig.get_path = _patched_get_path
        sysconfig.get_paths = _patched_get_paths
        print("INFO: Patched sysconfig to use short paths for includes.")

        # --- 2. Add 'libs' to LIB environment variable ---
        exec_prefix_short = get_short_path(sys.exec_prefix)
        libs_path = os.path.join(exec_prefix_short, 'libs')
        
        if os.path.exists(libs_path):
            current_lib = os.environ.get('LIB', '')
            os.environ['LIB'] = f"{libs_path};{current_lib}"
            print(f"INFO: Added '{libs_path}' to LIB environment variable.")
        else:
            print(f"WARNING: Could not find libs path at '{libs_path}'")

    except Exception as e:
        print(f"WARNING: Failed to setup Python build environment: {e}")

# Call it immediately on import
_fix_python_build_environment_windows()

_HAS_DIRECTML = False
try:
    import torch_directml
    _HAS_DIRECTML = True
except ImportError:
    _HAS_DIRECTML = False

def pick_device(args=None):
    """Picks the best available device for PyTorch training."""
    if args and hasattr(args, 'device') and args.device:
        if args.device in ['dml', 'directml']:
            if _HAS_DIRECTML: 
                print("INFO: User explicitly requested DirectML.")
                return torch_directml.device()
            else: 
                print("Warning: DirectML requested but torch-directml not available. Falling back to auto-detect.")
        elif args.device == 'cuda':
            if torch.cuda.is_available(): 
                return torch.device("cuda")
            else: 
                print("Warning: CUDA requested but not available. Falling back to auto-detect.")
        elif args.device == 'cpu':
            return torch.device("cpu")
    
    if torch.cuda.is_available():
        return torch.device("cuda")
    
    return torch.device("cpu")

import os
import subprocess
import sys
import json

def setup_msvc_environment():
    """Sets up the MSVC environment for torch.compile on Windows (Robust Parity)."""
    if os.name != 'nt': return

    CACHE_FILE = 'vcvars_path.cache.txt'
    vcvars_path = None

    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                cached_path = f.read().strip()
            if os.path.exists(cached_path):
                vcvars_path = cached_path
        except: pass

    if vcvars_path is None:
        vswhere_prog_x86 = os.path.expandvars(r"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe")
        vswhere_prog = os.path.expandvars(r"%ProgramFiles%\Microsoft Visual Studio\Installer\vswhere.exe")
        vswhere_path = vswhere_prog_x86 if os.path.exists(vswhere_prog_x86) else (vswhere_prog if os.path.exists(vswhere_prog) else None)
        
        if vswhere_path:
            try:
                cmd = [vswhere_path, "-latest", "-products", "*", "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-find", r"VC\Auxiliary\Build\vcvars64.bat", "-nologo"]
                result = subprocess.run(cmd, capture_output=True, text=True, check=True, encoding='utf-8')
                found_path = result.stdout.strip()
                if found_path and os.path.exists(found_path):
                    vcvars_path = found_path
            except: pass

    # Manual Fallback Prompt (Parity with original script)
    if vcvars_path is None:
        print("\n" + "---" * 20)
        print("MSVC 64-bit Compiler Setup (Manual Fallback)")
        print("---" * 20)
        print("Could not automatically find vcvars64.bat.")
        try:
            user_path = input("Enter full path to vcvars64.bat (or press Enter to skip): ").strip()
            if user_path and os.path.exists(user_path):
                vcvars_path = user_path
        except EOFError: pass
        print("---" * 20)

    if vcvars_path:
        try:
            with open(CACHE_FILE, 'w') as f: f.write(vcvars_path)
            vcvars_dir = os.path.dirname(vcvars_path)
            vcvarsall_path = os.path.join(vcvars_dir, "vcvarsall.bat")
            if os.path.exists(vcvarsall_path):
                python_exe = sys.executable
                cmd = f'"{vcvarsall_path}" x64 >NUL 2>&1 && echo ^"---ENV-JSON-START---^" && "{python_exe}" -c "import json, os; print(json.dumps(dict(os.environ)))"'
                
                # Robust output extraction
                result = subprocess.run(cmd, capture_output=True, text=True, shell=True, check=True, cwd=vcvars_dir, encoding='utf-8', errors='ignore')
                json_start_marker = "---ENV-JSON-START---"
                idx = result.stdout.find(json_start_marker)
                if idx != -1:
                    raw_after = result.stdout[idx + len(json_start_marker):]
                    j_start = raw_after.find('{')
                    j_end = raw_after.rfind('}')
                    if j_start != -1 and j_end != -1:
                        json_str = raw_after[j_start : j_end + 1]
                        env_vars = json.loads(json_str)
                        for k in ['PATH', 'LIB', 'INCLUDE']:
                            if k in env_vars: os.environ[k] = env_vars[k]
                        print("✅ INFO: MSVC environment loaded.")
        except Exception as e:
            print(f"Warning: Failed to load MSVC environment: {e}")

def set_threads(n: int):
    if n is None or n <= 0: return
    try:
        import torch
        torch.set_num_threads(n)
        os.environ['OMP_NUM_THREADS'] = str(n)
        print(f"INFO: Threads set to {n}")
    except Exception as e:
        print(f"Warning: Could not set thread count. {e}")

def is_directml_device(device):
    """Check if a device is DirectML."""
    if _HAS_DIRECTML:
        return isinstance(device, type(torch_directml.device()))
    return False

def get_device_type(device):
    """Get normalized device type string for compatibility checks."""
    if is_directml_device(device):
        return 'dml'
    elif isinstance(device, torch.device):
        return device.type
    return str(device)
