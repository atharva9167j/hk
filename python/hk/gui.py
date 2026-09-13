"""
HK Graphical Model Editor & Inspector GUI Entrypoint.
Allows launching the visual desktop editor directly via:
- python -m hk.gui [file.hk]
- hk-gui [file.hk]
- from hk.gui import launch_gui; launch_gui("model.hk")
"""

import sys
from pathlib import Path

def launch_gui(file_path: str = None):
    """Launch the HK Model Editor GUI."""
    pkg_dir = Path(__file__).resolve().parent
    repo_root = pkg_dir.parent.parent
    tools_gui = repo_root / "tools" / "hk_editor_gui.py"
    
    if tools_gui.exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location("hk_editor_gui", str(tools_gui))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        app = mod.HKEditorApp(file_path)
        app.mainloop()
    else:
        raise FileNotFoundError(f"HK Editor GUI script not found at {tools_gui}")

def main():
    initial_path = sys.argv[1] if len(sys.argv) > 1 else None
    launch_gui(initial_path)

if __name__ == "__main__":
    main()
