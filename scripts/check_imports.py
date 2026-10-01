"""Check that every Python dependency in requirements.txt can be imported."""

from importlib import import_module


MODULES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "opencv-python-headless": "cv2",
    "scikit-learn": "sklearn",
    "lightgbm": "lightgbm",
    "xgboost": "xgboost",
    "torch": "torch",
    "shap": "shap",
    "matplotlib": "matplotlib",
    "PyYAML": "yaml",
}


def main() -> int:
    failures = []
    for distribution, module in MODULES.items():
        try:
            imported = import_module(module)
            version = getattr(imported, "__version__", "version unavailable")
            print(f"OK  {distribution} ({module}) {version}")
        except Exception as exc:  # report broken native wheels and import errors too
            failures.append((distribution, module, exc))
            print(f"FAIL {distribution} ({module}): {type(exc).__name__}: {exc}")
    if failures:
        print(f"\n{len(failures)} package import(s) failed.")
        return 1
    print(f"\nAll {len(MODULES)} Python package imports succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
