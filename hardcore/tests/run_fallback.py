"""Minimal test runner used when pytest is unavailable: runs every test_* function."""
import importlib
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))


def main(argv):
    pattern = argv[1] if len(argv) > 1 else ""
    modules = sorted(f[:-3] for f in os.listdir(HERE) if f.startswith("test_") and f.endswith(".py"))
    passed = failed = skipped = 0
    failures = []
    for mod_name in modules:
        mod = importlib.import_module(mod_name)
        for name in sorted(dir(mod)):
            if not name.startswith("test_") or pattern not in f"{mod_name}::{name}":
                continue
            fn = getattr(mod, name)
            if not callable(fn):
                continue
            try:
                fn()
                passed += 1
                print(f"PASS {mod_name}::{name}")
            except BaseException as exc:   # noqa: BLE001 - pytest.skip() raises a BaseException
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                if type(exc).__name__ in ("Skipped", "SkipTest"):
                    skipped += 1
                    print(f"SKIP {mod_name}::{name}: {exc}")
                    continue
                failed += 1
                failures.append((f"{mod_name}::{name}", traceback.format_exc()))
                print(f"FAIL {mod_name}::{name}")
    for name, tb in failures:
        print("\n" + "=" * 70 + f"\n{name}\n" + tb)
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
