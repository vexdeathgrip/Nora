#!/usr/bin/env python3
"""Nora CLI — wrapper around hermes with Nora setup."""

import sys
import subprocess


def main():
    # Intercept "nora setup" to run Nora's setup
    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        from pathlib import Path
        from nora.setup import setup_nora
        from hermes_constants import get_hermes_home

        force = "--force" in sys.argv
        hermes_home = get_hermes_home()
        print(f"Installing Nora into {hermes_home}...")
        result = setup_nora(hermes_home, skip_existing=not force)
        for item in result.get("installed", []):
            print(f"  ✓ {item}")
        for item in result.get("skipped", []):
            print(f"  · {item} (skipped)")
        print("Done.")
        return

    # Everything else: pass through to hermes
    result = subprocess.run(
        ["hermes"] + sys.argv[1:],
        env={**__import__("os").environ},
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
