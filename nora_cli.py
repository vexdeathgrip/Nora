#!/usr/bin/env python3
"""Nora CLI — standalone setup and management."""

import sys
import argparse


def main():
    parser = argparse.ArgumentParser(
        prog="nora",
        description="Nora — your personal AI companion",
    )
    subparsers = parser.add_subparsers(dest="command")

    # nora setup
    setup_parser = subparsers.add_parser("setup", help="Install Nora's plugins, skills, and config")
    setup_parser.add_argument("--force", action="store_true", help="Overwrite existing files")

    args = parser.parse_args()

    if args.command == "setup":
        from pathlib import Path
        from nora.setup import setup_nora
        from hermes_constants import get_hermes_home

        hermes_home = get_hermes_home()
        print(f"Installing Nora into {hermes_home}...")
        result = setup_nora(hermes_home, skip_existing=not args.force)
        for item in result.get("installed", []):
            print(f"  ✓ {item}")
        for item in result.get("skipped", []):
            print(f"  · {item} (skipped)")
        print("Done.")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
