#!/usr/bin/env python3
"""Nora setup — installs plugins, skills, cron jobs, and config."""

import os
import shutil
from pathlib import Path


def get_nora_dir() -> Path:
    """Get the nora/ directory relative to this file."""
    return Path(__file__).parent


def setup_nora(hermes_home: Path, *, skip_existing: bool = True) -> dict:
    """Install Nora's plugins, skills, cron jobs, and config.

    Returns a dict with lists of installed/skipped items.
    """
    nora_dir = get_nora_dir()
    result = {"installed": [], "skipped": []}

    # Install plugins
    plugins_dir = hermes_home / "plugins"
    plugins_dir.mkdir(parents=True, exist_ok=True)
    for plugin in (nora_dir / "plugins").iterdir():
        if plugin.is_dir():
            dest = plugins_dir / plugin.name
            if skip_existing and dest.exists():
                result["skipped"].append(f"plugin:{plugin.name}")
            else:
                shutil.copytree(plugin, dest, dirs_exist_ok=True)
                result["installed"].append(f"plugin:{plugin.name}")

    # Install skills
    skills_dir = hermes_home / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    for skill in (nora_dir / "skills").iterdir():
        if skill.is_dir():
            dest = skills_dir / skill.name
            if skip_existing and dest.exists():
                result["skipped"].append(f"skill:{skill.name}")
            else:
                shutil.copytree(skill, dest, dirs_exist_ok=True)
                result["installed"].append(f"skill:{skill.name}")

    # Copy cron jobs (don't overwrite existing)
    cron_dir = hermes_home / "cron"
    cron_dir.mkdir(parents=True, exist_ok=True)
    jobs_file = nora_dir / "cron" / "jobs.json"
    if jobs_file.exists():
        dest = cron_dir / "jobs.json"
        if skip_existing and dest.exists():
            result["skipped"].append("cron:jobs.json")
        else:
            shutil.copy2(jobs_file, dest)
            result["installed"].append("cron:jobs.json")

    # Copy config (don't overwrite existing)
    config_file = nora_dir / "config" / "config.yaml"
    if config_file.exists():
        dest = hermes_home / "config.yaml"
        if skip_existing and dest.exists():
            result["skipped"].append("config:config.yaml")
        else:
            shutil.copy2(config_file, dest)
            result["installed"].append("config:config.yaml")

    # Install systemd service
    systemd_dir = Path.home() / ".config" / "systemd" / "user"
    systemd_dir.mkdir(parents=True, exist_ok=True)
    service_file = nora_dir / "systemd" / "llama.service"
    if service_file.exists():
        dest = systemd_dir / "llama.service"
        shutil.copy2(service_file, dest)
        result["installed"].append("systemd:llama.service")
        # Try to enable the service
        try:
            os.system("systemctl --user daemon-reload 2>/dev/null")
            os.system("systemctl --user enable llama.service 2>/dev/null")
        except Exception:
            pass

    return result
