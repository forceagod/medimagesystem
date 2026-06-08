#!/usr/bin/env python3
"""Update the latest entry of a DM log JSON file after user confirmation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def update_dm_log(
    dm_log_path: Path,
    updates: dict[str, str],
    dry_run: bool = False,
) -> None:
    """Update the latest entry in a DM log JSON array with the given key-value pairs."""
    with open(dm_log_path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or len(data) == 0:
        print(f"Error: {dm_log_path} is not a non-empty JSON array", file=sys.stderr)
        sys.exit(1)

    latest = data[-1]
    if not isinstance(latest, dict):
        print(f"Error: latest entry in {dm_log_path} is not a JSON object", file=sys.stderr)
        sys.exit(1)

    changes: list[tuple[str, str, str]] = []
    for key, new_value in updates.items():
        old_value = latest.get(key)
        if old_value == new_value:
            continue
        if key not in latest:
            print(f"Warning: key '{key}' does not exist in latest DM-log entry, adding it", file=sys.stderr)
        changes.append((key, old_value, new_value))
        latest[key] = new_value

    if not changes:
        print("No changes to apply.")
        return

    print(f"Updating latest DM-log entry (version: {latest.get('DMP版本号', 'unknown')}):")
    for key, old, new in changes:
        print(f"  {key}: {old!r} -> {new!r}")

    if dry_run:
        print("\n[dry-run] File not written.")
    else:
        with open(dm_log_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"\nUpdated {dm_log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Update the latest entry of a DM log JSON file")
    parser.add_argument("--dm-log", required=True, type=Path, help="Path to the DM log JSON file")
    parser.add_argument("--set", dest="updates", action="append", metavar="KEY=VALUE",
                        default=[], help="Key-value pair to set (repeatable)")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without writing")
    args = parser.parse_args()

    updates: dict[str, str] = {}
    for item in args.updates:
        if "=" not in item:
            print(f"Error: --set value must be in KEY=VALUE format, got: {item!r}", file=sys.stderr)
            sys.exit(1)
        key, value = item.split("=", 1)
        updates[key] = value

    if not updates:
        print("Error: at least one --set KEY=VALUE is required", file=sys.stderr)
        sys.exit(1)

    update_dm_log(args.dm_log, updates, args.dry_run)


if __name__ == "__main__":
    main()
