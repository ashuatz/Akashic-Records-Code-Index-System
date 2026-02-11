#!/usr/bin/env python3
"""
Fix paths in compile_commands.json when copying from another machine.

Usage:
    python fix_compile_commands_paths.py --input compile_commands.json --old-path "D:/unity" --new-path "C:/TA/unity"
"""

import argparse
import json
import re
from pathlib import Path


def fix_paths(input_file: str, output_file: str, old_path: str, new_path: str):
    """Replace old paths with new paths in compile_commands.json."""

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Normalize paths for comparison
    old_path = old_path.replace('\\', '/')
    new_path = new_path.replace('\\', '/')

    # Also handle backslash variants
    old_path_backslash = old_path.replace('/', '\\')
    new_path_backslash = new_path.replace('/', '\\')

    count = 0
    for entry in data:
        for key in ['directory', 'file', 'command']:
            if key in entry:
                original = entry[key]
                # Replace forward slash version
                entry[key] = entry[key].replace(old_path, new_path)
                # Replace backslash version
                entry[key] = entry[key].replace(old_path_backslash, new_path_backslash)
                if entry[key] != original:
                    count += 1

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

    print(f"Fixed {count} path occurrences")
    print(f"Output: {output_file}")
    print(f"Total entries: {len(data)}")


def main():
    parser = argparse.ArgumentParser(description='Fix paths in compile_commands.json')
    parser.add_argument('--input', '-i', required=True, help='Input compile_commands.json')
    parser.add_argument('--output', '-o', help='Output file (default: overwrite input)')
    parser.add_argument('--old-path', required=True, help='Old base path to replace')
    parser.add_argument('--new-path', required=True, help='New base path')

    args = parser.parse_args()

    output = args.output or args.input
    fix_paths(args.input, output, args.old_path, args.new_path)


if __name__ == '__main__':
    main()
