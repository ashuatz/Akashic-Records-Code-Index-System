#!/usr/bin/env python3
"""
Generate compile_commands.json from Visual Studio solution/project files.

This script parses .vcxproj files to extract compiler flags and scans
source directories to build a compile_commands.json for clangd.

Usage:
    python generate_compile_commands.py --sln "C:/TA/unity/Projects/VisualStudio/Unity.sln"
    python generate_compile_commands.py --vcxproj-dir "C:/TA/unity/Projects/VisualStudio"
    python generate_compile_commands.py --source-dir "C:/TA/unity" --vcxproj "C:/TA/unity/Projects/VisualStudio/Projects/WinPlayer.vcxproj"
"""

import argparse
import json
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Set
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# XML namespaces used in .vcxproj files
NS = {'msbuild': 'http://schemas.microsoft.com/developer/msbuild/2003'}


def parse_vcxproj(vcxproj_path: Path, config: str = "Debug", platform: str = "x64") -> List[Dict]:
    """
    Parse a .vcxproj file and extract compile commands.

    Args:
        vcxproj_path: Path to the .vcxproj file
        config: Build configuration (Debug/Release)
        platform: Platform (x64/Win32)

    Returns:
        List of compile command entries
    """
    try:
        tree = ET.parse(vcxproj_path)
        root = tree.getroot()
    except ET.ParseError as e:
        logger.warning(f"Failed to parse {vcxproj_path}: {e}")
        return []

    project_dir = vcxproj_path.parent.resolve()
    commands = []

    # Extract include directories
    include_dirs = set()
    defines = set()
    additional_options = []

    # Find all ItemDefinitionGroup for the specified config/platform
    condition_pattern = f"'$(Configuration)|$(Platform)'=='{config}|{platform}'"

    for item_def_group in root.findall('.//msbuild:ItemDefinitionGroup', NS):
        condition = item_def_group.get('Condition', '')
        if condition_pattern.lower() not in condition.lower() and condition:
            continue

        # Extract ClCompile settings
        for cl_compile in item_def_group.findall('.//msbuild:ClCompile', NS):
            # Include directories
            for inc_elem in cl_compile.findall('msbuild:AdditionalIncludeDirectories', NS):
                if inc_elem.text:
                    for inc in inc_elem.text.split(';'):
                        inc = inc.strip()
                        if inc and not inc.startswith('%'):
                            # Resolve relative paths
                            inc_path = Path(inc)
                            if not inc_path.is_absolute():
                                inc_path = project_dir / inc_path
                            include_dirs.add(str(inc_path.resolve()))

            # Preprocessor definitions
            for def_elem in cl_compile.findall('msbuild:PreprocessorDefinitions', NS):
                if def_elem.text:
                    for define in def_elem.text.split(';'):
                        define = define.strip()
                        if define and not define.startswith('%'):
                            defines.add(define)

            # Additional options
            for opt_elem in cl_compile.findall('msbuild:AdditionalOptions', NS):
                if opt_elem.text:
                    additional_options.append(opt_elem.text.strip())

    # Also check PropertyGroup for global settings
    for prop_group in root.findall('.//msbuild:PropertyGroup', NS):
        for inc_elem in prop_group.findall('msbuild:IncludePath', NS):
            if inc_elem.text:
                for inc in inc_elem.text.split(';'):
                    inc = inc.strip()
                    if inc and not inc.startswith('%') and not inc.startswith('$'):
                        inc_path = Path(inc)
                        if not inc_path.is_absolute():
                            inc_path = project_dir / inc_path
                        if inc_path.exists():
                            include_dirs.add(str(inc_path.resolve()))

    # Find all source files
    source_extensions = {'.cpp', '.c', '.cc', '.cxx'}
    source_files = []

    for item_group in root.findall('.//msbuild:ItemGroup', NS):
        for cl_compile in item_group.findall('msbuild:ClCompile', NS):
            include = cl_compile.get('Include')
            if include:
                # Check if file is excluded for this configuration
                excluded = False
                for exclude_elem in cl_compile.findall('msbuild:ExcludedFromBuild', NS):
                    condition = exclude_elem.get('Condition', '')
                    if condition_pattern.lower() in condition.lower():
                        if exclude_elem.text and exclude_elem.text.lower() == 'true':
                            excluded = True
                            break

                if not excluded:
                    file_path = Path(include)
                    if not file_path.is_absolute():
                        file_path = project_dir / file_path

                    if file_path.suffix.lower() in source_extensions:
                        source_files.append(file_path.resolve())

    # Build compile commands
    for source_file in source_files:
        if not source_file.exists():
            continue

        # Build command line
        cmd_parts = ['clang++']

        # Add include directories
        for inc_dir in sorted(include_dirs):
            cmd_parts.append(f'-I{inc_dir}')

        # Add defines
        for define in sorted(defines):
            cmd_parts.append(f'-D{define}')

        # Add standard flags for compatibility
        cmd_parts.extend([
            '-std=c++17',
            '-fms-compatibility',
            '-fms-extensions',
            '-Wno-microsoft',
        ])

        # Add source file
        cmd_parts.append(str(source_file))

        commands.append({
            'directory': str(project_dir),
            'command': ' '.join(cmd_parts),
            'file': str(source_file)
        })

    return commands


def find_vcxproj_files(directory: Path) -> List[Path]:
    """Find all .vcxproj files in a directory."""
    vcxproj_files = []
    for vcxproj in directory.rglob('*.vcxproj'):
        vcxproj_files.append(vcxproj)
    return vcxproj_files


def parse_solution(sln_path: Path) -> List[Path]:
    """Parse a .sln file and extract project file paths."""
    projects = []
    sln_dir = sln_path.parent

    try:
        with open(sln_path, 'r', encoding='utf-8-sig') as f:
            content = f.read()
    except Exception as e:
        logger.error(f"Failed to read solution file: {e}")
        return []

    # Match Project lines: Project("{GUID}") = "Name", "Path.vcxproj", "{GUID}"
    project_pattern = r'Project\("[^"]*"\)\s*=\s*"[^"]*",\s*"([^"]+\.vcxproj)"'

    for match in re.finditer(project_pattern, content, re.IGNORECASE):
        proj_path = match.group(1)
        full_path = sln_dir / proj_path.replace('\\', '/')
        if full_path.exists():
            projects.append(full_path.resolve())

    return projects


def generate_compile_commands(
    vcxproj_files: List[Path],
    output_path: Path,
    config: str = "Debug",
    platform: str = "x64"
) -> int:
    """
    Generate compile_commands.json from vcxproj files.

    Returns:
        Number of compile commands generated
    """
    all_commands = []
    seen_files = set()

    for vcxproj in vcxproj_files:
        logger.info(f"Parsing: {vcxproj.name}")
        commands = parse_vcxproj(vcxproj, config, platform)

        for cmd in commands:
            # Deduplicate by file path
            file_path = cmd['file']
            if file_path not in seen_files:
                seen_files.add(file_path)
                all_commands.append(cmd)

    # Write compile_commands.json
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_commands, f, indent=2)

    logger.info(f"Generated {len(all_commands)} compile commands")
    logger.info(f"Output: {output_path}")

    return len(all_commands)


def extract_flags_from_makefile_vcxproj(vcxproj_path: Path, config: str = "Debug", platform: str = "x64") -> Dict:
    """
    Extract compiler flags from a Makefile-style vcxproj.

    Returns dict with 'includes', 'defines', 'old_base_path'
    """
    try:
        tree = ET.parse(vcxproj_path)
        root = tree.getroot()
    except ET.ParseError as e:
        logger.warning(f"Failed to parse {vcxproj_path}: {e}")
        return {}

    includes = set()
    defines = set()
    old_base_path = None

    # Find PropertyGroup with NMake settings
    for prop_group in root.findall('.//msbuild:PropertyGroup', NS):
        condition = prop_group.get('Condition', '')

        # Try to match config/platform
        if config.lower() in condition.lower() or not condition:
            # Include paths
            inc_elem = prop_group.find('msbuild:NMakeIncludeSearchPath', NS)
            if inc_elem is not None and inc_elem.text:
                for inc in inc_elem.text.split(';'):
                    inc = inc.strip()
                    if inc:
                        includes.add(inc)
                        # Detect old base path
                        if 'unity' in inc.lower() and old_base_path is None:
                            # Extract base path (e.g., D:\Dev\Unity\unity)
                            match = re.match(r'^([A-Za-z]:[/\\][^;]+?unity)', inc, re.IGNORECASE)
                            if match:
                                old_base_path = match.group(1).replace('\\', '/')

            # Preprocessor definitions
            def_elem = prop_group.find('msbuild:NMakePreprocessorDefinitions', NS)
            if def_elem is not None and def_elem.text:
                for d in def_elem.text.split(';'):
                    d = d.strip()
                    if d:
                        defines.add(d)

    return {
        'includes': list(includes),
        'defines': list(defines),
        'old_base_path': old_base_path
    }


def generate_from_source_scan(
    source_dir: Path,
    vcxproj_path: Path,
    output_path: Path,
    new_base_path: str,
    config: str = "Debug",
    platform: str = "x64"
) -> int:
    """
    Generate compile_commands.json by scanning source directory
    and using flags from vcxproj.
    """
    # Extract flags from vcxproj
    flags = extract_flags_from_makefile_vcxproj(vcxproj_path, config, platform)

    if not flags:
        logger.error("Could not extract flags from vcxproj")
        return 0

    old_base = flags.get('old_base_path', '')
    includes = flags.get('includes', [])
    defines = flags.get('defines', [])

    logger.info(f"Old base path: {old_base}")
    logger.info(f"New base path: {new_base_path}")
    logger.info(f"Found {len(includes)} include paths")
    logger.info(f"Found {len(defines)} defines")

    # Fix paths
    fixed_includes = []
    for inc in includes:
        fixed = inc.replace('\\', '/')
        if old_base:
            fixed = fixed.replace(old_base.replace('\\', '/'), new_base_path)
        fixed_includes.append(fixed)

    # Scan for source files
    source_extensions = {'.cpp', '.c', '.cc', '.cxx'}
    source_files = []

    logger.info(f"Scanning source directory: {source_dir}")
    for ext in source_extensions:
        source_files.extend(source_dir.rglob(f'*{ext}'))

    logger.info(f"Found {len(source_files)} source files")

    # Generate compile commands
    commands = []
    for source_file in source_files:
        # Skip certain directories
        path_str = str(source_file).replace('\\', '/')
        if any(skip in path_str for skip in ['/External/', '/ThirdParty/', '/artifacts/', '/build/']):
            continue

        cmd_parts = ['clang++']

        # Add includes
        for inc in fixed_includes[:50]:  # Limit to avoid too long command lines
            if Path(inc).exists():
                cmd_parts.append(f'-I{inc}')

        # Add key defines
        for d in list(defines)[:100]:  # Limit defines
            cmd_parts.append(f'-D{d}')

        # Add standard flags
        cmd_parts.extend([
            '-std=c++17',
            '-fms-compatibility',
            '-fms-extensions',
            '-Wno-microsoft',
            str(source_file)
        ])

        commands.append({
            'directory': str(source_file.parent),
            'command': ' '.join(cmd_parts),
            'file': str(source_file)
        })

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(commands, f, indent=2)

    logger.info(f"Generated {len(commands)} compile commands")
    logger.info(f"Output: {output_path}")

    return len(commands)


def main():
    parser = argparse.ArgumentParser(
        description='Generate compile_commands.json from Visual Studio projects'
    )
    parser.add_argument(
        '--sln',
        type=str,
        help='Path to .sln solution file'
    )
    parser.add_argument(
        '--vcxproj-dir',
        type=str,
        help='Directory containing .vcxproj files'
    )
    parser.add_argument(
        '--vcxproj',
        type=str,
        help='Single vcxproj file to extract flags from (for Makefile projects)'
    )
    parser.add_argument(
        '--source-dir',
        type=str,
        help='Source directory to scan for C++ files'
    )
    parser.add_argument(
        '--new-base-path',
        type=str,
        help='New base path to replace old paths in vcxproj'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='compile_commands.json',
        help='Output file path (default: compile_commands.json)'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='Debug',
        help='Build configuration (default: Debug)'
    )
    parser.add_argument(
        '--platform', '-p',
        type=str,
        default='x64',
        help='Platform (default: x64)'
    )

    args = parser.parse_args()

    output_path = Path(args.output).resolve()

    # Mode 1: Source scan with vcxproj flags (for Makefile projects like Unity)
    if args.source_dir and args.vcxproj:
        source_dir = Path(args.source_dir).resolve()
        vcxproj_path = Path(args.vcxproj).resolve()
        new_base = args.new_base_path or str(source_dir)

        if not source_dir.exists():
            logger.error(f"Source directory not found: {source_dir}")
            return 1
        if not vcxproj_path.exists():
            logger.error(f"vcxproj not found: {vcxproj_path}")
            return 1

        count = generate_from_source_scan(
            source_dir,
            vcxproj_path,
            output_path,
            new_base.replace('\\', '/'),
            args.config,
            args.platform
        )
        return 0 if count > 0 else 1

    # Mode 2: Traditional vcxproj parsing
    if not args.sln and not args.vcxproj_dir:
        parser.error("Either --sln, --vcxproj-dir, or (--source-dir and --vcxproj) must be specified")

    vcxproj_files = []

    if args.sln:
        sln_path = Path(args.sln).resolve()
        if not sln_path.exists():
            logger.error(f"Solution file not found: {sln_path}")
            return 1

        logger.info(f"Parsing solution: {sln_path}")
        vcxproj_files = parse_solution(sln_path)
        logger.info(f"Found {len(vcxproj_files)} projects in solution")

    if args.vcxproj_dir:
        dir_path = Path(args.vcxproj_dir).resolve()
        if not dir_path.exists():
            logger.error(f"Directory not found: {dir_path}")
            return 1

        logger.info(f"Scanning directory: {dir_path}")
        vcxproj_files.extend(find_vcxproj_files(dir_path))
        logger.info(f"Found {len(vcxproj_files)} .vcxproj files")

    if not vcxproj_files:
        logger.error("No .vcxproj files found")
        return 1

    count = generate_compile_commands(
        vcxproj_files,
        output_path,
        args.config,
        args.platform
    )

    if count == 0:
        logger.warning("No compile commands generated")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
