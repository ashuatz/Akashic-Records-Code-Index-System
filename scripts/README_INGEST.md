# Akashic Records - Ingestion CLI

The `ingest.py` script is the primary tool for indexing codebases into the Akashic Records vector database system.

## Features

- **Multi-path ingestion**: Index multiple directories or files in one command
- **Incremental updates**: Only re-index files that have changed (based on MD5 hash)
- **Parallel processing**: Use multiple workers for faster indexing
- **Progress tracking**: Rich progress bars with real-time statistics
- **Error handling**: Continue on errors with comprehensive error reporting
- **Dry-run mode**: Preview what will be indexed without making changes
- **Flexible filtering**: Include/exclude patterns from settings
- **Statistics tracking**: View index status and metrics

## Installation

Ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

Initialize the database:

```bash
python scripts/init_collection.py --settings config/settings.yaml
```

## Usage

### Basic Indexing

Index a single directory:

```bash
python scripts/ingest.py ingest --path "D:/Code/MyProject"
```

Index multiple directories:

```bash
python scripts/ingest.py ingest \
  --path "D:/Code/Project1" \
  --path "D:/Code/Project2" \
  --path "D:/Code/Project3"
```

Index a single file:

```bash
python scripts/ingest.py ingest --path "D:/Code/MyProject/main.py"
```

### Incremental Updates

Only index files that have changed since last indexing:

```bash
python scripts/ingest.py ingest --path "D:/Code/MyProject" --incremental
```

This checks the MD5 hash of each file against the stored hash and skips unchanged files.

### Advanced Options

**Custom batch size and workers:**

```bash
python scripts/ingest.py ingest \
  --path "D:/Code/MyProject" \
  --batch-size 50 \
  --workers 8
```

**Non-recursive (only top-level files):**

```bash
python scripts/ingest.py ingest --path "D:/Code/MyProject" --no-recursive
```

**Skip errors and continue:**

```bash
python scripts/ingest.py ingest \
  --path "D:/Code/MyProject" \
  --skip-errors
```

**Dry-run mode (preview without indexing):**

```bash
python scripts/ingest.py ingest \
  --path "D:/Code/MyProject" \
  --dry-run
```

**Custom settings file:**

```bash
python scripts/ingest.py ingest \
  --path "D:/Code/MyProject" \
  --settings /path/to/settings.yaml
```

### View Index Status

Display statistics about the indexed codebase:

```bash
python scripts/ingest.py status
```

This shows:
- Total files, symbols, and chunks indexed
- Last indexing timestamp
- Breakdown by programming language
- File and chunk counts per language

Example output:

```
Akashic Records - Index Status

                Index Overview
┌─────────────────┬───────────────────────────┐
│ Metric          │ Value                     │
├─────────────────┼───────────────────────────┤
│ Total Files     │ 1,234                     │
│ Total Symbols   │ 45,678                    │
│ Total Chunks    │ 23,456                    │
│ Last Indexed    │ 2026-01-30T12:34:56.789   │
└─────────────────┴───────────────────────────┘

            Files by Language
┌────────────┬────────┬─────────┐
│ Language   │  Files │  Chunks │
├────────────┼────────┼─────────┤
│ python     │    450 │   8,900 │
│ javascript │    320 │   6,400 │
│ typescript │    280 │   5,600 │
│ csharp     │    184 │   2,556 │
└────────────┴────────┴─────────┘
```

### Delete Files from Index

Remove files matching a path pattern:

```bash
python scripts/ingest.py delete --path "D:/Code/OldProject"
```

This will:
1. Find all indexed files containing the path pattern
2. Show a preview of files to be deleted
3. Ask for confirmation
4. Remove files, chunks, and symbols from the database

**Warning**: This operation cannot be undone. Always review the preview carefully.

## Configuration

The ingestion behavior is controlled by `config/settings.yaml`:

### Include Extensions

Specify which file extensions to index:

```yaml
indexing:
  include_extensions:
    - ".cs"
    - ".cpp"
    - ".h"
    - ".py"
    - ".js"
    - ".ts"
    - ".tsx"
```

### Exclude Patterns

Skip files matching these patterns:

```yaml
indexing:
  exclude_patterns:
    - "ThirdParty/*"
    - "node_modules/*"
    - "__pycache__/*"
    - "*.min.js"
    - "*.generated.cs"
```

Patterns support:
- Glob-style wildcards: `ThirdParty/*`
- Extension wildcards: `*.min.js`
- Directory matching: `node_modules/*`

### Chunking Settings

Control how code is split into chunks:

```yaml
indexing:
  chunk:
    max_tokens: 4000
    overlap_tokens: 200
    min_chunk_size: 50
```

### Batch Processing

Default batch size and worker count:

```yaml
indexing:
  batch_size: 100
  parallel_workers: 4
```

## How It Works

### File Scanning

1. Recursively scans specified directories
2. Filters files by extension (include list)
3. Excludes files matching patterns (exclude list)
4. Returns list of files to process

### Incremental Mode

1. Calculates MD5 hash of file content
2. Compares with stored hash in database
3. Skips files with matching hashes
4. Indexes only new or changed files

### Processing Pipeline

For each file:

1. **Hash Calculation**: Compute MD5 hash for change detection
2. **Chunking**: Split file into semantic chunks using tree-sitter
3. **Embedding**: Generate vector embeddings via llama.cpp server
4. **Storage**: Store vectors in Qdrant and metadata in SQLite
5. **Metadata Update**: Record file hash, chunk count, timestamp

### Parallel Execution

- Files are processed in batches
- Multiple workers process batches concurrently
- Semaphore limits concurrent operations
- Progress bar shows real-time status

## Error Handling

### Skip Errors Mode

With `--skip-errors`:
- Failed files are logged but don't stop processing
- Summary shows success/failure counts
- First 10 errors are displayed

Without `--skip-errors`:
- Processing stops on first error
- Full error details are shown
- No partial indexing occurs

### Common Errors

**Database locked:**
```
sqlite3.OperationalError: database is locked
```
Solution: Ensure no other processes are accessing the database

**File permission errors:**
```
PermissionError: [Errno 13] Permission denied
```
Solution: Run with appropriate permissions or use `--skip-errors`

**Embedding service down:**
```
ConnectionError: Failed to connect to embedding server
```
Solution: Ensure llama.cpp server is running at configured URL

## Performance Tips

### Optimal Worker Count

- **CPU-bound**: workers = CPU cores
- **I/O-bound**: workers = 2-4x CPU cores
- **Network-bound**: workers = 8-16

Test with different values:

```bash
python scripts/ingest.py ingest --path /large/codebase --workers 8
```

### Batch Size

- **Small files**: Larger batch size (200-500)
- **Large files**: Smaller batch size (50-100)
- **Limited memory**: Reduce batch size

### Incremental Updates

For regular updates to actively developed codebases:

```bash
# Initial full index
python scripts/ingest.py ingest --path /project

# Daily incremental updates
python scripts/ingest.py ingest --path /project --incremental
```

Only changed files are re-indexed, dramatically reducing processing time.

## Integration Examples

### CI/CD Pipeline

```yaml
# .github/workflows/index-code.yml
name: Index Codebase

on:
  push:
    branches: [main]

jobs:
  index:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Index codebase
        run: |
          python scripts/ingest.py ingest \
            --path . \
            --incremental \
            --skip-errors
```

### Scheduled Indexing (Windows)

```batch
@echo off
REM index-daily.bat

cd /d "%~dp0\.."
python scripts\ingest.py ingest ^
  --path "<project-path-1>" ^
  --path "<project-path-2>" ^
  --incremental ^
  --skip-errors

echo Indexing complete at %date% %time%
```

Schedule with Task Scheduler to run daily.

### Scheduled Indexing (Linux/macOS)

```bash
#!/bin/bash
# index-daily.sh

cd /opt/akashic-records
python3 scripts/ingest.py ingest \
  --path /home/user/code/project1 \
  --path /home/user/code/project2 \
  --incremental \
  --skip-errors

echo "Indexing complete at $(date)"
```

Add to crontab:

```
0 2 * * * /opt/akashic-records/index-daily.sh >> /var/log/akashic-index.log 2>&1
```

## Troubleshooting

### No files found

**Problem**: "Found 0 files"

**Solutions**:
1. Check file extensions in `settings.yaml` include your file types
2. Verify path exists and is correct
3. Check exclude patterns aren't too broad
4. Use `--dry-run` to preview file detection

### Slow indexing

**Problem**: Processing takes too long

**Solutions**:
1. Increase `--workers` count
2. Increase `--batch-size`
3. Use `--incremental` for updates
4. Check embedding server performance
5. Ensure database is on fast storage (SSD)

### Memory issues

**Problem**: Out of memory errors

**Solutions**:
1. Reduce `--batch-size`
2. Reduce `--workers` count
3. Process directories separately
4. Increase system swap space

### Database errors

**Problem**: SQLite errors during processing

**Solutions**:
1. Ensure database directory exists
2. Check file permissions
3. Reinitialize database: `python scripts/init_collection.py --reset`
4. Check disk space

## Command Reference

### ingest command

```
python scripts/ingest.py ingest [OPTIONS]
```

**Options:**

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--path` | `-p` | TEXT | Required | Directory or file to index (multiple) |
| `--recursive` | | FLAG | True | Scan subdirectories |
| `--no-recursive` | | FLAG | False | Only scan specified directory |
| `--incremental` | | FLAG | False | Only index changed files |
| `--batch-size` | | INT | From config | Batch size for processing |
| `--workers` | | INT | From config | Number of parallel workers |
| `--settings` | | TEXT | config/settings.yaml | Path to settings file |
| `--skip-errors` | | FLAG | False | Continue on errors |
| `--dry-run` | | FLAG | False | Preview without indexing |

### status command

```
python scripts/ingest.py status [OPTIONS]
```

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--settings` | TEXT | config/settings.yaml | Path to settings file |

### delete command

```
python scripts/ingest.py delete [OPTIONS]
```

**Options:**

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--path` | `-p` | TEXT | Required | Path pattern to match |
| `--settings` | | TEXT | config/settings.yaml | Path to settings file |

## Future Enhancements

Planned features for future versions:

- [ ] Watch mode: Automatically re-index on file changes
- [ ] Progress resumption: Resume interrupted indexing
- [ ] Export/import: Backup and restore index data
- [ ] Deduplication: Detect and handle duplicate code
- [ ] Language detection: Auto-detect file language
- [ ] Compression: Compress stored chunks
- [ ] Cloud storage: Support S3/Azure blob storage
- [ ] Distributed indexing: Multi-machine processing
- [ ] Web UI: Browser-based indexing interface

## Contributing

To add support for new file types:

1. Add tree-sitter parser to `requirements.txt`
2. Add extension to `include_extensions` in settings
3. Update chunker to handle new language
4. Test with sample files

## License

Part of the Akashic Records project.
