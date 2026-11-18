#!/usr/bin/env python3
"""Inspect hnswlib index compatibility without loading graph bodies."""

import argparse
import csv
import json
import os
import sys

try:
    import hnswlib
except ImportError:
    sys.path.insert(
        0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    import hnswlib


DEFAULT_PATTERNS = (".bin", ".idx", ".index")


def discover_paths(inputs, recursive, patterns):
    paths = []
    for item in inputs:
        if os.path.isdir(item):
            if not recursive:
                raise ValueError(
                    "'%s' is a directory; pass --recursive to scan it" % item
                )
            for root, _, files in os.walk(item):
                for filename in files:
                    if not patterns or filename.endswith(tuple(patterns)):
                        paths.append(os.path.join(root, filename))
        else:
            paths.append(item)
    return sorted(dict.fromkeys(paths))


def flatten(result):
    rows = []
    for category in ("upgradeable", "rebuild_required"):
        for item in result[category]:
            metadata = item["metadata"]
            rows.append(
                {
                    "path": item["path"],
                    "category": category,
                    "action": item["action"],
                    "status": item["status"],
                    "reason": item["reason"],
                    "format_version": metadata["format_version"],
                    "min_reader_version": metadata["min_reader_version"],
                    "space": metadata["space"],
                    "dim": metadata["dim"],
                    "M": metadata["M"],
                    "ef_construction": metadata["ef_construction"],
                    "byte_order": metadata["byte_order"],
                    "warnings": " | ".join(item["warnings"]),
                }
            )
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Classify hnswlib indexes into upgradeable and "
            "rebuild-required batches. Only headers/footer metadata are read."
        )
    )
    parser.add_argument("paths", nargs="+", help="index files or directories")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument(
        "--pattern",
        action="append",
        default=None,
        help="suffix filter for recursive scans (default: .bin .idx .index)",
    )
    parser.add_argument("--space", default="", help="expected space: l2/ip/cosine")
    parser.add_argument("--dim", type=int, default=0)
    parser.add_argument("--M", type=int, default=0)
    parser.add_argument("--ef-construction", type=int, default=0)
    parser.add_argument("--reader-version", type=int, default=0)
    parser.add_argument("--format", choices=("json", "csv"), default="json")
    parser.add_argument("--output", help="write report to this path")
    args = parser.parse_args(argv)

    patterns = tuple(args.pattern) if args.pattern else DEFAULT_PATTERNS
    try:
        paths = discover_paths(args.paths, args.recursive, patterns)
    except ValueError as error:
        parser.error(str(error))

    result = hnswlib.scan_index_files(
        paths,
        space=args.space,
        dim=args.dim,
        M=args.M,
        ef_construction=args.ef_construction,
        reader_version=args.reader_version,
    )

    output = sys.stdout
    close_output = False
    if args.output:
        output = open(args.output, "w", newline="", encoding="utf-8")
        close_output = True
    try:
        if args.format == "json":
            json.dump(result, output, indent=2, sort_keys=True)
            output.write("\n")
        else:
            rows = flatten(result)
            fieldnames = [
                "path",
                "category",
                "action",
                "status",
                "reason",
                "format_version",
                "min_reader_version",
                "space",
                "dim",
                "M",
                "ef_construction",
                "byte_order",
                "warnings",
            ]
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    finally:
        if close_output:
            output.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
