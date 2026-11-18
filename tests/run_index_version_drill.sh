#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

CXX="${CXX:-g++}"
COMMON_FLAGS=(-std=c++11 -O2 -fopenmp -I"$ROOT/hnswlib")

"$CXX" "${COMMON_FLAGS[@]}" \
    -DHNSWLIB_INDEX_FORMAT_VERSION=1 -DHNSWLIB_INDEX_READER_VERSION=1 \
    "$ROOT/tests/cpp/index_version_drill.cpp" -o "$TMP/drill_v1"
"$CXX" "${COMMON_FLAGS[@]}" \
    -DHNSWLIB_INDEX_FORMAT_VERSION=2 -DHNSWLIB_INDEX_READER_VERSION=2 \
    "$ROOT/tests/cpp/index_version_drill.cpp" -o "$TMP/drill_v2"

echo "== build two format versions =="
"$TMP/drill_v1" build "$TMP/legacy_v1.idx"
"$TMP/drill_v2" build "$TMP/current_v2.idx"

echo "== same-version load =="
"$TMP/drill_v1" load "$TMP/legacy_v1.idx"
"$TMP/drill_v2" load "$TMP/current_v2.idx"

echo "== legacy v1 file in current v2 reader =="
"$TMP/drill_v2" inspect "$TMP/legacy_v1.idx"
"$TMP/drill_v2" load "$TMP/legacy_v1.idx"

echo "== current v2 file in legacy v1 reader (must reject) =="
"$TMP/drill_v1" inspect "$TMP/current_v2.idx"
set +e
REJECT_OUTPUT="$("$TMP/drill_v1" load "$TMP/current_v2.idx" 2>&1)"
REJECT_STATUS=$?
set -e
echo "$REJECT_OUTPUT"
if [[ $REJECT_STATUS -eq 0 ]]; then
    echo "expected legacy reader to reject v2 index" >&2
    exit 1
fi
if [[ "$REJECT_OUTPUT" != *"requires reader v2"* ]]; then
    echo "legacy reader rejected without an explicit version reason" >&2
    exit 1
fi

echo "== truncation must fail before normal load =="
NEW_SIZE="$(wc -c < "$TMP/current_v2.idx")"
head -c "$((NEW_SIZE - 32))" "$TMP/current_v2.idx" > "$TMP/truncated.idx"
set +e
TRUNCATED_OUTPUT="$("$TMP/drill_v2" load "$TMP/truncated.idx" 2>&1)"
TRUNCATED_STATUS=$?
set -e
echo "$TRUNCATED_OUTPUT"
if [[ $TRUNCATED_STATUS -eq 0 ]]; then
    echo "expected truncated index to fail" >&2
    exit 1
fi
if [[ "$TRUNCATED_OUTPUT" != *"truncated"* ]]; then
    echo "truncation failure was not explicit" >&2
    exit 1
fi

echo "== batch classification =="
"$TMP/drill_v2" scan "$TMP/legacy_v1.idx" "$TMP/current_v2.idx" \
    "$TMP/truncated.idx"
