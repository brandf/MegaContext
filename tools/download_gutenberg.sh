#!/usr/bin/env bash
set -euo pipefail

# Download a curated subset of Project Gutenberg texts for GistNet training.
# Each title is public domain; total size is < 1 GB compressed.

TARGET_DIR="${1:-data/raw/gutenberg}"
mkdir -p "${TARGET_DIR}"

declare -A URLS=(
  ["pride_and_prejudice"]="https://www.gutenberg.org/cache/epub/1342/pg1342.txt"
  ["sherlock_holmes"]="https://www.gutenberg.org/cache/epub/1661/pg1661.txt"
  ["frankenstein"]="https://www.gutenberg.org/cache/epub/84/pg84.txt"
  ["moby_dick"]="https://www.gutenberg.org/cache/epub/2701/pg2701.txt"
  ["dracula"]="https://www.gutenberg.org/cache/epub/345/pg345.txt"
  ["war_of_the_worlds"]="https://www.gutenberg.org/cache/epub/36/pg36.txt"
  ["little_women"]="https://www.gutenberg.org/cache/epub/37106/pg37106.txt"
  ["the_time_machine"]="https://www.gutenberg.org/cache/epub/35/pg35.txt"
  ["anne_of_green_gables"]="https://www.gutenberg.org/cache/epub/45/pg45.txt"
  ["the_oxford_book_of_american_essays"]="https://www.gutenberg.org/cache/epub/15393/pg15393.txt"
)

echo "Downloading Project Gutenberg subset into ${TARGET_DIR}"
for name in "${!URLS[@]}"; do
  url="${URLS[$name]}"
  dest="${TARGET_DIR}/${name}.txt"
  if [[ -f "${dest}" ]]; then
    echo "[skip] ${dest} already exists"
    continue
  fi
  echo "[fetch] ${name} from ${url}"
  curl -L "${url}" -o "${dest}"
done

echo "Download complete. Total files: $(ls -1 "${TARGET_DIR}" | wc -l)"
