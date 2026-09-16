#!/usr/bin/env bash
# Fetch the public request traces the experiments replay.
#
#   - Azure LLM Inference Trace 2023 (conversational + code)  -> T3, T4
#   - ShareGPT conversations                                  -> T1
#
# Everything lands in data/traces/.  Safe to re-run: existing files are
# left alone.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRACE_DIR="$REPO_ROOT/data/traces"
mkdir -p "$TRACE_DIR"

AZURE_BASE="https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data"
SHAREGPT_URL="https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"

fetch() {
  local url="$1" dest="$2"
  if [[ -s "$dest" ]]; then
    echo "==> $(basename "$dest") already present, skipping"
    return
  fi
  echo "==> Downloading $(basename "$dest")"
  # --fail so an HTML error page never gets written out as if it were data.
  if ! curl --fail --location --progress-bar --output "$dest.partial" "$url"; then
    rm -f "$dest.partial"
    echo "error: failed to download $url" >&2
    return 1
  fi
  mv "$dest.partial" "$dest"
}

fetch "$AZURE_BASE/AzureLLMInferenceTrace_conv.csv" \
      "$TRACE_DIR/AzureLLMInferenceTrace_conv.csv"
fetch "$AZURE_BASE/AzureLLMInferenceTrace_code.csv" \
      "$TRACE_DIR/AzureLLMInferenceTrace_code.csv"
fetch "$SHAREGPT_URL" \
      "$TRACE_DIR/ShareGPT_V3_unfiltered_cleaned_split.json"

echo
echo "==> Traces available in $TRACE_DIR:"
ls -1sh "$TRACE_DIR"
echo
echo "Note: the derived per-use-case traces for T3 (narrativeqa.csv,"
echo "      code_generation_kv.csv) are committed under data/traces/ and"
echo "      need no download."
