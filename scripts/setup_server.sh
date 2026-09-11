#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

TOOLS_DIR="$REPO_ROOT/.tools"
TOOLS_BIN="$TOOLS_DIR/bin"
HMMER_VERSION="3.4"
SKIP_PREPARE_TOOLS="${CALM_SKIP_PREPARE_TOOLS:-0}"

log() { printf '[calm-setup] %s\n' "$*"; }
warn() { printf '[calm-setup] WARN  %s\n' "$*" >&2; }
die() { printf '[calm-setup] ERROR %s\n' "$*" >&2; exit 1; }

mkdir -p "$TOOLS_BIN"
export PATH="$TOOLS_BIN:$HOME/.local/bin:$PATH"

# ~3 GB of CUDA wheels overrun uv's 30s default timeout on throttled links.
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-2}"
SYNC_RETRIES="${CALM_SYNC_RETRIES:-5}"

install_uv() {
  if command -v uv >/dev/null 2>&1; then
    log "uv found: $(command -v uv) ($(uv --version))"
    return 0
  fi
  command -v curl >/dev/null 2>&1 || die "curl is required to bootstrap uv"
  log "installing uv into ~/.local/bin"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null 2>&1 || die "uv bootstrap failed, install it manually from https://docs.astral.sh/uv/"
  log "uv installed: $(uv --version)"
}

sync_python_env() {
  log "resolving python 3.11 + torch cu124 environment (this downloads ~3 GB on first run)"
  uv python install 3.11
  log "UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT}s, up to ${SYNC_RETRIES} attempts (partial downloads resume between attempts)"
  local attempt=1
  # --reinstall-package forces the editable install (and its generated finder) to be
  # rebuilt from the sources as they are right now. A finder built while calm/__init__.py
  # was missing bakes in NAMESPACES={'calm': []} and keeps calm a namespace package
  # forever, even after the file is restored.
  until uv sync --extra preprocess --reinstall-package calm-repro; do
    if [ "$attempt" -ge "$SYNC_RETRIES" ]; then
      die "uv sync failed after ${SYNC_RETRIES} attempts; re-run 'bash scripts/setup_server.sh' or raise UV_HTTP_TIMEOUT / CALM_SYNC_RETRIES"
    fi
    warn "uv sync attempt ${attempt} failed (likely a network timeout), retrying in 5s ($((attempt + 1))/${SYNC_RETRIES})"
    attempt=$((attempt + 1))
    sleep 5
  done
  log "virtualenv ready at $REPO_ROOT/.venv"
  if uv pip install "parasail>=1.3.4" >/dev/null 2>&1; then
    log "parasail installed (fast leak-audit alignment backend)"
  else
    warn "parasail wheel unavailable, leak audit will use the biopython aligner (same results, slower)"
  fi
}

install_mmseqs() {
  if command -v mmseqs >/dev/null 2>&1; then
    log "mmseqs found: $(command -v mmseqs)"
    return 0
  fi
  if [ "$(uname -s)" != "Linux" ]; then
    warn "no static mmseqs2 build for $(uname -s), install it with 'brew install mmseqs2' or 'conda install -c bioconda mmseqs2'"
    return 1
  fi
  local arch flavour url
  arch="$(uname -m)"
  case "$arch" in
    x86_64)
      if grep -qm1 avx2 /proc/cpuinfo 2>/dev/null; then flavour="avx2"; else flavour="sse41"; fi
      ;;
    aarch64 | arm64)
      flavour="arm64"
      ;;
    *)
      warn "no prebuilt mmseqs2 for arch $arch, build it from https://github.com/soedinglab/MMseqs2"
      return 1
      ;;
  esac
  url="https://mmseqs.com/latest/mmseqs-linux-$flavour.tar.gz"
  log "downloading mmseqs2 static build ($flavour)"
  rm -rf "$TOOLS_DIR/mmseqs" "$TOOLS_DIR/mmseqs.tar.gz"
  curl -fsSL "$url" -o "$TOOLS_DIR/mmseqs.tar.gz"
  tar -xzf "$TOOLS_DIR/mmseqs.tar.gz" -C "$TOOLS_DIR"
  rm -f "$TOOLS_DIR/mmseqs.tar.gz"
  if [ ! -x "$TOOLS_DIR/mmseqs/bin/mmseqs" ]; then
    warn "mmseqs2 archive layout unexpected, expected $TOOLS_DIR/mmseqs/bin/mmseqs"
    return 1
  fi
  ln -sf "$TOOLS_DIR/mmseqs/bin/mmseqs" "$TOOLS_BIN/mmseqs"
  log "mmseqs2 installed: $("$TOOLS_BIN/mmseqs" version)"
}

install_hmmer() {
  if command -v hmmscan >/dev/null 2>&1; then
    log "hmmscan found: $(command -v hmmscan)"
    return 0
  fi
  if command -v apt-get >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
    log "installing hmmer via apt-get"
    if sudo -n apt-get install -y hmmer >/dev/null 2>&1 && command -v hmmscan >/dev/null 2>&1; then
      log "hmmscan installed: $(command -v hmmscan)"
      return 0
    fi
    warn "apt-get install hmmer failed, falling back to a source build"
  fi
  if ! command -v make >/dev/null 2>&1 || { ! command -v cc >/dev/null 2>&1 && ! command -v gcc >/dev/null 2>&1; }; then
    warn "no C toolchain found, skipping HMMER; 'calm prepare' will not run but 'data/calm_dataset.csv' is already in the repo"
    return 1
  fi
  local src="$TOOLS_DIR/src"
  log "building HMMER $HMMER_VERSION from source (a few minutes)"
  mkdir -p "$src"
  curl -fsSL "http://eddylab.org/software/hmmer/hmmer-$HMMER_VERSION.tar.gz" -o "$src/hmmer.tar.gz"
  rm -rf "$src/hmmer-$HMMER_VERSION"
  tar -xzf "$src/hmmer.tar.gz" -C "$src"
  (
    cd "$src/hmmer-$HMMER_VERSION"
    ./configure --prefix="$TOOLS_DIR/hmmer" >/dev/null
    make -j"$(getconf _NPROCESSORS_ONLN)" >/dev/null
    make install >/dev/null
  )
  local binary
  for binary in "$TOOLS_DIR/hmmer/bin/"*; do
    ln -sf "$binary" "$TOOLS_BIN/$(basename "$binary")"
  done
  rm -rf "$src"
  log "HMMER installed: $(hmmscan -h | head -2 | tail -1)"
}

report_gpus() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader | sed 's/^/[calm-setup] gpu /'
  else
    warn "nvidia-smi not found, the environment will fall back to CPU"
  fi
}

check_sources() {
  # Must run before uv sync: the editable install snapshots the package layout.
  local missing=() path
  for path in calm/__init__.py calm/schema.py calm/cli.py; do
    [ -s "$path" ] || missing+=("$path")
  done
  if [ "${#missing[@]}" -eq 0 ]; then
    return 0
  fi
  warn "missing source files: ${missing[*]}"
  command -v git >/dev/null 2>&1 || die "git is required to restore $REPO_ROOT/calm"
  if git -C "$REPO_ROOT" restore --source=HEAD -- calm; then
    log "restored calm/ from HEAD"
  else
    die "could not restore calm/ from git; run: git -C $REPO_ROOT checkout -- calm"
  fi
}

verify() {
  log "verifying installation"
  uv run python scripts/check_env.py
  uv run calm info
}

install_uv
check_sources
sync_python_env
report_gpus
install_mmseqs || warn "mmseqs2 missing: clustering will degrade to exact sequence identity and the 40/60/80 regimes become identical"
if [ "$SKIP_PREPARE_TOOLS" = "1" ]; then
  log "CALM_SKIP_PREPARE_TOOLS=1, skipping HMMER (only needed by 'calm prepare')"
else
  install_hmmer || true
fi
verify

log "done. activate with:  source scripts/env.sh"
log "or run everything through uv:  uv run calm experiment --config configs/rtx4090/ag80.yaml"
