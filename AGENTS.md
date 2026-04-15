# AGENTS.md

acestep.cpp is a C++17 implementation of ACE-Step 1.5 music generation using GGML (a patched fork). It takes text prompts and generates stereo 48kHz MP3/WAV audio. The codebase consists of a custom GGML backend with two new ops (`GGML_OP_SNAKE`, `GGML_OP_COL2IM_1D`), three main pipelines (LM, Synth, Understand), and an HTTP server with embedded WebUI.

## Quick Reference

| Action | Command |
|--------|---------|
| Build (CPU + BLAS) | `./buildcpu.sh` or `cmake .. -DGGML_BLAS=ON` |
| Build (CUDA) | `./buildcuda.sh` |
| Build (Vulkan) | `./buildvulkan.sh` |
| Build (All backends) | `./buildall.sh` |
| Build (macOS Metal) | `./buildcpu.sh` (Metal auto-enabled) |
| Format code | `./format.sh` or `clang-format -i file.cpp` |
| Lint check | `clang-format --dry-run --Werror file.cpp` |
| Download models | `./models.sh` (requires `pip install hf`) |
| Run server | `./build/ace-server --models models/ --loras loras/` |

## Project Structure

```
acestep.cpp/
├── ggml/                    # Patched GGML submodule (do NOT edit unless adding ops)
├── src/                     # Core headers only (no .cpp files)
│   ├── *.h                 # Self-contained header files (qwen3-*.h, dit-*.h, vae-*.h)
│   ├── pipeline-*.cpp/.h   # Three pipelines: lm, synth, understand
│   ├── request.cpp/.h      # JSON parsing (yyjson-based)
│   └── task-types.h        # Task constants and instructions
├── tools/                   # Executable entry points
│   ├── ace-lm.cpp          # LLM pipeline (CoT + audio codes)
│   ├── ace-synth.cpp       # DiT + VAE synthesis
│   ├── ace-server.cpp      # HTTP server with embedded WebUI
│   └── ace-understand.cpp  # Reverse pipeline (audio -> metadata)
├── mp3/                     # MIT-licensed MP3 encoder/decoder (no external deps)
├── vendor/                  # Third-party (cpp-httplib, yyjson)
├── build/                   # CMake output (binaries + backends)
├── models/                  # GGUF model directory
└── examples/                # Shell scripts + JSON examples
```

### File Organization Patterns

- **Headers only**: `src/*.h` files are self-contained (no separate .cpp). Architecture definitions are header-only (qwen3-lm.h, qwen3-enc.h, dit.h, vae.h, fsq-tok.h, fsq-detok.h, cond-enc.h).
- **Pipelines**: Three separate `.cpp` files (`pipeline-lm.cpp`, `pipeline-synth.cpp`, `pipeline-understand.cpp`) each implement one direction of the pipeline.
- **Tools**: Each `tools/*.cpp` is a standalone binary that links against `acestep-core` static library.
- **No C++ classes**: The codebase uses C-style structs with explicit initialization functions (`request_init()`, `ctx_init()`). No inheritance, no virtual functions.

## Build System & Gotchas

### CMake Flags

```bash
# Essential backends
-DGGML_CUDA=ON      # NVIDIA GPU (cuBLAS, requires CUDA Toolkit)
-DGGML_VULKAN=ON    # AMD/Intel GPU (Vulkan SDK required)
-DGGML_METAL=ON     # macOS Metal (auto-enabled on macOS)
-DGGML_BLAS=ON      # CPU BLAS (Accelerate on macOS, OpenBLAS on Linux)
-DGGML_CPU_ALL_VARIANTS=ON  # Enable all CPU arch variants

# Runtime loading (build all, load at runtime)
-DGGML_BACKEND_DL=ON  # Backends compile as .so/.dylib, loaded via dlopen
```

### Critical Build Notes

1. **Submodule required**: `git submodule update --init` is mandatory before building. The `ggml/` directory is a submodule, not a regular directory.
2. **Backend .so location**: With `-DGGML_BACKEND_DL=ON`, backend libraries are output to `build/` (not `build/lib/`). The `CMAKE_LIBRARY_OUTPUT_DIRECTORY` is set to `CMAKE_BINARY_DIR` so `ggml_backend_load_all()` finds them at runtime.
3. **Version header**: `tools/version.cmake` generates `build/version.h` with git commit hash. Every build must have this target (`add_dependencies(target version)`).
4. **WebUI embedding**: `tools/webui/public/index.html.gz` is converted to C header `build/index.html.gz.hpp` by `tools/xxd.cmake`. The `.gz` file is committed to git; the WebUI is built separately with `npm run build` in `tools/webui/`.
5. **CUDA architectures**: Default is `75-virtual;80-virtual;86-real;89-real` (Turing→Ampere). Override with `-DCMAKE_CUDA_ARCHITECTURES=native` for local builds.
6. **pthread on older glibc**: CMake explicitly finds Threads and links `Threads::Threads`. Older glibc (<2.34) requires explicit pthread linking.

### Build Scripts

- `buildall.sh`: CPU + CUDA + Vulkan + runtime loading (`-DGGML_CPU_ALL_VARIANTS=ON -DGGML_CUDA=ON -DGGML_VULKAN=ON -DGGML_BACKEND_DL=ON`)
- `buildcuda.sh`: CUDA only
- `buildvulkan.sh`: Vulkan only
- `buildcpu.sh`: CPU + BLAS (BLAS auto-detected: Accelerate on macOS, OpenBLAS on Linux if available)
- `buildwebui.sh`: Builds the WebUI separately (npm)

## Code Conventions

### Formatting

- **clang-format**: Column limit 120, 4-space indentation, no tabs (`UseTab: Never`)
- **Brace style**: Custom (attached for functions, custom wrapping rules in `.clang-format`)
- **Line endings**: LF only (`LineEnding: LF`)
- **Format command**: `./format.sh` formats all files. To check without modifying: `clang-format --dry-run --Werror file.cpp`

### Naming & Style

- **Structs**: Lowercase with hyphens converted to underscores (`AceRequest`, `ggml_context`, `ggml_backend`)
- **Functions**: Lowercase with underscores, C-style (`request_parse()`, `request_to_json()`, `ctx_init()`, `ctx_free()`)
- **Constants**: `inline constexpr const char *` for string constants in header files (`TASK_TEXT2MUSIC`, `LM_INSTRUCTION`)
- **No C++ features**: No classes, no virtual functions, no exceptions (C++17 standard but C-style code)
- **Comments**: `//` for single-line, block comments for file headers. No Doxygen-style `/** */`.

### Error Handling

- **Return values**: Functions return `bool` for success/failure or `-1`/`0` for errors. No exceptions.
- **File I/O**: Uses `fread()`/`fwrite()` with `_FORTIFY_SOURCE=2` (hardened libgcrc on Linux).
- **Memory**: All GGML tensors managed by `ggml_context`. Manual `ggml_free()` required after each inference step.
- **JSON parsing**: `yyjson` library (fast, no exceptions). `request_parse()` returns `false` on malformed JSON.

### Includes

- **Project headers**: `#include "filename.h"` (relative to `src/`)
- **GGML headers**: `#include "ggml.h"`, `#include "ggml-backend.h"` (from `ggml/include/`)
- **System headers**: `#include <cstdio>`, `#include <cstdint>`, `#include <string>`, `#include <vector>`
- **Ordering**: Project headers first, then system headers (clang-format `IncludeBlocks: Regroup`)

## Architecture & Data Flow

### The Three Pipelines

1. **ace-lm** (LLM): Takes `AceRequest` with `caption`, optionally `lyrics`/`bpm`/`duration`/etc. Outputs enriched JSON with `audio_codes` (comma-separated 5Hz token IDs).
   - Phase 1: CoT generates missing metadata (bpm, keyscale, timesignature, lyrics) via FSM-constrained decoding
   - Phase 2: Generates 5Hz FSQ audio codes (64000 vocab: 8×8×8×5×5×5)
   - Uses `qwen3-lm.h` (partial LM head for audio codes only)

2. **ace-synth** (DiT + VAE): Takes `AceRequest` with `audio_codes`. Outputs MP3 or WAV.
   - BPE tokenize → Qwen3-Embedding (text encoder) → CondEncoder (lyric + timbre) → FSQ detokenizer
   - DiT flow-matching (ODE Euler or SDE Stochastic) with LoRA support
   - VAE decode (AutoencoderOobleck, tiled with overlap) → stereo 48kHz → MP3 encoder

3. **ace-understand** (reverse): Takes audio file → VAE encode → FSQ tokenize → LLM understand prompt → metadata + lyrics JSON.

### Task Types (from `task-types.h`)

| Task | Input | LLM Used | DiT Context | Notes |
|------|-------|----------|-------------|-------|
| `text2music` | caption | Yes | Silence | Default mode |
| `cover` | caption + src-audio | No | FSQ roundtrip (lossy) | Diverges freely from source |
| `cover-nofsq` | caption + src-audio | No | Clean VAE latents | Stays close to source |
| `repaint` | caption + src-audio + region | No | Masked (silence in zone) | Regenerate time region |
| `lego` | caption + src-audio + track | Yes | Full backing track | Generate new instrument layer |
| `extract` | src-audio + track | No | Full source | Isolate stem |
| `complete` | src-audio + track | Yes | Single stem | Generate full mix |

**Track names**: `vocals`, `backing_vocals`, `drums`, `bass`, `guitar`, `keyboard`, `percussion`, `strings`, `synth`, `fx`, `brass`, `woodwinds`.

### GGML Ops & Patches

The `ggml/` submodule adds two custom ops:

1. **`GGML_OP_SNAKE`**: Fused Snake activation `y = x + sin²(a*x) * inv_b`. Used 36× per VAE tile. Without fusion, would require 5 separate kernels (mul, sin, sqr, mul, add).
2. **`GGML_OP_COL2IM_1D`**: Scatter-add for GEMM-based `conv_transpose_1d`. Decomposes transposed conv into `mul_mat + col2im_1d` to use cuBLAS/BLAS/MPS tensor cores.
3. **Metal `kernel_im2col_1d`**: Flat 1D dispatch for VAE's 1D convolutions (full SIMD utilization vs 2-7% with generic 2D kernel).
4. **CUDA bugfix**: `im2col` gridDim.y overflow fixed with grid-stride loop (VAE output widths up to 491520 exceed CUDA's 65535 gridDim limit).

### Memory Management

- **KV Cache**: Single 4D tensor `[D, max_seq, Nkv, n_sets]` shared across batch elements and CFG paths.
- **CFG batching**: N conditional + N unconditional sequences packed into one forward pass (2×N tokens). Combined as `logits = uncond + scale * (cond - uncond)`.
- **VAE tiling**: Decode uses tiles of 256 latent frames with 64-frame overlap (`--vae-chunk`, `--vae-overlap`). Reduces VRAM for long audio.
- **Model loading**: `--keep-loaded` flag keeps models in VRAM between requests (server mode). Without it, models are freed after each request.

## API & Endpoints

### HTTP Server (ace-server)

All POST endpoints are **asynchronous** (return job ID immediately):

| Endpoint | Method | Body | Returns |
|----------|--------|------|---------|
| `/lm` | POST | `AceRequest` JSON | `{"id":"..."}` |
| `/synth` | POST | `AceRequest` or array `[...]` or multipart | Job ID |
| `/synth?wav=1` | POST | Same as above | WAV output instead of MP3 |
| `/understand` | POST | multipart (audio) or JSON (codes-only) | Job ID |
| `/job?id=N` | GET | - | `{"status":"running|done|failed|cancelled"}` |
| `/job?id=N&result=1` | GET | - | Job result (JSON or audio) |
| `/job?id=N&cancel=1` | POST | - | Cancel job |
| `/props` | GET | - | Server config, available models, defaults |
| `/health` | GET | - | `{"status":"ok"}` |
| `/` | GET | - | Embedded WebUI (gzipped HTML) |

**Model selection**: Use `lm_model`, `synth_model`, `lora`, `lora_scale` fields in JSON body to select which model to load. Empty string = keep current or load first available.

**Job storage**: Completed jobs stored in memory (LRU, max 10 entries). Evicted when pool exceeds limit.

### CLI Batching

- **ace-lm**: `lm_batch_size` in JSON creates N variations (different lyrics, codes, metadata).
- **ace-synth**: `synth_batch_size` in JSON creates N variations (same codes, different diffusion trajectories). Multiple `--request` files combined with `synth_batch_size` = single GPU batch.
- **Seed resolution**: `seed=-1` → random once per input JSON. Explicit seed used as-is. Batching uses consecutive seeds internally.

## Testing & Debugging

### Test Scripts

Located in `tests/`:

- `debug-lm-logits.py`: Compare GGML vs PyTorch first-token logits (requires `../ACE-Step-1.5/`)
- `debug-dit-cossim.py`: Per-layer cosine similarity GGML vs Python
- `debug-detok-cossim.py`: FSQ detokenizer step-by-step comparison
- `test-philox.cpp`: RNG test (Philox noise generator)

### Debug Flags

Common debug options across CLIs:

```bash
--no-fsm          # Disable FSM constrained decoding (LM)
--no-fa           # Disable flash attention
--no-batch-cfg    # Split CFG into two N=1 forwards (debug CFG)
--clamp-fp16      # Clamp hidden states to FP16 range
--dump-logits <path>  # Dump prefill logits (binary f32)
--dump-tokens <path>  # Dump prompt token IDs (CSV)
--dump <dir>      # Dump intermediate tensors (synth)
```

### Common Pitfalls

1. **Model not found**: `--models` scans directory for GGUF files. Check `general.architecture` metadata matches: `acestep-lm`, `acestep-dit`, `acestep-text-enc`, `acestep-vae`.
2. **FSM hard-masks**: If LM outputs invalid field names, check `metadata-fsm.h` prefix tree construction.
3. **VAE VRAM OOM**: Reduce `--vae-chunk` (default 256) to lower tile size.
4. **CUDA gridDim overflow**: Only happens in VAE with large output widths. Already patched in `ggml-cuda/im2col.cu`.
5. **SDE vs ODE**: ODE Euler is deterministic (same seed = same result). SDE Stochastic uses fresh Philox noise at each step (varied results). Default is ODE.

## Model Files

### GGUF Locations

- **LM**: `models/acestep-5Hz-lm-*.gguf` (0.6B, 1.7B, 4B variants)
- **Text encoder**: `models/Qwen3-Embedding-0.6B-*.gguf`
- **DiT**: `models/acestep-v15-*.gguf` (turbo, sft, base, xl-* variants)
- **VAE**: `models/vae-BF16.gguf` (always BF16, small, bandwidth-bound)

### Downloading

```bash
./models.sh              # Q8_0 turbo essentials (~7.7 GB)
./models.sh --all        # All models, all quants (~97 GB)
./models.sh --quant Q6_K # Specific quantization
./models.sh --sft        # Add SFT DiT variant
./models.sh --shifts     # Add shift1/shift3/continuous variants
```

Requires `pip install hf` (HuggingFace CLI).

### LoRA Support

- **Formats**: PEFT directories (`adapter_model.safetensors` + `adapter_config.json`) or ComfyUI single `.safetensors`
- **Merging**: LoRA deltas merged into DiT weights at load time (static merge, no runtime overhead)
- **Location**: `--loras <dir>` scans directory; select via `lora` field in JSON

## WebUI

- **Location**: `tools/webui/` (Svelte + TypeScript + Vite)
- **Build**: `cd tools/webui && npm install && npm run build`
- **Embedding**: Output `tools/webui/public/index.html` is gzipped to `tools/public/index.html.gz`, then embedded into `build/index.html.gz.hpp` at CMake time.
- **Update flow**: Edit WebUI → build with npm → gzip → rebuild ace-server (CMake regenerates header)

## Language & Tools

- **C++17**: Standard C++17 but no modern features (no classes, no exceptions)
- **JSON**: `yyjson` (vendor/yyjson/) - fast, header-only
- **HTTP**: `cpp-httplib` (vendor/cpp-httplib/) - single-header HTTP server
- **Audio**: Custom WAV reader, minimp3 decoder (CC0), custom MP3 encoder (MIT)
- **RNG**: Philox (same as ACE-Step Python reference)
- **Build**: CMake 3.14+, GCC/Clang/MSVC

## Key Files to Read

| File | Purpose |
|------|---------|
| `src/task-types.h` | Task constants, instructions for LM/DiT |
| `src/request.h` | AceRequest struct (API contract) |
| `src/pipeline-lm.h` | LLM pipeline interface |
| `src/pipeline-synth.h` | DiT+VAE pipeline interface |
| `src/pipeline-understand.h` | Reverse pipeline interface |
| `src/qwen3-lm.h` | Qwen3 LM architecture (partial head) |
| `src/qwen3-enc.h` | Qwen3 text encoder |
| `src/dit.h` | DiT architecture |
| `src/vae.h` | AutoencoderOobleck VAE |
| `src/fsq-tok.h` / `fsq-detok.h` | FSQ tokenizer/detokenizer |
| `src/lora-merge.h` | LoRA merging logic |
| `tools/ace-server.cpp` | HTTP server, job queue, model registry |
| `CMakeLists.txt` | Build configuration, backend detection |

## Agent Workflow Tips

1. **Never edit `ggml/`** unless adding a new GGML op. The submodule is patched but stable.
2. **Read `docs/ARCHITECTURE.md`** for complete API reference and task type details.
3. **Check `examples/`** for working JSON + shell script pairs before implementing new features.
4. **Format before committing**: Run `./format.sh` (clang-format with `.clang-format` config).
5. **Test with CPU first**: `./buildcpu.sh` builds fastest, no GPU needed for basic testing.
6. **Model loading is lazy**: Server starts with zero GPU. Models load on first request.
7. **Seed behavior**: `seed=-1` is resolved once per input JSON, not per batch element.
8. **Task type detection**: Empty `task_type` auto-detects from data (audio_codes present = cover, src-audio present = cover, etc.).

## License & Attribution

- **acestep.cpp**: Independent C++ implementation
- **Models**: ACE Studio and StepFun (see `LICENSE`)
- **GGML**: Modified upstream with custom ops
- **MP3 encoder**: MIT license
- **minimp3 decoder**: CC0

**Key paper**: "ACE-Step 1.5: Pushing the Boundaries of Open-Source Music Generation" (Gong et al., 2026).

</content>