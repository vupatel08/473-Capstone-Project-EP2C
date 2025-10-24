## Parsing

Parsing with MinerU requires the following:

---

## Backend Options

| Feature | `pipeline` | `vlm-transformers` | `vlm-vllm` |
|---------|------------|------------------|------------|
| **Operating System** | Linux / Windows / macOS | Linux / Windows | Linux / Windows (via WSL2) |
| **CPU Inference Support** | ✅ | ❌ | ❌ |
| **GPU Requirements** | Turing architecture and later, 6GB+ VRAM or Apple Silicon | Turing architecture and later, 8GB+ VRAM | Turing architecture and later, 8GB+ VRAM |
| **Memory Requirements** | Minimum 16GB+, recommended 32GB+ | Minimum 16GB+, recommended 32GB+ | Minimum 16GB+, recommended 32GB+ |
| **Disk Space Requirements** | 20GB+, SSD recommended | 20GB+, SSD recommended | 20GB+, SSD recommended |
| **Python Version** | 3.10–3.13 | 3.10–3.13 | 3.10–3.13 |

---

## Recommendations

- **Memory:** At least 16GB of RAM is required; 32GB+ is recommended for optimal performance.  
- **Disk:** SSDs are recommended to improve loading times.  
- **GPU:** Turing architecture or newer is required for GPU inference. Apple Silicon is supported for CPU inference with `pipeline`.  
