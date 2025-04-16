# NZBench
Benchmarking the Future of Computing

**NZBench** is a Python framework designed to benchmark the system-level performance of language models (LLMs) using various workloads.

## Installation

```bash
git clone https://github.com/devcow85/nzbench.git
cd nzbench

# (Optional) Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install as an editable package
pip install -e .
```

## Usage

```bash
nzbench --model <model_name> --benchmark <benchmark_name> --output_fd <output_file.json>
```

### Arguments

| Argument       | Description                               | Example                                     |
|----------------|-------------------------------------------|---------------------------------------------|
| `--model`      | Huggingface model name (required)         | `meta-llama/Llama-3.2-3B-Instruct`          |
| `--benchmark`  | Benchmark name (`mmlu` or `mmlu_pro`)     | `mmlu`                                      |
| `--output_fd`  | Output file name to save results (JSON)   | `result.json`                               |

---

### Example

```bash
nzbench \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --benchmark mmlu \
  --output_fd result.json
```

## Output

- The benchmark results are saved to the JSON file specified in `--output_fd`.
- Each result contains information like:

```json
{
  "question": "...",
  "response": "...",
  "answer": "C",
  "subject": "physics"
}
```

> Support for accuracy, timing, and profiling metrics can be added in future versions.
