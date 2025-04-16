from nzbench.benchmark import NZBenchmark
import argparse

def main():
    print("Welcome to NZBench Framework")

    parser = argparse.ArgumentParser(description="NZBench CLI")
    parser.add_argument("--benchmark", type=str, default="mmlu", help="Benchmark name (mmlu or mmlu_pro)")
    parser.add_argument("--model", type=str, required=True, help="Huggingface Model name")
    parser.add_argument("--output_fd", type=str, required=True, help="output file name")
    args = parser.parse_args()

    nzb = NZBenchmark(benchmark=args.benchmark, model=args.model, output_fd=args.output_fd)
    nzb.run()

if __name__ == "__main__":
    main()
