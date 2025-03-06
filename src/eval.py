from utils.data_execution import load_data

from model.modelclass import Model
from benchmark.Benchmark import Benchmark

import argparse

def main(args):
    data = load_data(args.data_file)

    ####### BENCHMARK #######

    benchmark = Benchmark(data)

    if args.benchmark_name == "Streaming":
        from benchmark.StreamingBench import StreamingBench
        benchmark = StreamingBench(data)
    if args.benchmark_name == "StreamingProactive":
        from benchmark.StreamingBenchProactive import StreamingBenchProactive
        benchmark = StreamingBenchProactive(data)
    if args.benchmark_name == "StreamingSQA":
        from benchmark.StreamingBenchSQA import StreamingBenchSQA
        benchmark = StreamingBenchSQA(data)
    if args.benchmark_name == "StreamingOpenStreamText":
        from benchmark.StreamingOpenStreamText import StreamingOpenStreamText
        benchmark = StreamingOpenStreamText(data)

    ##########################

    ####### MODEL ############

    model = Model()
 
    if args.model_name == "MiniCPM-V":
        from model.MiniCPMV import MiniCPMV
        model = MiniCPMV()
    if args.model_name == "MiniCPMo":
        print("MiniCPMo loaded")
        from model.MiniCPMo import MiniCPMo
        model = MiniCPMo()

    ######################

    benchmark.eval(data, model, args.output_file, args.context_time)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, required=True, help="Path to the data file")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model")
    parser.add_argument("--benchmark_name", type=str, required=True, help="Name of the benchmark")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output file")
    parser.add_argument("--context_time", type=int, required=True, help="Time before the query")
    args = parser.parse_args()
    main(args)
