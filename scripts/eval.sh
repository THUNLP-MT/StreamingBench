cd ../src

# Change the model name to the model you want to evaluate

EVAL_MODEL="MiniCPM-V"
Devices=0

# For real-time visual understanding(Offline + Text Instruction)

TASK="real"
DATA_FILE="./data/questions_${TASK}.json"
OUTPUT_FILE="./data/${TASK}_output_${EVAL_MODEL}.json"
BENCHMARK="Streaming"

if [ "$EVAL_MODEL" = "MiniCPM-V" ]; then
    conda activate MiniCPM-V
    CUDA_VISIBLE_DEVICES=$Devices python eval.py --model_name $EVAL_MODEL --benchmark_name $BENCHMARK --data_file $DATA_FILE --output_file $OUTPUT_FILE
fi

# For omni-source understanding(Offline + Text Instruction)

TASK="omni"
DATA_FILE="./data/questions_${TASK}.json"
OUTPUT_FILE="./data/${TASK}_output_${EVAL_MODEL}.json"
BENCHMARK="Streaming"

if [ "$EVAL_MODEL" = "MiniCPM-V" ]; then
    conda activate MiniCPM-V
    CUDA_VISIBLE_DEVICES=$Devices python eval.py --model_name $EVAL_MODEL --benchmark_name $BENCHMARK --data_file $DATA_FILE --output_file $OUTPUT_FILE
fi

# For sequential question answering(Offline + Text Instruction)

TASK="sqa"
DATA_FILE="./data/questions_${TASK}.json"
OUTPUT_FILE="./data/${TASK}_output_${EVAL_MODEL}.json"
BENCHMARK="StreamingSQA"

if [ "$EVAL_MODEL" = "MiniCPM-V" ]; then
    conda activate MiniCPM-V
    CUDA_VISIBLE_DEVICES=$Devices python eval.py --model_name $EVAL_MODEL --benchmark_name $BENCHMARK --data_file $DATA_FILE --output_file $OUTPUT_FILE
fi
# For proactive output(Offline + Text Instruction)

TASK="proactive"
DATA_FILE="./data/questions_${TASK}.json"
OUTPUT_FILE="./data/${TASK}_output_${EVAL_MODEL}.json"
BENCHMARK="StreamingProactive"

if [ "$EVAL_MODEL" = "MiniCPM-V" ]; then
    conda activate MiniCPM-V
    CUDA_VISIBLE_DEVICES=$Devices python eval.py --model_name $EVAL_MODEL --benchmark_name $BENCHMARK --data_file $DATA_FILE --output_file $OUTPUT_FILE
fi

# (Streaming/Online + Text Instruction)
# Optional Task(real, omni, sqa)

TASK="real"
DATA_FILE="./data/questions_${TASK}_stream.json"
OUTPUT_FILE="./data/${TASK}_text_stream_output_${EVAL_MODEL}.json"
BENCHMARK="StreamingOpenStreamText"

if [ "$EVAL_MODEL" = "MiniCPM-V" ]; then
    conda activate MiniCPM-V
    CUDA_VISIBLE_DEVICES=$Devices python eval.py --model_name $EVAL_MODEL --benchmark_name $BENCHMARK --data_file $DATA_FILE --output_file $OUTPUT_FILE
fi
