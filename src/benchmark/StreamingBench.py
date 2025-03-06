from tqdm import tqdm
import os
import json
from utils.data_execution import get_model_response
from utils.video_execution import split_video

from benchmark.Benchmark import Benchmark

PROMPT_TEMPLATE = '''You are an advanced video question-answering AI assistant. You have been provided with some frames from the video and a multiple-choice question related to the video. Your task is to carefully analyze the video and provide the best answer to question, choosing from the four options provided. Respond with only the letter (A, B, C, or D) of the correct option.

Question: {}

Options:
{}
{}
{}
{}'''

PROMPT_TEMPLATE_WITHOUT_OPTIONS = '''You are an advanced video question-answering AI assistant. You have been provided with a video and a question related to the video. Your task is to carefully analyze the video and provide the answer to the question. 

Question: {}
'''

class StreamingBench(Benchmark):
    def __init__(self, data):
        StreamingBenchInit(data)

    def eval(self, data, model, output_path, context_time):
        StreamingBenchEval(data, model, output_path, context_time)

def StreamingBenchInit(data):
    pass

def StreamingBenchEval(data, MODEL, output_path, context_time):
    for subset in tqdm(data):
        for question in subset["questions"]:
            if MODEL.name() in question and question[MODEL.name()]:
                continue

            video_path = subset["video_path"]
            timestamp = question["time_stamp"]
            # convert timestamps like "00:03:10" to seconds
            timestamp = sum(int(x) * 60 ** i for i, x in enumerate(reversed(timestamp.split(":"))))

            if context_time > 0:
                time_start = max(0, timestamp - context_time)
            else:
                time_start = 0

            file = split_video(video_path, time_start, timestamp)

            ques = question["question"]
            if "options" in question.keys():
                options = question["options"]
                if not options[0].startswith("A."):
                    options = [f"A. {options[0]}", f"B. {options[1]}", f"C. {options[2]}", f"D. {options[3]}"]

                inp = PROMPT_TEMPLATE.format(ques, *options)
                inp += "\n\nThe best option is:"
            else:
                inp = PROMPT_TEMPLATE_WITHOUT_OPTIONS.format(ques)
                inp += "\n\nAnswer:"

            print(f"input: {inp}")

            response = get_model_response(MODEL, file, inp)
            question[MODEL.name()] = response

            with open(output_path, "w") as f:
                json.dump(data, f, indent=4)

            # remove the clip file
            # os.remove(file)