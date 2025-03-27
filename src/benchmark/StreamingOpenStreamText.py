from tqdm import tqdm
import os
import json
from utils.data_execution import get_model_response_text_stream
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

class StreamingOpenStreamText(Benchmark):
    def __init__(self, data):
        StreamingOpenStreamTextInit(data)

    def eval(self, data, model, output_path, context_time):
        StreamingOpenStreamTextEval(data, model, output_path, context_time)

def StreamingOpenStreamTextInit(data):
    pass

def StreamingOpenStreamTextEval(data, MODEL, output_path, context_time):
    for subset in tqdm(data):
        isBegin = True
        prev_time = 0
        for question in subset["questions"]:
            if MODEL.name() in question:
                continue

            video_path = subset["video_path"]
            timestamp = question["time_stamp"]
            timestamp = sum(int(x) * 60 ** i for i, x in enumerate(reversed(timestamp.split(":"))))
            file = split_video(video_path, max(0, prev_time), timestamp)
            prev_time = timestamp
            ques = question["question"]
            audio_path = question["audio_path"]
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
            if isBegin:
                response, results = get_model_response_text_stream(MODEL, file, audio_path, '123', isBegin, inp)
                isBegin = False
            else:
                response, results= get_model_response_text_stream(MODEL, file, audio_path, '123', isBegin, inp)

            question[MODEL.name()] = response
            question['results'] = results

            with open(output_path, "w") as f:
                json.dump(data, f, indent=4)
