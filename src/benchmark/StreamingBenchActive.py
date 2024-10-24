import tqdm
import os
import time
import json
from utils.data_execution import get_model_response
from utils.video_execution import split_video

from benchmark.Benchmark import Benchmark

PROMPT_TEMPLATE_ACTIVE = '''You are an advanced image question-answering AI assistant. You have been provided with images and a question related to the images. Your task is to carefully analyze the images and provide the answer to the question. You need to carefully confirm whether the images content meet the conditions of the question, and then output the correct content.

Question: {}

The answer is:
'''

class StreamingBenchActive(Benchmark):
    def __init__(self, data):
        StreamingBenchActiveInit(data)

    def eval(self, data, model):
        StreamingBenchActiveEval(data, model)

def StreamingBenchActiveInit(data):
    pass

def StreamingBenchActiveEval(data, MODEL, output_path):
    for subset in tqdm.tqdm(data):
        for question in subset["questions"]:
            if MODEL.name() in question and question[MODEL.name()]['dialog_history'][-1]['content']:
                continue

            video_path = subset["video_path"]
            timestamp = question["time_stamp"]
            ground_truth_timestamp = question["ground_truth_time_stamp"]

            # convert timestamps like "00:03:10" to seconds
            start_time = sum(int(x) * 60 ** i for i, x in enumerate(reversed(timestamp.split(":"))))
            ground_truth_time = sum(int(x) * 60 ** i for i, x in enumerate(reversed(ground_truth_timestamp.split(":"))))
            max_time = ground_truth_time + 4  # Maximum polling time: ground truth + 4 seconds

            dialog_history = []
            answered = False
            # Prepare input for the model
            query = f"{question['question']} Is it the right time to output \"{question['ground_truth_output']}\"? You can only answer yes or no."
            inp = PROMPT_TEMPLATE_ACTIVE.format(query)

            current_time = start_time + 1

            while current_time <= max_time:

                interval = 1

                if current_time == 0:
                    current_time += 1

                clip_file = split_video(video_path, current_time)

                # Model inference
                time_s = time.time()
                response = get_model_response(MODEL, clip_file, inp)
                time_e = time.time()
                timecost = time_e - time_s

                # Record the interaction
                dialog_history.append({
                    'role': 'user', 'content': query, 'time': current_time, 'cost': timecost
                })
                dialog_history.append({
                    'role': 'assistant', 'content': response, 'time': current_time, 'cost': timecost
                })

                if 'yes' in response.strip().lower():
                    inp = PROMPT_TEMPLATE_ACTIVE.format(question['question'])
                    time_s = time.time()
                    response = get_model_response(MODEL, clip_file, inp)
                    time_e = time.time()
                    timecost = time_e - time_s

                    # Record the interaction
                    dialog_history.append({
                        'role': 'user', 'content': question['question'], 'time': current_time, 'cost': timecost
                    })
                    dialog_history.append({
                        'role': 'assistant', 'content': response, 'time': current_time, 'cost': timecost
                    })

                    answered = current_time
                    break

                current_time += interval

            question[MODEL.name()] = {
                "answered": answered,
                "dialog_history": dialog_history
            }

            with open(output_path, "w") as f:
                json.dump(data, f, indent=4)