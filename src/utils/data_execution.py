import json

def get_timestamps(question_set):
    """
    """
    timestamps = []
    
    for question in question_set["questions"]:
        timestamps.append(question['time_stamp'])

    return timestamps

def load_data(EVAL_DATA_FILE):
    with open(EVAL_DATA_FILE, "r") as f:
        data = json.load(f)
    
    return data

def get_model_response(model, file, inp):
    """
    Get the model response for the given input
    Model: Model name
    file: Video file path
    inp: Input prompt
    """
    return model.Run(file, inp)


def get_model_response_text_stream(model, file, audio, session, isBegin, inp):
    """
    Get the model response for the given input
    Model: Model name
    file: Video file path
    audio: Audio instruction file path
    session: Streaming session
    isBegin: Whether the first query or not
    inp: Input prompt
    """
    return model.Run_Text_Stream(file, audio, session, isBegin, inp)
