from model.modelclass import Model

class VideoLLaMA2(Model):
    def __init__(self):
        VideoLLaMA2_Init()

    def Run(self, file, inp):
        return VideoLLaMA2_Run(file, inp)
    
    def name(self):
        return "VideoLLaMA2"

from videollama2 import model_init, mm_infer

model, processor, tokenizer = None, None, None

def VideoLLaMA2_Init():
    global model, processor, tokenizer

    # 1. Initialize the model.
    model_path = 'DAMO-NLP-SG/VideoLLaMA2-7B'
    # Base model inference (only need to replace model_path)
    # model_path = 'DAMO-NLP-SG/VideoLLaMA2-7B-Base'
    model, processor, tokenizer = model_init(model_path)

def VideoLLaMA2_Run(file, inp):
    # Video Inference
    # Reply:
    # The video features a kitten and a baby chick playing together. The kitten is seen laying on the floor while the baby chick hops around. The two animals interact playfully with each other, and the video has a cute and heartwarming feel to it.
    modal = 'video'

    # 2. Visual preprocess (load & transform image or video).
    output = mm_infer(processor[modal](file), inp, model=model, tokenizer=tokenizer, do_sample=False, modal=modal)

    return output[0]