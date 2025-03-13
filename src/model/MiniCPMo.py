import os
import torch
from transformers import AutoModel, AutoTokenizer

import numpy as np
import librosa
from PIL import Image
import tempfile
import ffmpeg
from model.modelclass import Model

def get_frame_at_time(file, time, width, height):
    if time < 0:
        out, _ = (
            ffmpeg
            .input(file, sseof=time)
            .output('pipe:', vframes=1, format='rawvideo', pix_fmt='rgb24')
            .run(capture_stdout=True, capture_stderr=True)
        )
    else:
        out, _ = (
            ffmpeg
            .input(file, ss=time)
            .output('pipe:', vframes=1, format='rawvideo', pix_fmt='rgb24')
            .run(capture_stdout=True, capture_stderr=True)
        )
    if len(out) == 0:
        raise ValueError(f"No frame data extracted at time {time}s.")
    frame = np.frombuffer(out, np.uint8).reshape([height, width, 3])
    return frame

class MiniCPMo(Model):
    def __init__(self):
        self.MiniCPMo_Init()

    def Run(self, file, inp):
        return self.MiniCPMo_Run(file, inp)
    
    def name(self):
        return "MiniCPMo"

    def MiniCPMo_Init(self):
        self.model = AutoModel.from_pretrained(
            'OpenBMB/MiniCPM-o-2_6',
            trust_remote_code=True,
            attn_implementation='sdpa',
            torch_dtype=torch.bfloat16,
            init_vision=True,
            init_audio=True,
            init_tts=True
        )
        self.model = self.model.eval().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained('OpenBMB/MiniCPM-o-2_6', trust_remote_code=True)
        self.model.init_tts()
        self.model.tts.float()


    def MiniCPMo_Run(self, file, inp):
        # 使用 ffmpeg-probe 获取视频信息
        probe = ffmpeg.probe(file)
        video_info = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        duration = float(video_info['duration'])
        width = int(video_info['width'])
        height = int(video_info['height'])
        print('video_duration:', duration)

        # 创建临时文件用于保存音频
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
            temp_audio_file_path = temp_audio_file.name
            (
                ffmpeg
                .input(file)
                .output(temp_audio_file_path, acodec='pcm_s16le', ar='16000', vn=None)
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )

        # 加载音频
        audio_np, sr = librosa.load(temp_audio_file_path, sr=16000, mono=True)
        os.remove(temp_audio_file_path)  # 删除临时音频文件

        num_units = int(duration)
        # interval = 2 # 0.5 fps
        cnts= []
        for i in range(0, num_units):
            time = i + 1  # 获取时间点
            frame = get_frame_at_time(file, time-num_units-0.1, width, height)
            # print('frame:', frame.shape)
            image = Image.fromarray(frame.astype(np.uint8))
            # print(image.size)

            audio = audio_np[sr*i: sr*(i+1)]
            cnts += ["<unit>", image, audio]

        msg = {"role":"user", "content": cnts + [inp]}
        msgs = [msg]

        res = self.model.chat(
            image=None, 
            msgs=msgs, 
            context=None,
            tokenizer=self.tokenizer,
            sampling=False,
            max_new_tokens=4096,
            stream=False,
            stream_input=True,
            omni_input=True,
            use_tts=True,
            max_slice_nums=1,   # set 1 when stream_input=True
            use_image_id=False, # set False when stream_input=True
        )
        return res

