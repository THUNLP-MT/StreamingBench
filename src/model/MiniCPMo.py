import torch
from PIL import Image
from modelscope import AutoModel, AutoTokenizer
import math
import numpy as np
from PIL import Image
from moviepy.editor import VideoFileClip
import tempfile
import librosa
import soundfile as sf

import time
import psutil
import torch
from PIL import Image
from modelscope import AutoModel, AutoTokenizer
import math
import numpy as np
from PIL import Image
from moviepy.editor import VideoFileClip
import tempfile
import librosa
import soundfile as sf
import logging



# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_video_chunk_content(video_path, new_audio_path=None, flatten=False):
    try:
        video = VideoFileClip(video_path)
        logging.info(f'video_duration: {video.duration}')

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio_file:
            temp_audio_file_path = temp_audio_file.name
            try:
                video.audio.write_audiofile(temp_audio_file_path, codec="pcm_s16le", fps=16000)
            except Exception as e:
                logging.error(f"Error writing audio file: {e}", exc_info=True)
                return []

            try:
                audio_np, sr = librosa.load(temp_audio_file_path, sr=16000, mono=True)
            except Exception as e:
                logging.error(f"Error loading audio with librosa: {e}", exc_info=True)
                return []

        # --- Main Change:  Sampling 10 frames evenly ---
        num_frames = 60
        frame_times = np.linspace(0, video.duration - 0.05, num_frames)  # Subtract small value to avoid end
        contents = []

        for t in frame_times:
            try:
                frame = video.get_frame(t)
                image = Image.fromarray(frame.astype(np.uint8))

                # Calculate corresponding audio segment
                start_sample = int(t * sr)
                #  Crucially important:  Ensure we don't go past the end of the audio
                end_sample = min(start_sample + int((video.duration / num_frames) * sr), len(audio_np))
                audio = audio_np[start_sample:end_sample]

                if flatten:
                    contents.extend(["<unit>", image, audio])
                else:
                    contents.append(["<unit>", image, audio])
            except Exception as e:
                logging.warning(f"Error processing frame/audio chunk at time {t}: {e}.  Continuing...", exc_info=True)
                continue  # Skip this chunk


        # Add new audio chunk (same as before, but with improved error handling)
        if new_audio_path:
            try:
                new_audio_np, new_sr = librosa.load(new_audio_path, sr=16000, mono=True)
                # Get the last frame of the video
                try:
                    last_frame = video.get_frame(video.duration - 0.05)  # Safer
                    last_image = Image.fromarray(last_frame.astype(np.uint8))
                except Exception as e:
                    logging.error(f"Error getting last frame: {e}", exc_info=True)
                    return []  # Or handle differently

                new_audio_duration = librosa.get_duration(y=new_audio_np, sr=new_sr)
                new_audio_num_units = math.ceil(new_audio_duration)

                for i in range(new_audio_num_units):
                    audio_chunk = new_audio_np[new_sr * i:new_sr * (i + 1)]
                    if flatten:
                        contents.extend(["<unit>", last_image, audio_chunk])
                    else:
                        contents.append(["<unit>", last_image, audio_chunk])
            except Exception as e:
                logging.error(f"Error processing new audio: {e}", exc_info=True)
                return []

        return contents

    except OSError as e:
        logging.error(f"MoviePy/FFmpeg OSError: {e}.  Check file and FFmpeg installation.", exc_info=True)
        return []
    except (RuntimeError, ValueError) as e:
        logging.error(f"Librosa Error: {e}", exc_info=True)
        return []
    except Exception as e:
        logging.error(f"An unexpected error occurred in get_video_chunk_content: {e}", exc_info=True)
        return []
    finally:
        try:
            video.close()
        except Exception as e:
            logging.error(f"Error closing video file: {e}", exc_info=True)


class MiniCPMo():
    def __init__(self):
        self.MiniCPMo_Init()

    def Run_Text_Stream(self, file, new_audio_path, session_id, isBegin, inp):
        return self.MiniCPM_StreamRunText(file, new_audio_path, session_id, isBegin, inp)
        
    def Run(self, file, inp):
        return self.MiniCPMo_TextRun(file, inp)
    
    def name(self):
        return "MiniCPMo"

    def release_memory(self):
        if torch.cuda.is_available():
            if hasattr(self, 'model') and self.model is not None:
                self.model.cpu()  # Move model to CPU before deleting
        if hasattr(self, 'model') and self.model is not None:
            del self.model
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            del self.tokenizer
        if hasattr(self, "model") and hasattr(self.model, 'tts') and self.model.tts is not None:
            del self.model.tts
            self.model.tts = None
        self.model = None
        self.tokenizer = None
        import gc
        gc.collect()


    def reload_model(self):
        self.MiniCPMo_Init()

    def MiniCPMo_Init(self):
        # load omni model default, the default init_vision/init_audio/init_tts is True
        # if load vision-only model, please set init_audio=False and init_tts=False
        # if load audio-only model, please set init_vision=False
        self.model = AutoModel.from_pretrained(
            'OpenBMB/MiniCPM-o-2_6',
            trust_remote_code=True,
            attn_implementation='sdpa', # sdpa or flash_attention_2
            torch_dtype=torch.bfloat16,
            init_vision=True,
            init_audio=True,
            init_tts=True
        )
        self.model = self.model.eval().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained('OpenBMB/MiniCPM-o-2_6', trust_remote_code=True)

        # In addition to vision-only mode, tts processor and vocos also needs to be initialized
        self.model.init_tts()
        self.model.tts.float()

    def MiniCPMo_TextRun(self, file, new_audio_path, session_id, isBegin, inp):
        try:
            # Removed redundant model loading (use self.model and self.tokenizer)

            # if use voice clone prompt, please set ref_audio
            # ref_audio_path = 'assets/demo.wav'
            # ref_audio, _ = librosa.load(ref_audio_path, sr=16000, mono=True)
            # sys_msg = model.get_sys_prompt(ref_audio=ref_audio, mode='omni', language='en')
            # or use default prompt
            sys_msg = self.model.get_sys_prompt(mode=None, language='en')

            contents = get_video_chunk_content(file, None)
            if not contents:  # Check if get_video_chunk_content returned an empty list (error)
                return {"error": "Failed to process video or audio"}
            msg = {"role": "user", "content": contents + [inp]}
            msgs = [sys_msg, msg]

            # please set generate_audio=True and output_audio_path to save the tts result
            generate_audio = False
            output_audio_path = 'output.wav'

            res = self.model.chat(
                msgs=msgs,
                tokenizer=self.tokenizer,
                sampling=True,
                temperature=0.5,
                max_new_tokens=4096,
                omni_input=True,  # please set omni_input=True when omni inference
                use_tts_template=True,
                generate_audio=generate_audio,
                output_audio_path=output_audio_path,
                max_slice_nums=1,
                use_image_id=False,
                return_dict=True
            )
            print(res)
            return res

        except Exception as e:
            logging.error(f"An error occurred in MiniCPMo_Run: {e}", exc_info=True)
            return {"error": str(e)}


    def MiniCPM_StreamRunText(self, file, new_audio_path, session_id, isBegin, inp):
        video_path = file
        generate_audio = False
        sys_msg = self.model.get_sys_prompt(mode=None, language='en')
        # 1. prefill system prompt
        if isBegin:
            self.model.reset_session()
            contents = get_video_chunk_content(video_path, None, False)
            res = self.model.streaming_prefill(
                session_id=session_id,
                msgs=[sys_msg], 
                tokenizer=self.tokenizer
            )
        else:
            contents = get_video_chunk_content(video_path, None, False)
        # 2. prefill video/audio chunks
        total_prefill_time = 0
        for content in contents:
            msgs = [{"role":"user", "content": content}]
            try:
                prefill_time = self.model.streaming_prefill(
                    session_id=session_id,
                    msgs=msgs, 
                    tokenizer=self.tokenizer
                )
                total_prefill_time += prefill_time
            except AssertionError:
                print(f"AssertionError occurred during prefill for content: {content}. Skipping this chunk.")
                continue ß

        msgs = [{"role":"user", "content": inp}]
        try:
            prefill_time = self.model.streaming_prefill(
                    session_id=session_id,
                    msgs=msgs, 
                    tokenizer=self.tokenizer
                )
            total_prefill_time += prefill_time
        except AssertionError:
            print(f"AssertionError occurred during prefill for input: {inp}. Skipping this input.")
            total_prefill_time += 0

        # 3. generate
        try:
            res, metrics = self.model.streaming_generate(
                session_id=session_id,
                tokenizer=self.tokenizer,
                temperature=0.5,
                generate_audio=generate_audio,
                prefill_time = total_prefill_time,
            )
            text = ""
            for r in res:
                text += r['text']
            print("text:", text)
            return text, metrics
        except Exception as e: 
            print(f"An error occurred during generation:{e}")
            return "",None
   