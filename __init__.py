import os
import sys

import torchaudio
import torch
import numpy as np
import librosa

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
sys.path.append(f"{ROOT_DIR}/CosyVoice")
sys.path.insert(0, f"{ROOT_DIR}/CosyVoice/third_party/Matcha-TTS")

from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
import folder_paths

models_root = os.path.join(folder_paths.get_folder_paths("cosyvoice")[0],"pretrained_models")

GLOBAL_CATEGORY = "HJH_CosyVoice🪅"

class CosyVoiceModel:
    @classmethod
    def INPUT_TYPES(cls):
        """定义输入参数"""
        return {
            "required": {
                "model":([
                    "CosyVoice2-0.5B",
                    "CosyVoice-300M-SFT",
                    "CosyVoice-300M",
                    "CosyVoice-300M-25Hz",
                    "CosyVoice-300M-Instruct",
                ],),
                "fp16": ([False, True],),
            },
        }

    RETURN_TYPES = ("COSYVOICEMODEL",)
    RETURN_NAMES = ("cosyvoice_model",)
    FUNCTION = "load"
    CATEGORY = GLOBAL_CATEGORY

    def __init__(self):
        pass

    def load(self, model, fp16=False):
        if model == "CosyVoice2-0.5B":
            cosyvoice = CosyVoice2(os.path.join(models_root,"CosyVoice2-0.5B"), load_jit=False, load_trt=False, fp16=fp16)
        elif model == "CosyVoice-300M-SFT":
            cosyvoice = CosyVoice(os.path.join(models_root,"CosyVoice-300M-SFT"), load_jit=False, load_trt=False, fp16=fp16)
        elif model == "CosyVoice-300M":
            cosyvoice = CosyVoice(os.path.join(models_root,"CosyVoice-300M"), load_jit=False, load_trt=False, fp16=fp16)
        elif model == "CosyVoice-300M-25Hz":
            cosyvoice = CosyVoice(os.path.join(models_root,"CosyVoice-300M-25Hz"), load_jit=False, load_trt=False, fp16=fp16)
        elif model == "CosyVoice-300M-Instruct":
            cosyvoice = CosyVoice(os.path.join(models_root,"CosyVoice-300M-Instruct"), load_jit=False, load_trt=False, fp16=fp16)

        return cosyvoice,


max_val = 0.8
target_sr = 16000
def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    speech = torch.concat([speech, torch.zeros(1, int(target_sr * 0.2))], dim=1)
    return speech

def get_output_data(generator):
    chunks = []
    for chunk in generator:
        chunks.append(chunk['tts_speech'].numpy().flatten())
    output = np.array(chunks)
    return torch.from_numpy(output).unsqueeze(0)

def audio_prepare(audio):
    waveform = audio['waveform'].squeeze(0)
    source_sr = audio['sample_rate']
    speech = waveform.mean(dim=0,keepdim=True)
    if source_sr != target_sr:
        speech = torchaudio.transforms.Resample(orig_freq=source_sr, new_freq=target_sr)(speech)

    return postprocess(speech)


class CosyVoiceNode:
    @classmethod
    def INPUT_TYPES(cls):
        """定义输入参数"""
        return {
            "required": {
                "cosyvoice_model": ("COSYVOICEMODEL",),
                "mode":(["Zero Shot", "Cross Lingual", "Instruct2","SFT", "VC", "Instruct"],),
                "tts_text": ("STRING", {"default":""}),
                "speed":("FLOAT",{"min":0.1,"max":2.0,"default":1.0}),
                "text_frontend":([True,False],),
            },
            "optional":{
                "prompt_audio": ("AUDIO",),
                "source_audio": ("AUDIO",),
                "prompt_text": ("STRING", {"default":""}),
                "instruct_text": ("STRING", {"default":""}),
                "spk_id": (['中文女', '中文男', '日语男', '粤语女', '英文女', '英文男', '韩语女'],),
            },
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("generated_audio",)
    FUNCTION = "generate"
    CATEGORY = GLOBAL_CATEGORY

    OUTPUT_NODE = True

    def __init__(self):
        pass

    def generate(self, cosyvoice_model, mode, tts_text, prompt_audio=None,source_audio=None, prompt_text="",instruct_text="",spk_id="",speed=1.0,text_frontend=False):
        cosyvoice = cosyvoice_model

        is_CosyVoice2 = isinstance(cosyvoice, CosyVoice2)

        # prompt_speech_16k = prompt_wav["waveform"].squeeze(0)

        if prompt_audio is not None:
            prompt_speech_16k = audio_prepare(prompt_audio)

        if source_audio is not None:
            source_speech_16k = audio_prepare(source_audio)

        generator = None
        if mode == "Zero Shot":
            '''
            cosyvoice.inference_zero_shot params:
                tts_text: 需要合成语音的文本。
                prompt_text: 提供的提示文本，用于辅助生成语音。
                prompt_speech_16k: 提供的提示语音，采样率为16kHz，用于辅助生成语音。
                stream: 布尔值，表示是否以流式方式输出语音。如果设置为 True，则函数会逐块生成并返回语音数据。
                speed: 控制生成语音的语速，1.0表示正常语速。默认: 1.0
                text_frontend: 布尔值，表示是否使用前端文本处理模块。默认: True
            '''
            if is_CosyVoice2:
                generator = cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, speed=speed, stream=False,text_frontend=text_frontend)
            else:
                generator = cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, stream=False,)

        elif mode == "Cross Lingual":
            '''
            tts_text, prompt_speech_16k, stream=False, speed=1.0, text_frontend=True
            标记参考: custom_nodes/ComfyUI_CosyVoice2/CosyVoice/cosyvoice/tokenizer/tokenizer.py
            '''
            if is_CosyVoice2:
                generator = cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k, speed=speed, stream=False,text_frontend=text_frontend)
            else:
                generator = cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k, stream=False,)
                
        elif mode == "Instruct2":
            '''
            inference_instruct2 params:
                tts_text: 需要转换为语音的文本。
                instruct_text: 指令文本，用于引导语音合成的过程。
                prompt_speech_16k: 一个参考语音信号，采样率为16kHz。
                stream: 是否以流的形式输出语音，默认为False。
                speed: 生成语音的速度，1.0表示正常速度。
                text_frontend: 是否使用文本前端处理，默认为True。
            '''
            generator = cosyvoice.inference_instruct2(tts_text, instruct_text, prompt_speech_16k, speed=speed, stream=False,text_frontend=text_frontend)
        elif mode == "SFT":
            generator = cosyvoice.inference_sft(tts_text, spk_id, stream=False)
        elif mode == "VC":
            generator = cosyvoice.inference_vc(source_speech_16k, prompt_speech_16k, stream=False)
        elif mode == "Instruct":
            generator = cosyvoice.inference_instruct(tts_text, spk_id, instruct_text, stream=False)

        if generator is not None:
            output_audio = get_output_data(generator)
            output_audio = {"waveform":output_audio, "sample_rate": cosyvoice.sample_rate}
        
        return output_audio, 

# from comfy_extras.nodes_audio import insert_or_replace_vorbis_comment
from comfy_extras.nodes_audio import SaveAudio as SA
class HJHCosyVoiceSaveAudio(SA):
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    OUTPUT_IS_LIST =(True,)

    FUNCTION = "save"
    # def __init__(self):
    #     super().__init__()

    def save(self, audio, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        # full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir)

        result = super().save_audio(audio, filename_prefix, prompt, extra_pnginfo)
        # results.append({
        #     "filename": file,
        #     "subfolder": subfolder,
        #     "type": self.type
        # })
        ui_result = result["ui"]["audio"]
        paths = []
        for i in range(len(ui_result)):
            path = os.path.join(folder_paths.get_output_directory(),ui_result[i]["subfolder"], ui_result[i]["filename"])
            paths.append(path)

        result["result"] = paths,
        print("******************",result)
        return result


# Add custom API routes, using router
# from aiohttp import web
# from server import PromptServer

# @PromptServer.instance.routes.get("/hello")
# async def get_hello(request):
#     return web.json_response("hello")



NODE_CLASS_MAPPINGS = {
    "CosyVoiceModel": CosyVoiceModel,
    "CosyVoiceNode": CosyVoiceNode,
    "HJHCosyVoiceSaveAudio": HJHCosyVoiceSaveAudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CosyVoiceModel": "HJH-CosyVoice - Load Model",
    "CosyVoiceNode": "HJH-CosyVoice - Generate Audio",
    "HJHCosyVoiceSaveAudio": "HJH-CosyVoice - Save Audio",
}