from base_handler import BaseHandler
from flask_socketio import emit

from time import perf_counter
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
)
import torch

from rich.console import Console
import logging

logger = logging.getLogger(__name__)
console = Console()

# distil-whisper/distil-large-v3 是 Whisper 模型的蒸馏版本, 使用下来只输出英文
# 使用下来openai/whisper-large-v3的效果最好，能够准确识别中英文
# SenseVoice 支持多语言，但是没有使用 transfomers 进行加载和管理，需要用funsr，支持多语言，同时支持输出情绪参数
# 但是容易识别错误成其他语言
class WhisperSTTHandler(BaseHandler):
    """
    Handles the Speech To Text generation using a Whisper model.
    ref :https://huggingface.co/openai/whisper-large-v3
    2024-09-28 use  whisper-large-v3-turbo
    """

    def setup(
            self,
            # model_name="openai/whisper-large-v3",
            model_name="ylacombe/whisper-large-v3-turbo",
            device="cuda:0",  
            torch_dtype="float16",  
            compile_mode=None,
            gen_kwargs={}
        ): 
        self.device = device
        self.torch_dtype = getattr(torch, torch_dtype)
        self.compile_mode=compile_mode
        self.gen_kwargs = gen_kwargs
        self.socketio = None  

        # 确定设备
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        logger.info("VAD will be asign to {self.device}")

        print("load the WhisperSTTmodel")
        
        self.processor = AutoProcessor.from_pretrained(model_name, clean_up_tokenization_spaces=True)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            torch_dtype=self.torch_dtype,
        ).to(self.device)
        
        # compile
        if self.compile_mode:
            self.model.generation_config.cache_implementation = "static"
            self.model.forward = torch.compile(self.model.forward, mode=self.compile_mode, fullgraph=True)
        self.warmup()
    
    
    def prepare_model_inputs(self, spoken_prompt):
        input_features = self.processor(
            spoken_prompt, sampling_rate=16000, return_tensors="pt"
        ).input_features
        input_features = input_features.to(self.device, dtype=self.torch_dtype)

        return input_features
        
    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")

        # 2 warmup steps for no compile or compile mode with CUDA graphs capture 
        n_steps = 1 if self.compile_mode == "default" else 2
        dummy_input = torch.randn(
            (1,  self.model.config.num_mel_bins, 3000),
            dtype=self.torch_dtype,
            device=self.device
        ) 
        if self.compile_mode not in (None, "default"):
            # generating more tokens than previously will trigger CUDA graphs capture
            # one should warmup with a number of generated tokens above max tokens targeted for subsequent generation
            warmup_gen_kwargs = {
                "min_new_tokens": self.gen_kwargs["max_new_tokens"],
                "max_new_tokens": self.gen_kwargs["max_new_tokens"],
                **self.gen_kwargs
            }
        else:
            warmup_gen_kwargs = self.gen_kwargs

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start_event.record()
        for _ in range(n_steps):
            _ = self.model.generate(dummy_input, **warmup_gen_kwargs)
        end_event.record()
        torch.cuda.synchronize()

        logger.info(f"{self.__class__.__name__}:  warmed up! time: {start_event.elapsed_time(end_event) * 1e-3:.3f} s")

    def set_socketio(self, socketio):
        self.socketio = socketio 

    def process(self, audio_data):
        logger.debug("infering whisper...")

        try:
            global pipeline_start
            pipeline_start = perf_counter()

            input_features = self.prepare_model_inputs(audio_data)
            pred_ids = self.model.generate(input_features, **self.gen_kwargs)
            pred_text = self.processor.batch_decode(
                pred_ids, 
                skip_special_tokens=True,
                decode_with_timestamps=False
            )[0]

            logger.debug("finished whisper inference")
            console.print(f"[yellow]USER: {pred_text}")
            # 发送 STT 结果到前端
            # 使用 socketio 发送结果到前端
            if self.socketio:
                self.socketio.emit('stt_result', {'text': pred_text})

            yield pred_text
        except Exception as e:
            logger.error(f"Error in WhisperSTTHandler: {e}",exc_info=True)
        finally:
            self.is_processing = False  # 处理完成
