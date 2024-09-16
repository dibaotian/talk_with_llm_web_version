from base_handler import BaseHandler
from flask_socketio import emit

import ChatTTS
import logging

import librosa
import numpy as np
from rich.console import Console
import torch

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

console = Console()


class ChatTTSHandler(BaseHandler):
    def setup(
        self,
        should_listen,
        device="cuda:1",
        gen_kwargs={},  # Unused
        stream=True,
        chunk_size=512,
    ):
        self.should_listen = should_listen
        self.device = device
        self.model = ChatTTS.Chat()
        self.model.load(compile=False)  # Doesn't work for me with True
        self.chunk_size = chunk_size
        self.stream = stream
        rnd_spk_emb = self.model.sample_random_speaker()
        self.params_infer_code = ChatTTS.Chat.InferCodeParams(
            spk_emb=rnd_spk_emb,
        )
        self.warmup()

    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")
        _ = self.model.infer("text hello world TTS")

    
    def set_socketio(self, socketio):
        self.socketio = socketio  # 添加这个方法

    def process(self, llm_sentence):
        
        # console.print(f"[green]ASSISTANT: {llm_sentence}")

        # 发送 LLM 结果到前端
        if self.socketio:
            self.socketio.emit('llm_response', {'text': llm_sentence})
        
        wavs_gen = self.model.infer(
            llm_sentence, params_infer_code=self.params_infer_code, stream=self.stream
        )

        if self.stream:
            wavs = [np.array([])]
            for gen in wavs_gen:
                if gen[0] is None or len(gen[0]) == 0:
                    self.should_listen.set()
                    return
                
                # audio_chunk = (gen[0] * 32768).astype(np.int16) 
                audio_chunk = librosa.resample(gen[0], orig_sr=24000, target_sr=16000)
                audio_chunk = (audio_chunk * 32768).astype(np.int16)

                

                #  # 确保 audio_chunk 是一个数组
                # if not isinstance(audio_chunk, np.ndarray):
                #     logger.warning(f"Unexpected audio_chunk type: {type(audio_chunk)}")
                #     audio_chunk = np.array([audio_chunk])  # 将标量转换为数组
                    
                while len(audio_chunk) > self.chunk_size:
                    yield audio_chunk[: self.chunk_size]  # 返回前 chunk_size 字节的数据
                    audio_chunk = audio_chunk[self.chunk_size :]  # 移除已返回的数据
                yield np.pad(audio_chunk, (0, self.chunk_size - len(audio_chunk)))
        else:
            wavs = wavs_gen
            if len(wavs[0]) == 0:
                self.should_listen.set()
                return
            audio_chunk = librosa.resample(wavs[0], orig_sr=24000, target_sr=16000)
            audio_chunk = (audio_chunk * 32768).astype(np.int16)
            for i in range(0, len(audio_chunk), self.chunk_size):
                yield np.pad(
                    audio_chunk[i : i + self.chunk_size],
                    (0, self.chunk_size - len(audio_chunk[i : i + self.chunk_size])),
                )
        self.should_listen.set()