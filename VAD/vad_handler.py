from base_handler import BaseHandler
from VAD.vad_iterator import VADIterator
import numpy as np
import torch
from rich.console import Console

import torchaudio
from utils import int2float
from df.enhance import enhance, init_df

import logging

logger = logging.getLogger(__name__)

console = Console()

class VADHandler(BaseHandler):
    """
    Handles voice activity detection. 
    When voice activity is detected, audio will be accumulated until the end of speech is detected and then passed
    to the following part.
    Here we can assign it to CPU or GPU, default is using CPU
    2024-10-06 add DeepFilterNet  https://github.com/Rikorose/DeepFilterNet/tree/main
    """

    def setup(
            self, 
            should_listen,
            thresh=0.3, 
            sample_rate=16000, 
            min_silence_ms=1000,
            min_speech_ms=500, 
            max_speech_ms=float('inf'),
            speech_pad_ms=30,
            audio_enhancement=False,
        ):
        self.should_listen = should_listen
        self.sample_rate = sample_rate
        self.min_silence_ms = min_silence_ms
        self.min_speech_ms = min_speech_ms
        self.max_speech_ms = max_speech_ms

        # 确定设备
        print("load the VAD model")
        # Silero VAD supports 8000 Hz and 16000 Hz sampling rates.
        self.model, _ = torch.hub.load('snakers4/silero-vad', 'silero_vad')
        logger.info("VAD model loaded successfully")

        # set the process to GPU cause VAD can not work properly
        # self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # logger.info("VAD will be asign to {self.device}")
        # self.model = self.model.to(self.device)

        self.iterator = VADIterator(
            self.model,
            threshold=thresh,
            sampling_rate=sample_rate,
            min_silence_duration_ms=min_silence_ms,
            speech_pad_ms=speech_pad_ms,
        )

        # self.audio_enhancement = audio_enhancement
        # if audio_enhancement:
        #     self.enhanced_model, self.df_state, _ = init_df()

    def process(self, audio_chunk):
        try:
            # logger.debug("Processing audio chunk for VAD")
            # print("audio_chunk len", len(audio_chunk))
            if not isinstance(audio_chunk, bytes):
                raise ValueError("Expected audio_chunk to be of type bytes")
            
            audio_int16 = np.frombuffer(audio_chunk, dtype=np.int16)
            audio_float32 = int2float(audio_int16)

             # 确保输入长度符合模型要求
            num_samples = 512  # 对于 16000 Hz 的采样率
            if len(audio_float32) < num_samples:
                # 如果样本数不足，进行填充
                print("audio data padding")
                audio_float32 = np.pad(audio_float32, (0, num_samples - len(audio_float32)), 'constant')
            elif len(audio_float32) > num_samples:
                # 如果样本数过多，进行裁剪
                print("audio data truncation")
                audio_float32 = audio_float32[:num_samples]
                
            # 应用VAD
            vad_output = self.iterator(torch.from_numpy(audio_float32))
            if vad_output is not None and len(vad_output) != 0:
                logger.debug("VAD: end of speech detected")
                # logger.debug(f"VAD output: {vad_output}")
                array = torch.cat(vad_output).cpu().numpy()
                duration_ms = len(array) / self.sample_rate * 1000
                if duration_ms < self.min_speech_ms or duration_ms > self.max_speech_ms:
                    logger.debug(f"audio input of duration: {len(array) / self.sample_rate}s, skipping")
                else:
                    self.should_listen.clear()
                    logger.debug("Stop listening")
                    # if self.audio_enhancement:
                    #     if self.sample_rate != self.df_state.sr():
                    #         audio_float32 = torchaudio.functional.resample(
                    #             torch.from_numpy(array),
                    #             orig_freq=self.sample_rate,
                    #             new_freq=self.df_state.sr(),
                    #         )
                    #         enhanced = enhance(
                    #             self.enhanced_model,
                    #             self.df_state,
                    #             audio_float32.unsqueeze(0),
                    #         )
                    #         enhanced = torchaudio.functional.resample(
                    #             enhanced,
                    #             orig_freq=self.df_state.sr(),
                    #             new_freq=self.sample_rate,
                    #         )
                    # else:
                    #     enhanced = enhance(
                    #         self.enhanced_model, self.df_state, audio_float32
                    #     )
                    #     array = enhanced.numpy().squeeze()

                    yield array
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e},exc_info=True")
