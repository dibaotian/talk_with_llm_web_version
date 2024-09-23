from base_handler import BaseHandler
from flask_socketio import emit

from melo.api import TTS
import logging
import librosa
import numpy as np
from rich.console import Console
import torch

import nltk
nltk.download('averaged_perceptron_tagger_eng')

logger = logging.getLogger(__name__)

console = Console()

WHISPER_LANGUAGE_TO_MELO_LANGUAGE = {
    "en": "EN_NEWEST",
    "fr": "FR",
    "es": "ES",
    "zh": "ZH",
    "ja": "JP",
    "ko": "KR",
}

WHISPER_LANGUAGE_TO_MELO_SPEAKER = {
    "en": "EN-Newest",
    "fr": "FR",
    "es": "ES",
    "zh": "ZH",
    "ja": "JP",
    "ko": "KR",
}


class MeloTTSHandler(BaseHandler):


    def setup(
        self,
        should_listen,
        device="cuda:0",
        language="zh",
        speaker_to_id="zh",
        gen_kwargs={},  # Unused
        blocksize=512,
    ):
        self.should_listen = should_listen
        self.device = device
        self.language = language
        print("load the MeloTTS model")
        self.model = TTS(
            language=WHISPER_LANGUAGE_TO_MELO_LANGUAGE[self.language], device=device
        )
        self.speaker_id = self.model.hps.data.spk2id[
            WHISPER_LANGUAGE_TO_MELO_SPEAKER[speaker_to_id]
        ]
        self.blocksize = blocksize
        self.warmup()

    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")
        _ = self.model.tts_to_file("text, 宇宙的终极答案是什么", self.speaker_id, quiet=True)


    def set_socketio(self, socketio):
        self.socketio = socketio  

    def process(self, llm_sentence):
        language_code = None

        if isinstance(llm_sentence, tuple):
            llm_sentence, language_code = llm_sentence

        # console.print(f"[green]ASSISTANT: {llm_sentence}")

        # 发送 LLM 结果到前端
        if self.socketio:
            self.socketio.emit('llm_response', {'text': llm_sentence})
        

        if language_code is not None and self.language != language_code:
            try:
                self.model = TTS(
                    language=WHISPER_LANGUAGE_TO_MELO_LANGUAGE[language_code],
                    device=self.device,
                )
                self.speaker_id = self.model.hps.data.spk2id[
                    WHISPER_LANGUAGE_TO_MELO_SPEAKER[language_code]
                ]
                self.language = language_code
            except KeyError:
                console.print(
                    f"[red]Language {language_code} not supported by Melo. Using {self.language} instead."
                )

        if self.device == "mps":
            import time

            start = time.time()
            torch.mps.synchronize()  # Waits for all kernels in all streams on the MPS device to complete.
            torch.mps.empty_cache()  # Frees all memory allocated by the MPS device.
            _ = (
                time.time() - start
            )  # Removing this line makes it fail more often. I'm looking into it.

        try:
            audio_chunk = self.model.tts_to_file(
                llm_sentence, self.speaker_id, quiet=True
            )
        except (AssertionError, RuntimeError) as e:
            logger.error(f"Error in MeloTTSHandler: {e}")
            audio_chunk = np.array([])
        if len(audio_chunk) == 0:
            self.should_listen.set()
            return
        audio_chunk = librosa.resample(audio_chunk, orig_sr=44100, target_sr=16000)
        audio_chunk = (audio_chunk * 32768).astype(np.int16)
        for i in range(0, len(audio_chunk), self.blocksize):
            yield np.pad(
                audio_chunk[i : i + self.blocksize],
                (0, self.blocksize - len(audio_chunk[i : i + self.blocksize])),
            )

        self.should_listen.set()