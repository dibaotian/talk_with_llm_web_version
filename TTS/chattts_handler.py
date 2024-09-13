from base_handler import BaseHandler
from flask_socketio import emit

import ChatTTS

# import librosa
import numpy as np
from rich.console import Console
import torch

import lzma
import pybase16384 as b14
from time import perf_counter


import logging
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
        compile_mode=False,
        temperature = .3,
        top_P = 0.7,        # top P decode
        top_K = 20,         # top K decode
        description=(
            "TTS"
        ),
    ):
        self.should_listen = should_listen
        self.model =  ChatTTS.Chat(logger)
        logger.info("load ChatTTS")
        # use force_redownload=True if the weights have been updated.
        # self.model.load(source="huggingface")
        # chat.load(source='local') same source not set   
        self.model.load(compile=compile_mode, device=device, use_vllm=False) # Set to compile =  True for better performance
        
        ###################################
        # Sample a speaker from Gaussian.
        # rand_spk = self.model.sample_random_speaker()
        # # print(rand_spk) # save it for later timbre recovery
        # self.params_infer_code = self.model.InferCodeParams(
        #     spk_emb = rand_spk, # add sampled speaker 
        #     temperature = temperature,   # 控制音频情感波动性，范围为 0-1，数字越大，波动性越大
        #     top_P = top_P,        # 控制音频的情感相关性，范围为 0.1-0.9，数字越大，相关性越高
        #     top_K = top_K,        # 控制音频的情感相似性，范围为 1-20，数字越小，相似性越高
        # )


        # 指定音色种子值每次生成 spk_emb 和重复使用预生成好的 spk_emb 效果有较显著差异
        # 使用 .pt 音色文件或者音色码效果会好一些
        # pt 下载 https://huggingface.co/spaces/taa/ChatTTS_Speaker  音色控制
        spk = torch.load("TTS/pt/seed_929_restored_emb.pt", map_location=torch.device(device)).detach()
        spk_emb_str = ChatTTSHandler._encode_spk_emb(spk)

        self.params_infer_code = self.model.InferCodeParams(
            spk_emb = spk_emb_str, # add sampled speaker 
            temperature = temperature,   # 控制音频情感波动性，范围为 0-1，数字越大，波动性越大
            top_P = top_P,        # 控制音频的情感相关性，范围为 0.1-0.9，数字越大，相关性越高
            top_K = top_K,        # 控制音频的情感相似性，范围为 1-20，数字越小，相似性越高
        )

        # use oral_(0-9), laugh_(0-2), break_(0-7) 
        # to generate special token in text to synthesize.
        self.params_refine_text = self.model.RefineTextParams(
            prompt='[oral_2][laugh_0][break_6]',
        )

        # ChatTTSHandler.warmup()
    
    @staticmethod
    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")
        _ = self.model.infer("warming up chattts for more efficiency tts process")


    @staticmethod
    @torch.no_grad()
    def _encode_spk_emb(spk_emb: torch.Tensor) -> str:
        arr: np.ndarray = spk_emb.to(dtype=torch.float16, device="cpu").numpy()
        s = b14.encode_to_string(
            lzma.compress(
                arr.tobytes(),
                format=lzma.FORMAT_RAW,
                filters=[{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}],
            ),
        )
        del arr
        return s

    # stream状态更新。数据量不足的stream，先存一段时间，直到拿到足够数据，监控小块数据情况
    @staticmethod
    def _update_stream(history_stream_wav, new_stream_wav, thre):
        if history_stream_wav is not None:
            result_stream = np.concatenate([history_stream_wav, new_stream_wav], axis=1)
            is_keep_next = result_stream.shape[0] * result_stream.shape[1] < thre
            if random.random() > 0.1:
                print(
                    "update_stream",
                    is_keep_next,
                    [i.shape if i is not None else None for i in result_stream],
                )
        else:
            result_stream = new_stream_wav
            is_keep_next = result_stream.shape[0] * result_stream.shape[1] < thre

        return result_stream, is_keep_next

    # 已推理batch数据保存
    @staticmethod
    def _accum(accum_wavs, stream_wav):
        if accum_wavs is None:
            accum_wavs = stream_wav
        else:
            accum_wavs = np.concatenate([accum_wavs, stream_wav], axis=1)
        return accum_wavs

    # batch stream数据格式转化
    @staticmethod
    def batch_stream_formatted(stream_wav, output_format="PCM16_byte"):
        if output_format in ("PCM16_byte", "PCM16"):
            format_data = ChatTTSHandler.float_to_int16(stream_wav)
        else:
            format_data = stream_wav
        return format_data

    # 数据格式转化
    @staticmethod
    def formatted(data, output_format="PCM16_byte"):
        if output_format == "PCM16_byte":
            format_data = data.astype("<i2").tobytes()
        else:
            format_data = data
        return format_data

    # 检查声音是否为空
    @staticmethod
    def checkvoice(data):
        if np.abs(data).max() < 1e-6:
            return False
        else:
            return True

    # 将声音进行适当拆分返回
    @staticmethod
    def _subgen(data, thre=12000):
        for stard_idx in range(0, data.shape[0], thre):
            end_idx = stard_idx + thre
            yield data[stard_idx:end_idx]

    def float_to_int16(audio: np.ndarray) -> np.ndarray:
        am = int(math.ceil(float(np.abs(audio).max())) * 32768)
        am = 32767 * 32768 // am
        return np.multiply(audio, am).astype(np.int16)

    # 流式数据获取，支持获取音频编码字节流
    def generate(self, streamchat, output_format=None):
        assert output_format in ("PCM16_byte", "PCM16", None)
        curr_sentence_index = 0
        history_stream_wav = None
        article_streamwavs = None
        for stream_wav in streamchat:
            print(np.abs(stream_wav).max(axis=1))
            n_texts = len(stream_wav)
            n_valid_texts = (np.abs(stream_wav).max(axis=1) > 1e-6).sum()
            if n_valid_texts == 0:
                continue
            else:
                block_thre = n_valid_texts * 8000
                stream_wav, is_keep_next = ChatTTSHandler._update_stream(
                    history_stream_wav, stream_wav, block_thre
                )
                # 数据量不足，先保存状态
                if is_keep_next:
                    history_stream_wav = stream_wav
                    continue
                # 数据量足够，执行写入操作
                else:
                    history_stream_wav = None
                    stream_wav = ChatTTSHandler.batch_stream_formatted(
                        stream_wav, output_format
                    )
                    article_streamwavs = ChatTTSHandler._accum(
                        article_streamwavs, stream_wav
                    )
                    # 写入当前句子
                    if ChatTTSHandler.checkvoice(stream_wav[curr_sentence_index]):
                        for sub_wav in ChatTTSHandler._subgen(
                            stream_wav[curr_sentence_index]
                        ):
                            if ChatTTSHandler.checkvoice(sub_wav):
                                yield ChatTTSHandler.formatted(sub_wav, output_format)
                    # 当前句子已写入完成，直接写下一个句子已经推理完成的部分
                    elif curr_sentence_index < n_texts - 1:
                        curr_sentence_index += 1
                        print("add next sentence")
                        finish_stream_wavs = article_streamwavs[curr_sentence_index]

                        for sub_wav in ChatTTSHandler._subgen(finish_stream_wavs):
                            if ChatTTSHandler.checkvoice(sub_wav):
                                yield ChatTTSHandler.formatted(sub_wav, output_format)

                    # streamchat遍历完毕，在外层把剩余结果写入
                    else:
                        break
        # 本轮剩余最后一点数据写入
        if is_keep_next:
            if len(list(filter(lambda x: x is not None, stream_wav))) > 0:
                stream_wav = ChatTTSHandler.batch_stream_formatted(
                    stream_wav, output_format
                )
                if ChatTTSHandler.checkvoice(stream_wav[curr_sentence_index]):

                    for sub_wav in ChatTTSHandler._subgen(
                        stream_wav[curr_sentence_index]
                    ):
                        if ChatTTSHandler.checkvoice(sub_wav):
                            yield ChatTTSHandler.formatted(sub_wav, output_format)
                    article_streamwavs = ChatTTSHandler._accum(
                        article_streamwavs, stream_wav
                    )
        # 把已经完成推理的下几轮剩余数据写入
        for i_text in range(curr_sentence_index + 1, n_texts):
            finish_stream_wavs = article_streamwavs[i_text]

            for sub_wav in ChatTTSHandler._subgen(finish_stream_wavs):
                if ChatTTSHandler.checkvoice(sub_wav):
                    yield ChatTTSHandler.formatted(sub_wav, output_format)

    def set_socketio(self, socketio):
        self.socketio = socketio  # 添加这个方法

    def process(self,llm_sentence):

        # console.print(f"[green]ASSISTANT: {llm_sentence}")

        # 发送 LLM 结果到前端
        if self.socketio:
            self.socketio.emit('llm_response', {'text': llm_sentence})

        streamchat = self.model.infer(
            llm_sentence,
            skip_refine_text=True,
            stream=True,
            params_infer_code=self.params_infer_code,
            params_refine_text=self.params_refine_text,
        )

        streamer = ChatTTSHandler.generate(self, streamchat, output_format=None)

        for i, audio_chunk in enumerate(streamer):
            # if i == 0:
            #     logger.info(f"Time to first audio: {perf_counter() - pipeline_start:.3f}")
            # 将音频数据从浮点数格式转换为 16 位整数格式
            # 这里不处理会出现很多背景噪声
            audio_chunk = np.int16(audio_chunk * 32767)
            yield audio_chunk

        logger.info("wav_chunk tts processed")
        self.should_listen.set() # 设置监听状态
