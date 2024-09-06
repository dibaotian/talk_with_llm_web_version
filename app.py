from flask import Flask, render_template
from flask_socketio import SocketIO
import wave
import base64

import os
import numpy as np
import threading
from threading import Event, Thread
from queue import Queue


# from time import perf_counter
from VAD.vad_handler import VADHandler
from STT.whisper_stt_handler import WhisperSTTHandler
from LLM.large_language_model_handler import LargeLanguageModelHandler
from TTS.chattts_handler import ChatTTSHandler
# from SEND_STREAMER.send_streamer_handler import SendAudioHandler  # 导入类

import logging
global logger
logging.basicConfig(
    filename='log_app.log',  # 将日志写入文件
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True,
)
logger = logging.getLogger(__name__)
logger = logging.getLogger("Server_logger")
# logger.setLevel(logging.DEBUG)


class ThreadManager:
    """
    Manages multiple threads used to execute given handler tasks.
    """

    def __init__(self, handlers):
        self.handlers = handlers
        self.threads = []

    def start(self):
        for handler in self.handlers:
            thread = threading.Thread(target=handler.run)
            self.threads.append(thread)
            thread.start()

    def stop(self):
        for handler in self.handlers:
            handler.stop_event.set()
        for thread in self.threads:
            thread.join()


class SendAudioHandler:
    """
    Handles streaming audio data to clients via SocketIO.
    """

    def __init__(self, stop_event, queue_in):
        self.stop_event = stop_event
        self.queue_in = queue_in
        logger.info('AudioStreamer process')

    def setup(self):
        """
        Perform any necessary setup before processing audio data.
        """
        logger.info("AudioStreamer setup complete.")

    def run(self):
        """
        Continuously process audio data from the queue and send it to clients.
        """
        while not self.stop_event.is_set():
            try:
                audio_chunk = self.queue_in.get()
                logger.info(f'Received audio chunk: {type(audio_chunk)}, shape: {getattr(audio_chunk, "shape", None)}')
                
                if isinstance(audio_chunk, np.ndarray) and audio_chunk.size > 0:
                    # 将NumPy数组转换为字节串
                    audio_bytes = audio_chunk.tobytes()
                    encoded_audio = base64.b64encode(audio_bytes).decode('utf-8')
                    socketio.emit('play_audio_stream', {'data': encoded_audio})
                elif isinstance(audio_chunk, bytes) and len(audio_chunk) > 0:
                    encoded_audio = base64.b64encode(audio_chunk).decode('utf-8')
                    socketio.emit('play_audio_stream', {'data': encoded_audio})
                    logger.info("Sent audio chunk to client")
                else:
                    logger.warning(f"Received invalid or empty audio chunk: {audio_chunk}")
            except Exception as e:
                logger.error(f"Error processing audio chunk: {e}", exc_info=True)



app = Flask(__name__)
socketio = SocketIO(app)

UPLOAD_FOLDER = 'uploads'
# 创建一个目录来保存录制的音频
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Build the pipeline
stop_event = Event()
# used to stop putting received audio chunks in queue until all setences have been processed by the TTS
should_listen = Event()

# 创建pipline每一个进程的queue
recv_audio_chunks_queue = Queue()
send_audio_chunks_queue = Queue()
spoken_prompt_queue = Queue()
text_prompt_queue = Queue()
lm_response_queue = Queue()


# 用于存储接收到的音频数据，用于存文件
audio_data_list = []

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('audio')
def handle_audio(data):
    global audio_data_list
    audio_data_list.append(data)  # 将接收到的数据添加到列表中
    # recv_audio_chunks_queue.put(data)  # 将接收到的数据放入队列,要注意数据长度需要满足VAD的输入要求
    # 检查当前音频数据的总长度
    if sum(len(chunk) for chunk in audio_data_list) >= 1024:  # 如果总长度达到1024
        combined_data = b''.join(audio_data_list)  # 合并所有音频数据
        # print("Combined data length:", len(combined_data))  # 打印合并后的数据长度
        recv_audio_chunks_queue.put(combined_data)  # 将合并后的数据放入队列
        audio_data_list.clear()  # 清空列表以准备接收新的数据

@socketio.on('stop')
def stop_recording():
    print("stop_recording data")
    save_data = False

    # 发送一个特殊的信号来指示处理进程停止
    stop_event.set()
    recv_audio_chunks_queue.put(b'END')

    if (save_data):
        global audio_data_list
        if audio_data_list:
            # 将接收到的音频数据合并
            combined_data = b''.join(audio_data_list)
            audio_data = np.frombuffer(combined_data, dtype=np.float32)  # 假设 PCM 数据是 float32 类型
            print(f"Received audio data length: {len(combined_data)}")  # 打印接收到的数据长度

            audio_data = (audio_data * 32767).astype(np.int16)  # 转换为 int16

            print("将接收到的音频数据保存为 WAV 文件")

            # 将接收到的音频数据保存为 WAV 文件
            with wave.open(os.path.join(UPLOAD_FOLDER, 'recorded_audio.wav'), 'wb') as wf:
                wf.setnchannels(1)  # 单声道
                wf.setsampwidth(2)  # 16 位
                wf.setframerate(441000)  # 采样率
                wf.writeframes(audio_data.tobytes())  # 写入音频数据

            # 清空音频数据列表
            audio_data_list = []

def main():
   
    # 测试日志是否正常打印
    # logger.info("安装funasr后需要使用自定义日志记录器。")


    vad = VADHandler(
        stop_event,
        queue_in=recv_audio_chunks_queue,
        queue_out=spoken_prompt_queue,
        setup_args=(should_listen,),
        # setup_kwargs=vars(vad_handler_kwargs),
    )

    #create whisper stt instance
    whisper_stt = WhisperSTTHandler(
        stop_event,
        queue_in=spoken_prompt_queue,
        queue_out=text_prompt_queue,
    )

    # create LLM instance
    llm = LargeLanguageModelHandler(
        stop_event,
        queue_in=text_prompt_queue,
        queue_out=lm_response_queue,
    )

    # create TTS instance
    chat_tts = ChatTTSHandler(
        stop_event,
        queue_in=lm_response_queue,
        queue_out=send_audio_chunks_queue,
        setup_args=(should_listen,),
    )

    send_audio = SendAudioHandler(
        stop_event, 
        send_audio_chunks_queue,
    )

    try:
        # pipeline_manager = ThreadManager([vad, whisper_stt, llm, chat_tts])
        pipeline_manager = ThreadManager([vad, whisper_stt, llm, chat_tts, send_audio])
        # pipeline_manager = ThreadManager([llm])
        pipeline_manager.start()

    except KeyboardInterrupt:
        pipeline_manager.stop()

    # 设置为监听所有 IP 地址，并将端口改为 8888
    # socketio.run(app, host='0.0.0.0', port=8888, debug=True)
    socketio.run(app, host='0.0.0.0', port=8888, ssl_context=('cert.pem', 'key.pem'))


if __name__ == '__main__':
    main()
    

   

