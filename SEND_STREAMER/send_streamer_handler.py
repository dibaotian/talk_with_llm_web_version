from app import socketio

import logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class SendAudioHandler:
    """
    Handles streaming audio data to clients via SocketIO.
    """

    def __init__(self, stop_event, queue_in):
        self.stop_event = stop_event
        self.queue_in = queue_in
        logger.info('AudioStreamer get chunk...')

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
            audio_chunk = self.queue_in.get()
            logger.info('AudioStreamer get chunk...')
            if audio_chunk:
                # 将音频数据编码为 base64
                logger.info('audio_chunk encoded...')
                encoded_audio = base64.b64encode(audio_chunk).decode('utf-8')
                # 发送音频数据到前端
                socketio.emit('play_audio_stream', {'data': encoded_audio})
                logger.info("Sent audio chunk to client")