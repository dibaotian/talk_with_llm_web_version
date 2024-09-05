// audio-processor.js
class AudioProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.port.onmessage = (event) => {
            // 处理来自主线程的消息
            console.log('Message from main thread:', event.data);
        };
    }

    process(inputs, outputs, parameters) {
        const input = inputs[0];
        if (input.length > 0) {
            const channelData = input[0]; // 获取第一个通道的数据
            
            // 将浮点数转换为 16 位整数
            const int16Array = new Int16Array(channelData.length);
            for (let i = 0; i < channelData.length; i++) {
                int16Array[i] = Math.max(-1, Math.min(1, channelData[i])) * 32767; // 量化到 16 位
            }

            this.port.postMessage(int16Array.buffer); // 发送 ArrayBuffer 到主线程
        }
        return true; // 返回 true 以继续处理
    }
}

registerProcessor('audio-processor', AudioProcessor);