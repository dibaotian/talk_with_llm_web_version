from base_handler import BaseHandler

from threading import Thread

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    TextIteratorStreamer,
)
import torch

from LLM.chat import Chat
from rich.console import Console

from nltk import sent_tokenize

import logging

logger = logging.getLogger(__name__)

console = Console()


class LargeLanguageModelHandler(BaseHandler):
    """
    Handles the language model part. 
    Here I use Qwen2-7B
    """

    def setup(
            self,
            model_name ="Qwen/Qwen2-7B-Instruct",
            device = None,  # let system select
            torch_dtype = "auto",
            # gen_kwargs={'return_full_text': False, 'temperature': 0.7, 'do_sample': False},
            gen_kwargs={'return_full_text': False,'do_sample': True},
            user_role="user",
            chat_size=1,
            init_chat_role=None, 
            init_chat_prompt="你是一个AI助手.",
        ):

        # 让系统选择是CPU还是GPU
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device


        logger.info(f"LLM {model_name} will be assigned to device {self.device}")

        # 加载与指定模型对应的分词器（Tokenizer）（从 Hugging Face 的模型库或本地路径）
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # 加载模型，指定模型的计算精度（数据类型），在指定设备上运行
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True
        ).to(device)
        
        # 文本生成管道
        self.pipe = pipeline( 
            "text-generation", 
            device=device,
            model=self.model, 
            tokenizer=self.tokenizer, 
            torch_dtype="auto", 
            max_new_tokens = 512,  # 设置生成的新 maxtoken 数量 输出序列太长会导致TTS处理不过来
            min_new_tokens = 10  # 设置生成的新 mintoken 数量
        ) 

        # 流式输出，处理生成的文本
        # 用于逐步处理文本生成任务中的输出。在文本生成的过程中逐个处理生成的 token，不是等待整个文本生成完毕。
        self.streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True, #跳过初始的提示部分（用户输入的文本），只返回生成的内容。
            skip_special_tokens=True,  #跳过特殊 token（如 <|endoftext|> 等），使输出的文本更干净
        )
        self.gen_kwargs = {
            "streamer": self.streamer,
            "return_full_text": False,
            **gen_kwargs
        }

        # 初始化Chat 对象，并根据条件设置初始聊天信息和用户角色， 
        # chat_size 聊天记录的大小，（限制缓冲区中存储的对话轮数）
        self.chat = Chat(chat_size)
        if init_chat_role:
            if not init_chat_prompt:
                raise ValueError(f"An initial promt needs to be specified when setting init_chat_role.")
            self.chat.init_chat(
                {"role": init_chat_role, "content": init_chat_prompt}
            )
        self.user_role = user_role

        self.warmup()

    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")

        dummy_input_text = "我是一个语音机器人"
        dummy_chat = [{"role": self.user_role, "content": dummy_input_text}]
        warmup_gen_kwargs = {
            "min_new_tokens": 20,
            "max_new_tokens": 50,
            **self.gen_kwargs
        }

        # warmup_gen_kwargs {'min_new_tokens': 64, 'max_new_tokens': 64, 'streamer': <transformers.generation.streamers.TextIteratorStreamer object at 0x7f7464088040>, 'return_full_text': False, 'temperature': 0.0, 'do_sample': False}

        n_steps = 2

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start_event.record()
        for _ in range(n_steps):
            thread = Thread(target=self.pipe, args=(dummy_chat,), kwargs=warmup_gen_kwargs)
            thread.start()
            for _ in self.streamer: 
                pass    
        end_event.record()
        torch.cuda.synchronize()

        logger.info(f"{self.__class__.__name__}:  warmed up! time: {start_event.elapsed_time(end_event) * 1e-3:.3f} s")

    def process(self, prompt):
        logger.info("infering language model...")

        self.chat.append(
            {"role": self.user_role, "content": prompt}
        )

        thread = Thread(target=self.pipe, args=(self.chat.to_list(),), kwargs=self.gen_kwargs)
        thread.start()

        generated_text, printable_text = "", ""
        for new_text in self.streamer:
            generated_text += new_text
            printable_text += new_text
            sentences = sent_tokenize(printable_text)
            if len(sentences) > 1:
                yield(sentences[0])
                printable_text = new_text

        self.chat.append(
            {"role": "assistant", "content": generated_text}
        )

        # don't forget last sentence
        print(printable_text)
        yield printable_text
