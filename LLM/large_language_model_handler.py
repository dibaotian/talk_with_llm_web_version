from base_handler import BaseHandler
from threading import Thread

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    TextIteratorStreamer,
    BitsAndBytesConfig,
)
import torch

from LLM.chat import Chat
from rich.console import Console

from nltk import sent_tokenize

import json
from flask_socketio import emit
import re

import logging

from LLM.prompt import minsprompts

from LLM.ddg_search import DuckDuckGoSearch

logger = logging.getLogger(__name__)

console = Console()

search_agent = DuckDuckGoSearch()

class LargeLanguageModelHandler(BaseHandler):
    """
    Handles the language model part. 
    Here I use Qwen2-7B
    """

    def __init__(self, stop_event, queue_in, queue_out):
        super().__init__(stop_event, queue_in, queue_out)
        self.socketio = None

    def setup(
            self,
            # model_name ="Qwen/Qwen2-7B-Instruct",
            model_name ="Qwen/Qwen2.5-7B-Instruct",
            # model_name ="THUDM/glm-4v-9b",
            # model_name ="Qwen/Qwen2-7B-Instruct-GPTQ-Int4", # load fail
            # model_name ="Qwen/Qwen2-7B-Instruct-GPTQ-Int8",  # too slow in v100
            device = None,  # let system select
            gpu_id = 0,  # 新增参数，默认使用第一个 GPU
            torch_dtype = "auto",
            # gen_kwargs={'return_full_text': False, 'temperature': 0.7, 'do_sample': False},
            gen_kwargs={'return_full_text': False,'do_sample': True},
            user_role="user",
            chat_size=10,
            init_chat_role="user", 
            # init_chat_prompt="""1. 你是一个AI家庭助手, 你的名字叫Joey或者周一, 你可以提供家庭生活中的帮助, 你的回答应简洁明了, 并以友好、轻松、可爱的口吻回应用户的问题。
   
            #                     2. 判断是否是技术性问题的回答, 如果是技术性问题, 可以回答得详细一点。对于一般常识性问题的回答, 最好不要超过100个字符。你可以使用多种agent, 比如控制家里的电器, 上网查询, 控制打印机等, 你可以根据需要判断是否要使用这些agent。

            #                     3. 如果用户询问家里的温度情况或者说有点热, 请说出你的想法。如果你认为有必要调用工具, 请生成一个 JSON 格式的agent动作, 例如:
            #                     {"action": "check_temperature", "rooms": ["living_room", "bedroom"]}

            #                     4. 如果用户询问家里的灯光或光线问题, 请先表达你的想法。如果你认为有必要调用工具, 请生成一个 JSON 格式的agent动作, 如下:  
            #                     - 开灯: {"device":"light", "rooms": ["living room"],"action": "set_power", "value":"on"}
            #                     - 关灯: {"device":"light", "rooms": ["bed room"], "action": "set_power", "value":"off"}
            #                     - 调整亮度(范围0-100,最亮是100): {"device":"light", "rooms": ["living room"],"action": "adjust_brightness", "value": 70}
            #                     - 调整色温(范围2700-6000,从暖光到冷光): {"device":"light", "rooms": ["bathroom"], "action": "adjust_color_temperature", "value": 3000}

            #                     5. 如果用户询问与饭有关的问题, 如果你认为有必要调用工具, 可以生成一个查询米家电饭煲的信息的JSON格式的agent动作  
            #                     - 查询: {"device":"Rice Cooker", "rooms": ["kitchen"], "action": "check_status", "value":""}

            #                     6. 如果用户要求打印内容, 如果你认为有必要调用工具, 请生成一个包含打印内容的 JSON 格式动作, 如下:
            #                     {"device": "printer", "action": "print", "content": "需要打印的内容"}

            #                     7. 如果你判断用户询问需要到外部知识库获取,例如是提到搜索的时候(如搜索当前天气、汇率等与时间相关的问题, 请生成一个包含要搜索内容的 JSON 格式动作并提醒用户你生成了该动作。例如:  
            #                     {"device": "ddg", "action": "search", "content": "需要搜索的内容"}  
            #                     然后可以调用搜索的agent

            #                     8. 在生成JSON格式的agent动作时, 请确保将其放在单独的一行, 并用大括号{}包围。
            #                 """,

            init_chat_prompt= minsprompts
        ):

        # 让系统选择是CPU还是GPU
        if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
            device = f"cuda:{gpu_id}"
        else:
            logger.warning(f"Specified GPU {gpu_id} is not available. Using default device.")
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device


        logger.info(f"LLM {model_name} will be assigned to device {self.device}")

        # 加载与指定模型对应的分词器（Tokenizer）（从 Hugging Face 的模型库或本地路径）
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)

        ##################################################################################
        # if model_name == "THUDM/glm-4v-9b":
        if model_name == "Qwen/Qwen2.5-7B-Instruct":

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                load_in_4bit=True,  # 或 load_in_4bit=True, 视具体情况而定
                device_map="auto",  # 自动分配设备
                trust_remote_code=True,
            )

            # 文本生成管道
            self.pipe = pipeline( 
                "text-generation", 
                model=self.model, 
                tokenizer=self.tokenizer, 
                torch_dtype="auto", 
                max_new_tokens = 256,  # 设置生成的新 maxtoken 数量 输出序列太长会导致TTS处理不过来
                min_new_tokens = 10  # 设置生成的新 mintoken 数量
            ) 
        else:
            # # 加载模型，指定模型的计算精度（数据类型），在指定设备上运行
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            ).to(device)
    

            # 文本生成管道
            self.pipe = pipeline( 
                "text-generation", 
                device=device,
                model=self.model, 
                tokenizer=self.tokenizer, 
                torch_dtype="auto", 
                max_new_tokens = 256,  # 设置生成的新 maxtoken 数量 输出序列太长会导致TTS处理不过来
                min_new_tokens = 10  # 设置生成的新 mintoken 数量
            ) 
        ##################################################################################

        # 16G 内存不够了，加载模型，启用8-bit量化
        # 配置 8-bit 量化
        # quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        # # 加载模型，启用量化
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     model_name,
        #     quantization_config=quantization_config,
        #     trust_remote_code=True
        # )

        # # please install AutoGPTQ following the readme to use quantization
        # from auto_gptq import AutoGPTQForCausalLM
        # model = AutoGPTQForCausalLM.from_quantized(
        #     "Qwen/Qwen-7B-Chat-Int4", 
        #     device="cuda:0", 
        #     trust_remote_code=True, 
        #     use_safetensors=True, 
        #     use_flash_attn=False
        # ).eval()
        
        # # 文本生成管道
        # self.pipe = pipeline( 
        #     "text-generation", 
        #     model=self.model, 
        #     tokenizer=self.tokenizer, 
        #     torch_dtype="auto", 
        #     max_new_tokens = 512,  # 设置生成的新 maxtoken 数量 输出序列太长会导致TTS处理不过来
        #     min_new_tokens = 10  # 设置生成的新 mintoken 数量
        # ) 
        ##################################################################################

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

    def set_socketio(self, socketio):
        self.socketio = socketio

    def send_json_to_frontend(self, data):
        if self.socketio:
            self.socketio.emit('agent_action', data)

    def process(self, prompt):
        logger.info("inferring language model...")

        self.chat.append(
            {"role": self.user_role, "content": prompt}
        )

        thread = Thread(target=self.pipe, args=(self.chat.to_list(),), kwargs=self.gen_kwargs)
        thread.start()

        generated_text, printable_text = "", ""
        for new_text in self.streamer:
            generated_text += new_text
            printable_text += new_text

            # # 检查是否有图片显示请求
            # image_path = self.check_for_image_display(printable_text)
            # if image_path:
            #     image_url = url_for('static', filename=f'images/{image_path}')
            #     self.send_to_frontend(f"<img src='{image_url}' alt='Displayed Image'>")
            #     printable_text = ""
            # else:
            #     # 发送文本到前端
            #     self.send_to_frontend(new_text)

            # 检查是否有 agent 动作
            agent_action, remaining_text = self.extract_agent_action(printable_text)

            if agent_action:
                if agent_action.get('device') == 'ddg' and agent_action.get('action') == 'search':
                    search_query = agent_action.get('content', '')
                    search_summary = self.search_agent_action(search_query)
                    yield f"请稍等，我需要搜索外部知识库:"
                    self.send_json_to_frontend(agent_action)
                else:
                    self.send_json_to_frontend(agent_action)
                printable_text = remaining_text  # 保留非 JSON 部分的文本
            else:
                sentences = sent_tokenize(printable_text)
                while len(sentences) > 1:
                    sentence = sentences.pop(0)
                    if not self.is_json_like(sentence):  # 添加这个检查
                        yield sentence
                printable_text = sentences[0] if sentences else ""

        self.chat.append(
            {"role": "assistant", "content": generated_text}
        )

         # 输出最后的句子
        if printable_text:
            yield printable_text

    def send_to_frontend(self, text):
        if self.socketio:
            self.socketio.emit('llm_response', {'text': text}, namespace='/chat')

    def clean_summary(self, summary):
        # 移除可能的提示内容
        cleaned = summary.replace("以下是关于", "").replace("的搜索结果。请总结重要信息", "")
        cleaned = cleaned.replace("请提供一个简洁的总结,最好不要超过100个字,其中包括最重要的信息和任何相关的日期或事实。", "")
        
        # 移除可能的前缀
        prefixes_to_remove = [
            "搜索结果总结：",
            "总结：",
            "简要总结：",
            "概括：",
        ]
        for prefix in prefixes_to_remove:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
        
        # 移除可能的 JSON 格式内容
        cleaned = re.sub(r'\{.*?\}', '', cleaned)
        
        # 移除多余的空白字符
        cleaned = ' '.join(cleaned.split())
        
        return cleaned.strip()

    def extract_agent_action(self, text):
        # 使用正则表达式查找 JSON 格式的 agent 动作
        pattern = r'(\{[^{}]*\})'
        matches = re.finditer(pattern, text)
        
        for match in matches:
            try:
                action = json.loads(match.group())
                # 从原文本中移除 JSON 部分，但保留其他文本
                remaining_text = text[:match.start()] + text[match.end():]
                return action, remaining_text.strip()
            except json.JSONDecodeError:
                continue
        
        return None, text

    def is_json_like(self, text):
        return text.strip().startswith('{') and text.strip().endswith('}')

    def search_agent_action(self, query):
        search_results = search_agent(query)
        
        # 准备提示
        prompt = (
            f"你现在有关于'{query}'的搜索结果。请从中提取出最重要的信息，并简洁地总结。\n\n"
            "以下是搜索结果的简要概述：\n"
        )

        for result in search_results[:5]:  # 限制前5个结果
            prompt += f"- {result['title']}:\n  {result['body']}\n"

        prompt += (
            "\n请基于以上信息,撰写一个不超过100字的总结,包含最关键的事实、日期或其他相关细节，"
            "帮助用户快速了解最重要的信息。请保持语言简洁易懂。"
        )

        # 使用语言模型生成总结，并只返回生成的文本部分
        summary = self.generate_response(prompt).strip()  # 清除多余的空格和换行符
        return summary

    def generate_response(self, prompt):
        self.chat.append({"role": "user", "content": prompt})
        
        thread = Thread(target=self.pipe, args=(self.chat.to_list(),), kwargs=self.gen_kwargs)
        thread.start()

        generated_text = ""
        for new_text in self.streamer:
            generated_text += new_text

        self.chat.append({"role": "assistant", "content": generated_text})

        # 移除可能的提示内容
        response = generated_text.replace(prompt, "").strip()
        
        return response

