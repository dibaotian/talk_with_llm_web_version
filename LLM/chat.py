class Chat:
    """
    Handles the chat using to avoid OOM issues.
    避免内存被耗尽
    """

    def __init__(self, size):
        self.size = size
        self.init_chat_message = None
        # maxlen is necessary pair, since a each new step we add an prompt and assitant answer
        self.buffer = []

    def append(self, item):
        # 向buffer中添加新消息
        self.buffer.append(item)
        # 检查 buffer 的长度
        # 如果超过了允许的大小（2 * (self.size + 1)），则删除最旧的两条消息（一个用户输入和一个助手回复）
        # 确保了聊天记录不会无限增长，避免占用过多内存
        if len(self.buffer) == 2 * (self.size + 1):
            self.buffer.pop(0)
            self.buffer.pop(0)   

    def init_chat(self, init_chat_message):
        # 初始化一个空的聊天消息
        self.init_chat_message = init_chat_message

    def to_list(self):
        # 模型记忆短期记忆（聊天上下文）
        # 返回列表，包含初始聊天消息和当前的聊天历史记录。
        if self.init_chat_message:
            return [self.init_chat_message] + self.buffer
        else:
            return self.buffer