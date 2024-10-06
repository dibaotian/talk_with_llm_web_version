import requests
import base64
from PIL import Image
from io import BytesIO

def send_image_for_analysis(image_path, question):
    # 服务的URL
    url = "http://localhost:5000/analyze"

    # 读取图像文件并转换为base64
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

    # 准备请求数据
    data = {
        "image": encoded_image,
        "question": question
    }

    try:
        # 发送POST请求
        response = requests.post(url, json=data)
        
        # 检查请求是否成功
        response.raise_for_status()
        
        # 解析JSON响应
        result = response.json()
        
        # 返回分析结果
        return result["response"]
    
    except requests.exceptions.RequestException as e:
        print(f"发送请求时出错: {e}")
        return None

# 使用示例
image_path = "/home/xilinx/Documents/talk_with_llm_web_version/static/frames/frame_000.jpg"  # 替换为您的图像文件路径
question = "这张图片里有什么？"  # 您想问的问题

result = send_image_for_analysis(image_path, question)

if result:
    print("分析结果:", result)
else:
    print("无法获取分析结果")