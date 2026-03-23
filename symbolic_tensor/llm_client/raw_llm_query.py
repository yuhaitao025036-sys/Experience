# 安装：pip3 install openai
import asyncio
import os
from openai import AsyncOpenAI  # 导入异步客户端

# 创建异步客户端
client = AsyncOpenAI(
    api_key=os.environ.get('LLM_API_KEY'),
    base_url=os.environ.get('LLM_BASE_URL')
)

async def raw_llm_query(prompt: str):
    # 发送异步请求
    response = await client.chat.completions.create(
        model=os.environ.get('LLM_MODEL'),
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
        stream=False
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    ret = asyncio.run(raw_llm_query("Hello"))
    print(ret)