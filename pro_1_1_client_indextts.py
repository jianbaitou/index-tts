import json
import os
import time

import requests


def prompt_audio_and_prompt_text_synthesize_and_save(
        text,
        prompt_wav_file,
        emotion_text,
        save_path="output.wav"
):
    """
    利用prompt音频 指定情绪音频
    调用TTS服务并保存音频
    :param text: 生成音频文本内容
    :param prompt_wav_file: 模仿音频文件路径
    :param emotion_text: 情绪引导内容
    :param save_path: 保存路径
    """

    url = "http://10.45.120.197:5200/generate_refer_emotional_audio_and_prompt_text"
    start_time = time.time()

    with open(prompt_wav_file, "rb") as f1:

        files = {
            "prompt_audio": (os.path.basename(prompt_wav_file), f1, 'application/octet-stream')
        }

        payload = {
            "text": text,
            "emo_text": emotion_text
        }

        data = {
            "payload": json.dumps(payload)  # 注意：转成字符串
        }

        try:
            response = requests.post(url, data=data, files=files)
        except Exception as e:
            print(f"请求异常: {e}")
            return False

    latency = time.time() - start_time
    print(f"TTS 请求耗时: {latency:.4f} 秒")

    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"✅ 音频已保存: {save_path}")
        return True
    else:
        print(f"❌ 请求失败: {response.status_code}, {response.text}")
        return False

if __name__ == '__main__':
    text = "我讨厌吃青菜"
    prompt_wav_file = "./test_10_1.wav"
    emotion_text = "生气的情绪"
    prompt_audio_and_prompt_text_synthesize_and_save(text,prompt_wav_file,emotion_text)
