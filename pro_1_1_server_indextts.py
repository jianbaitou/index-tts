import os
import io
from flask import Flask, request, send_file, jsonify
import torchaudio
import torch
import json
import numpy as np
import io
import flask
from indextts.infer_v2 import IndexTTS2


app = Flask(__name__)

print("Loading IndexTTS-2 model...")
tts = IndexTTS2(cfg_path="D:/prodect_voice/index-tts/checkpoints/config.yaml",
                model_dir="D:/prodect_voice/index-tts/checkpoints/")
print("Model loaded.")


# 获取上传的音频文件
def extract_audio(audio_name):
    if audio_name not in request.files:
        return None

    audio_file = request.files[audio_name]
    if audio_file.filename == '':
        return None

    # 确保 temp 目录存在
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)  # 添加这一行
    # 保存上传的音频到临时文件
    audio_path = f"temp/{audio_file.filename}"
    audio_file.save(audio_path)
    return audio_path

@app.route('/generate_refer_emotional_audio_auto', methods=['POST'])
def generate_refer_emotional_audio_auto():
    """
    使用 use_emo_text 自动生成情感向量
    """
    payload_str = request.form.get("payload")
    if payload_str is None:
        return jsonify({"error": "no payload"}), 400
    data = json.loads(payload_str)

    text = data.get('text')
    if not text:
        return jsonify({"error": "no text provided"}), 400

    spk_audio_name = 'prompt_audio'
    spk_audio_path = extract_audio(spk_audio_name)
    if not spk_audio_path:
        return jsonify({"error": "no prompt_audio provided"}), 400


    sr, wav_np = tts.infer(
        spk_audio_prompt=spk_audio_path,
        text=text,
        output_path=None,
        emo_alpha=0.6,
        use_emo_text=True,
        use_random=False,
        stream_return=False,
        verbose=True
    )

    return handle_result_audio(sr, wav_np)


@app.route('/generate_refer_emotional_audio', methods=['POST'])
def generate_refer_emotional_audio():
    """
    同时模仿 音色音频 和 情绪音频
    """
    payload_str = request.form.get("payload")
    if payload_str is None:
        return jsonify({"error": "no payload"}), 400
    data = json.loads(payload_str)

    text = data.get('text')
    if not text:
        return jsonify({"error": "no text provided"}), 400

    spk_audio_name = 'prompt_audio'
    spk_audio_path = extract_audio(spk_audio_name)
    if not spk_audio_path:
        return jsonify({"error": "no prompt_audio provided"}), 400

    emo_audio_name = 'emotion_audio'
    emo_audio_path = extract_audio(emo_audio_name)
    if not emo_audio_path:
        return jsonify({"error": "no emotion_audio provided"}), 400

    sr, wav_np = tts.infer(
        spk_audio_prompt=spk_audio_path,
        text=text,
        output_path=None,
        emo_audio_prompt=emo_audio_path,
        stream_return=False,
        verbose=True
    )

    return handle_result_audio(sr, wav_np)

@app.route('/generate_refer_emotional_audio_and_prompt_text', methods=['POST'])
def generate_refer_emotional_audio_and_prompt_text():
    """
    同时模仿 音色音频 和 用文字引导情绪
    """
    payload_str = request.form.get("payload")
    if payload_str is None:
        return jsonify({"error": "no payload"}), 400
    data = json.loads(payload_str)

    text = data.get('text')
    if not text:
        return jsonify({"error": "no text provided"}), 400

    emo_text = data.get('emo_text')
    if not emo_text:
        return jsonify({"error": "no emo_text provided"}), 400

    spk_audio_name = 'prompt_audio'
    spk_audio_path = extract_audio(spk_audio_name)
    if not spk_audio_path:
        return jsonify({"error": "no prompt_audio provided"}), 400


    sr, wav_np = tts.infer(
	spk_audio_prompt=spk_audio_path,
	text=text,
	output_path=None,
        emo_alpha=0.3,
	use_emo_text=True,
	emo_text=emo_text,
	use_random=False,
	verbose=True
    )

    return handle_result_audio(sr, wav_np)


@app.route('/generate_emotion_vector', methods=['POST'])
def generate():
    """
    接收 JSON 请求，格式例如：
    {
      "text": "今天的天气真是太好了！",
      "prompt_audio": <上传的音频文件>,
      # 情感向量：[高兴, 愤怒, 悲伤, 害怕, 厌恶, 忧郁, 惊讶, 平静] [happy, angry, sad, afraid, hate, melancholy, surprise, calm]
      "emotion_vector": [0.0, 0.0, 1, 0.0, 0.0, 0.0, 0.0, 0.0])
    }
    """
    # 2️⃣ 取 payload
    payload_str = request.form.get("payload")
    if payload_str is None:
        return jsonify({"error": "no payload"}), 400

    data = json.loads(payload_str)

    text = data.get('text')
    emotion_vector = data.get('emotion_vector')
    if not text:
        return jsonify({"error": "no text provided"}), 400

    prompt_audio_name = 'prompt_audio'
    prompt_audio_path = extract_audio(prompt_audio_name)
    if not prompt_audio_path:
        return jsonify({"error": "no prompt_audio provided"}), 400

    try:
        print(type(text), text)
        print(type(emotion_vector), emotion_vector)
        sr, wav_np = tts.infer(
            spk_audio_prompt=prompt_audio_path,  # 确保 infer 内部能读 file-like；否则见下面备注
            text=text,
            emo_vector=emotion_vector,
            output_path=None,  # 关键：不落盘
            stream_return=False
        )

        return handle_result_audio(sr, wav_np)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def handle_result_audio(sr, wav_np):
    """
    修复版：纯 scipy 保存 WAV，彻底抛弃坏的 torchaudio
    """
    try:
        # 1. 归一化音频（保证音量正常）
        wav_np = wav_np / np.max(np.abs(wav_np))
        wav_int16 = (wav_np * 32767).astype(np.int16)

        # 2. 用 scipy 写入内存缓冲区（无依赖报错）
        import scipy.io.wavfile as wavfile
        buffer = io.BytesIO()
        wavfile.write(buffer, sr, wav_int16)
        buffer.seek(0)

        # 3. 返回音频文件
        return flask.send_file(
            buffer,
            mimetype="audio/wav",
            as_attachment=True,
            download_name="output.wav"
        )

    except Exception as e:
        print("保存音频失败:", e)
        return flask.jsonify({"error": "保存音频失败"}), 500


if __name__ == '__main__':
    # 你可以配置 host、端口等
    app.run(host='0.0.0.0', port=5200, debug=False)
