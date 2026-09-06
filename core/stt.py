import json
import os
import sys
import numpy as np
import pyaudio
from vosk import KaldiRecognizer, Model as VoskModel

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

try:
    from openwakeword.model import Model as OwwModel
    import openwakeword.utils
except ImportError:
    print("\n[Ошибка] Библиотека openwakeword не установлена!")
    print("Выполните: py -3.11 -m pip install openwakeword\n")
    sys.exit(1)


class SpeachToText:
    def __init__(self, config=None):
        if not isinstance(config, dict):
            from core.config import CONFIG
            config = CONFIG

        print("[STT] Загрузка локальной модели Vosk...")
        model_path = os.path.join(os.path.dirname(__file__), "..", "models", "small")
        if not os.path.exists(model_path):
            model_path = "models/small"

        self.vosk_model = VoskModel(model_path)
        self.recognizer = KaldiRecognizer(self.vosk_model, 16000)

        assistant_cfg = config.get("assistant", {}) if isinstance(config, dict) else {}
        self.wakeword_name = assistant_cfg.get("wakeword", "jarvis").lower()
        config_threshold = assistant_cfg.get("confidence_threshold", 75)
        self.threshold = config_threshold / 100.0

        print(f"[STT] Загрузка нейросети wake word для слова '{self.wakeword_name}'...")
        try:
            self.ww_model = OwwModel(wakeword_models=[self.wakeword_name], inference_framework="onnx")
        except Exception:
            print("[STT] Загрузка недостающих моделей wake word...")
            openwakeword.utils.download_models()
            self.ww_model = OwwModel(wakeword_models=[self.wakeword_name], inference_framework="onnx")

        # Инициализация аудиопотока с микрофона
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=1280,
        )
        self.stream.start_stream()
        print("[STT] Все системы успешно запущены локально и бесплатно!")

    def listen(self, timeout=None, **kwargs):
        """
        Слушает микрофон и возвращает распознанный текст.
        Если задан timeout, ждет команду указанное количество секунд.
        """
        import time

        start_time = time.time()
        recognized_text = ""

        # Сбрасываем предыдущее состояние распознавателя
        self.recognizer.Reset()

        while True:
            # Проверка выхода по таймауту
            if timeout is not None and (time.time() - start_time) > timeout:
                break

            try:
                data = self.stream.read(1280, exception_on_overflow=False)
                if len(data) == 0:
                    continue

                # 1. Проверяем распознавание речи Vosk
                if self.recognizer.AcceptWaveform(data):
                    res = json.loads(self.recognizer.Result())
                    text = res.get("text", "").strip()
                    if text:
                        return text

                # 2. Проверяем wake word (если нужно вернуть само слово активации)
                audio_data = np.frombuffer(data, dtype=np.int16)
                prediction = self.ww_model.predict(audio_data)
                if prediction.get(self.wakeword_name, 0.0) >= self.threshold:
                    return self.wakeword_name

            except Exception:
                continue

        # Если время вышло, проверяем финальный остаток распознавания
        try:
            final_res = json.loads(self.recognizer.FinalResult())
            recognized_text = final_res.get("text", "").strip()
        except Exception:
            recognized_text = ""

        return recognized_text if recognized_text else None

    def listen_for_keyword(self, pcm_frame=None):
        if pcm_frame is None:
            pcm_frame = self.stream.read(1280, exception_on_overflow=False)
        try:
            audio_data = np.frombuffer(pcm_frame, dtype=np.int16)
            prediction = self.ww_model.predict(audio_data)
            return prediction.get(self.wakeword_name, 0.0) >= self.threshold
        except Exception:
            return False

    def recognize_speech(self, pcm_frame=None):
        if pcm_frame is None:
            pcm_frame = self.stream.read(1280, exception_on_overflow=False)
        if self.recognizer.AcceptWaveform(pcm_frame):
            result = json.loads(self.recognizer.Result())
            return result.get("text", "")
        return ""

    def stop(self):
        try:
            if hasattr(self, "stream") and self.stream.is_active():
                self.stream.stop_stream()
                self.stream.close()
            if hasattr(self, "p"):
                self.p.terminate()
        except Exception:
            pass