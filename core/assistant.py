from core.ai import OllamaAI

class Assistant:
    def __init__(self, config):
        self.config = config
        self.ai = OllamaAI(config)
        # ... инициализация STT (Vosk), TTS (pyttsx3) ...

    def process_command(self, text: str):
        text = text.lower().strip()
        if not text:
            return

        # 1. Сначала проверяем, системная ли это команда (звук, приложения, выключение)
        if self.router.is_system_command(text):
            self.router.execute(text)
            return

        # 2. Если системная команда не распознана — отправляем в Ollama
        print(f"[Пользователь]: {text}")
        reply = self.ai.ask(text)
        print(f"[Джарвис]: {reply}")
        
        # 3. Озвучиваем голосом через pyttsx3
        self.tts.say(reply)