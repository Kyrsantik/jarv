import logging
from core import config
from core.ai import OllamaAI
from core.command_router import CommandRouter
from core.tts import TextToSpeach

logger = logging.getLogger("Jarvis.Assistant")


class Assistant:
    def __init__(self, stt, commands):
        self.stt = stt
        self.commands = commands
        self.config = getattr(config, "CONFIG", {})

        # Инициализируем синтез речи
        try:
            self.tts = TextToSpeach()
        except Exception as e:
            print(f"[TTS Ошибка] Не удалось запустить озвучку: {e}")
            self.tts = None

        # Инициализируем локальный ИИ
        try:
            self.ai = OllamaAI(self.config)
        except Exception as e:
            print(f"[AI Ошибка] Не удалось подключиться к Ollama: {e}")
            self.ai = None

        # Инициализируем роутер команд
        router_threshold = 70
        if isinstance(self.config, dict):
            router_threshold = self.config.get("router", {}).get("threshold", 70)
        
        self.router = CommandRouter(self.commands, threshold=router_threshold)

    def _ask_ai(self, text: str) -> str:
        """Безопасный запрос к модели Ollama"""
        if not self.ai:
            return "ИИ модуль недоступен."

        # Проверяем, какой метод реализован в OllamaAI
        if hasattr(self.ai, "ask"):
            return self.ai.ask(text)
        elif hasattr(self.ai, "chat"):
            return self.ai.chat(text)
        elif hasattr(self.ai, "generate"):
            return self.ai.generate(text)
        else:
            # Прямой вызов через клиент Ollama
            try:
                res = self.ai.client.chat(
                    model=self.ai.model,
                    messages=[{"role": "user", "content": text}]
                )
                return res["message"]["content"]
            except Exception as e:
                return f"Ошибка генерации: {e}"

    def say(self, text: str):
        """Озвучка ответа"""
        if self.tts and text:
            try:
                self.tts.say(text)
            except Exception as e:
                print(f"[TTS Ошибка]: {e}")

    def process_command(self, text: str):
        text = text.strip()
        if not text:
            return

        # Если это просто слово активации (wake word), откликаемся и ждем команду
        if text.lower() == getattr(self.stt, "wakeword_name", "jarvis"):
            print("[Джарвис]: Слушаю вас, сэр.")
            self.say("Слушаю вас")
            next_command = self.stt.listen(timeout=5)
            if next_command:
                self.process_command(next_command)
            return

        # 1. Проверяем системные команды (запуск приложений, громкость и т.д.)
        detected_commands = self.router.detect(text)

        if detected_commands:
            for action, score, phrase in detected_commands:
                print(f"[Команда]: {phrase} (совпадение: {score}%)")
                try:
                    if callable(action):
                        action()
                    elif hasattr(action, "execute"):
                        action.execute()
                except Exception as e:
                    print(f"[Ошибка выполнения команды]: {e}")
            return

        # 2. Если команда не найдена в списке — отправляем вопрос в Ollama
        print(f"\n[Вы]: {text}")
        reply = self._ask_ai(text)
        print(f"[Джарвис]: {reply}\n")

        # 3. Озвучиваем ответ голосом
        self.say(reply)

    def run(self):
        print("\n" + "="*50)
        print("🤖 Джарвис успешно запущен и слушает микрофон...")
        print("="*50 + "\n")

        while True:
            try:
                recognized_text = self.stt.listen()
                if recognized_text:
                    self.process_command(recognized_text)
            except Exception as e:
                print(f"[Ошибка в цикле]: {e}")
                continue