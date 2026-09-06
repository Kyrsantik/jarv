import logging
import re
import ollama

logger = logging.getLogger("Jarvis.AI")

class OllamaAI:
    """Локальный ИИ-модуль Jarvis на базе Ollama."""

    def __init__(self, config: dict):
        ai_cfg = config.get("ai", {})
        self.enabled = ai_cfg.get("enabled", True)
        self.host = ai_cfg.get("host", "http://localhost:11434")
        self.model = ai_cfg.get("model", "qwen2.5:3b")
        self.temperature = ai_cfg.get("temperature", 0.6)
        self.max_history_turns = ai_cfg.get("max_history_turns", 4)

        self.client = ollama.Client(host=self.host)

        # Системная инструкция для кратких ответов голосового ассистента
        self.system_prompt = {
            "role": "system",
            "content": (
                "Ты — голосовой помощник Джарвис. "
                "Отвечай кратко, емко, дружелюбно и строго по делу (не более 1-3 предложений). "
                "Не используй Markdown, списки, ссылки и спецсимволы — твой ответ пойдет напрямую в голосовой синтезатор речи."
            )
        }
        self.history = [self.system_prompt]

        if self.enabled:
            self._verify_connection()

    def _verify_connection(self):
        """Проверяет доступность Ollama и наличие выбранной модели."""
        try:
            available_models = [m["name"] for m in self.client.list().get("models", [])]
            # Учитываем, что имя модели может быть с тегом :latest
            matched = any(self.model in name for name in available_models)
            
            if not matched:
                logger.warning(
                    f"ВНИМАНИЕ: Модель '{self.model}' не найдена в Ollama! "
                    f"Выполните в терминале: ollama run {self.model}"
                )
            else:
                logger.info(f"Ollama успешно подключена. Активная модель: {self.model}")
        except Exception as e:
            logger.error(
                f"Не удалось подключиться к Ollama на {self.host}. "
                f"Убедитесь, что Ollama запущена! Ошибка: {e}"
            )

    def _clean_for_speech(self, text: str) -> str:
        """Очищает текст от разметки Markdown, скобок и лишних знаков перед озвучкой."""
        # Убираем жирный текст, курсив, заголовки, списки
        text = re.sub(r"[\*#_`~]", "", text)
        text = re.sub(r"\[.*?\]\(.*?\)", "", text) # убираем ссылки
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def ask(self, user_text: str) -> str:
        """Отправляет запрос пользователя в Ollama и возвращает текст для озвучки."""
        if not self.enabled:
            return "Искусственный интеллект отключен в настройках."

        self.history.append({"role": "user", "content": user_text})

        # Ограничиваем историю, сохраняя системный промпт
        if len(self.history) > (self.max_history_turns * 2 + 1):
            self.history = [self.system_prompt] + self.history[-(self.max_history_turns * 2):]

        try:
            response = self.client.chat(
                model=self.model,
                messages=self.history,
                options={
                    "temperature": self.temperature,
                    "num_predict": 150 # Ограничиваем генерацию короткими фразами
                }
            )
            raw_reply = response["message"]["content"]
            clean_reply = self._clean_for_speech(raw_reply)

            # Запоминаем ответ
            self.history.append({"role": "assistant", "content": clean_reply})
            return clean_reply

        except Exception as e:
            logger.error(f"Ошибка вызова Ollama: {e}")
            return "Сэр, произошла ошибка при обращении к локальной нейросети. Проверьте, запущена ли Ollama."

    def reset(self):
        """Сброс контекста разговора."""
        self.history = [self.system_prompt]