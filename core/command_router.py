import re
from rapidfuzz import fuzz
from core import config


class CommandRouter:

    def __init__(self, commands: dict, threshold: int = 75):
        self.commands = commands
        self.threshold = threshold

        # Загружаем списки фильтров и разделителей
        self.filters = getattr(config, "FILTERS", ["пожалуйста", "ну", "давай", "короче", "типа"])
        self.separators = getattr(config, "SEPARATORS", [" и ", " потом ", " затем ", " далее "])

        # Кэшируем нормализованные варианты команд заранее
        self.preprocessed_commands = []
        for variants, action in self.commands.items():
            norm_variants = []
            if isinstance(variants, (list, tuple, set)):
                for v in variants:
                    norm_variants.append(self.normalize(v))
            else:
                norm_variants.append(self.normalize(str(variants)))
            self.preprocessed_commands.append((norm_variants, action))

    def normalize(self, text: str) -> str:
        """Приводит текст к нижнему регистру, заменяет ё->е и убирает мусор."""
        if not text:
            return ""

        text = text.lower().replace("ё", "е")
        # Убираем знаки препинания
        text = re.sub(r"[^\w\s]", " ", text)

        # Удаляем слова-паразиты строго как отдельные слова (границы \b)
        for f in self.filters:
            text = re.sub(rf"\b{re.escape(f)}\b", "", text)

        return re.sub(r"\s+", " ", text).strip()

    def split_phrases(self, text: str) -> list[str]:
        """Разделяет цепочку команд по союзам-разделителям."""
        parts = [text]
        for sep in self.separators:
            new_parts = []
            for p in parts:
                new_parts.extend(p.split(sep))
            parts = new_parts

        # Отсекаем фразы короче 4 символов (защита от случайных звуков)
        return [p.strip() for p in parts if len(p.strip()) >= 4]

    def detect(self, user_text: str):
        """Определяет, какую системную команду вызвал пользователь."""
        normalized_text = self.normalize(user_text)

        # Если фраза слишком короткая — сразу отсекаем
        if len(normalized_text) < 4:
            return []

        phrases = self.split_phrases(normalized_text)
        if not phrases:
            phrases = [normalized_text]

        final_commands = []

        for phrase in phrases:
            if len(phrase) < 4:
                continue

            best_score = 0
            best_action = None
            phrase_words = set(phrase.split())

            for norm_variants, action in self.preprocessed_commands:
                for norm_variant in norm_variants:
                    if not norm_variant:
                        continue

                    var_words = set(norm_variant.split())

                    # Защита: если сказано 1 короткое слово, а команда из 3-4 слов,
                    # не даем им ложно сработать
                    if len(phrase_words) == 1 and len(var_words) >= 2 and len(norm_variant) - len(phrase) > 6:
                        continue

                    # Считаем схожесть по двум метрикам
                    ratio_score = fuzz.ratio(phrase, norm_variant)
                    token_score = fuzz.token_sort_ratio(phrase, norm_variant)
                    score = max(ratio_score, token_score)

                    # Небольшой контролируемый бонус за точные совпадения слов
                    keyword_overlap = phrase_words & var_words
                    if keyword_overlap and score >= 60:
                        score += min(len(keyword_overlap) * 4, 12)

                    score = min(score, 100.0)

                    if score > best_score:
                        best_score = score
                        best_action = action

            # Добавляем команду только если набран порог совпадения
            if best_score >= self.threshold and best_action:
                final_commands.append((best_action, best_score, phrase))

        # Убираем дубликаты действий
        unique = []
        used = set()

        for act, score, phrase in final_commands:
            act_id = id(act)
            if act_id not in used:
                unique.append((act, score, phrase))
                used.add(act_id)

        return unique