import re


class TextCleaner:
    @staticmethod
    def clean_text(text: str, preserve_typos: bool = True) -> str:
        if not isinstance(text, str):
            return ""
        # Remove null characters or extreme control chars but keep normal punctuation, emojis, typos
        cleaned = text.replace("\0", "").strip()

        # Collapse excessive whitespace
        cleaned = re.sub(r"\s+", " ", cleaned)

        return cleaned

    @staticmethod
    def extract_brand_mentions(text: str) -> list[str]:
        if not isinstance(text, str):
            return []
        return re.findall(r"@([A-Za-z0-9_]+)", text)
