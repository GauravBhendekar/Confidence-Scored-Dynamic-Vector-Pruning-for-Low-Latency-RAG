import re

class TextCleaner:
    @staticmethod
    def clean(text: str) -> str:
        if not text:
            return ""
        # Remove null bytes or non-printable control characters except line breaks
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
        # Normalize carriage returns
        cleaned = cleaned.replace('\r\n', '\n').replace('\r', '\n')
        # Remove excessive blank lines (more than 2 consecutive newlines)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        # Strip trailing trailing whitespaces per line
        lines = [line.rstrip() for line in cleaned.split('\n')]
        return '\n'.join(lines).strip()
