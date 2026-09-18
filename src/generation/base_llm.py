from abc import ABC, abstractmethod
from typing import Dict, Any, Generator

class LLMProvider(ABC):
    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 512,
        temperature: float = 0.1
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        pass

    @abstractmethod
    def health_check(self) -> bool:
        pass
