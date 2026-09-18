import os
import time
import urllib.request
import json
from typing import Dict, Any, Optional
from src.generation.base_llm import LLMProvider
from src.common.logger import logger

class LlamaCppProvider(LLMProvider):
    """LLM provider with a preference order of llama-cpp GGUF, then a local
    Ollama server, then a deterministic extractive fallback.

    The active backend is exposed via `provider_name` and `health_check()` so
    callers (and the benchmark report) can tell whether real generation ran.
    """

    DEFAULT_OLLAMA_HOST = "http://localhost:11434"
    DEFAULT_OLLAMA_MODEL = "llama3.2"

    def __init__(
        self,
        model_path: str = "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        ollama_model: Optional[str] = None,
        ollama_host: Optional[str] = None,
        prefer_ollama: bool = False,
        request_timeout: float = 120.0
    ):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.ollama_host = (ollama_host or os.environ.get("OLLAMA_HOST") or self.DEFAULT_OLLAMA_HOST).rstrip("/")
        self.ollama_model = ollama_model or os.environ.get("OLLAMA_MODEL") or self.DEFAULT_OLLAMA_MODEL
        self.prefer_ollama = prefer_ollama
        self.request_timeout = request_timeout
        self.llm = None
        self._active_backend = "extractive_fallback"

        # 1. Try loading a local GGUF model first (unless Ollama is preferred)
        if not prefer_ollama:
            self._try_load_llama_cpp()

        # 2. Fall back to a local Ollama server if no GGUF model is loaded
        if self.llm is None and self._ollama_available():
            self._active_backend = "ollama"
            logger.info(f"Using Ollama backend at '{self.ollama_host}' with model '{self.ollama_model}'.")

        if self._active_backend == "extractive_fallback":
            logger.warning(
                "No local LLM available (no GGUF model loaded and no Ollama server reachable). "
                "Operating in deterministic extractive fallback mode; latency and quality metrics "
                "from this mode are NOT valid for research benchmarking."
            )

    def _try_load_llama_cpp(self):
        if not os.path.exists(self.model_path):
            logger.info(f"GGUF model not found at '{self.model_path}'. Skipping llama-cpp backend.")
            return
        try:
            from llama_cpp import Llama
            logger.info(f"Initializing LlamaCpp from '{self.model_path}' (GPU layers={self.n_gpu_layers})...")
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                verbose=False
            )
            self._active_backend = "llama_cpp"
        except Exception as e:
            logger.warning(f"Could not load local GGUF model via llama-cpp-python ({e}).")

    def _ollama_available(self) -> bool:
        """Return True if an Ollama server is reachable AND serves the target model."""
        try:
            url = f"{self.ollama_host}/api/tags"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                if resp.status != 200:
                    return False
                data = json.loads(resp.read().decode("utf-8"))
        except Exception:
            return False

        names = [m.get("name", "") for m in data.get("models", [])]
        if not names:
            logger.warning(f"Ollama is reachable at '{self.ollama_host}' but has no models pulled.")
            return False

        # Exact match, or match ignoring the ":latest" tag suffix
        if self.ollama_model in names:
            return True
        for n in names:
            if n.split(":")[0] == self.ollama_model.split(":")[0]:
                self.ollama_model = n
                return True

        logger.warning(
            f"Ollama is reachable but model '{self.ollama_model}' is not pulled. "
            f"Available: {', '.join(names)}. Run: ollama pull {self.ollama_model}"
        )
        return False

    @property
    def provider_name(self) -> str:
        if self._active_backend == "llama_cpp":
            return f"llama_cpp ({os.path.basename(self.model_path)})"
        if self._active_backend == "ollama":
            return f"ollama ({self.ollama_model})"
        return "extractive_fallback (no LLM)"

    @property
    def is_real_llm(self) -> bool:
        return self._active_backend in ("llama_cpp", "ollama")

    def warm_up(self) -> bool:
        """Run one throwaway generation so the model is loaded into memory.

        Without this, the first measured query absorbs model cold-start cost
        (tens of seconds for a 3B model) and skews the benchmark mean/P95
        latency. Returns True if a real backend was warmed.
        """
        if not self.is_real_llm:
            return False
        logger.info(f"Warming up '{self.provider_name}'...")
        start = time.time()
        try:
            self.generate(
                prompt="Say OK.",
                system_prompt="You are a helpful assistant.",
                max_tokens=8,
                temperature=0.0
            )
        except Exception as e:
            logger.warning(f"Warm-up generation failed ({e}); continuing anyway.")
            return False
        logger.info(f"Warm-up complete in {round((time.time() - start) * 1000, 2)} ms.")
        return True

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 512,
        temperature: float = 0.1
    ) -> Dict[str, Any]:
        start_time = time.time()

        if self.llm:
            formatted_prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{prompt}\n<|assistant|>"
            res = self.llm(
                formatted_prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            text = res["choices"][0]["text"].strip()
            eval_tokens = res["usage"]["completion_tokens"]
        elif self._active_backend == "ollama":
            ollama_response = self._ollama_generate(prompt, system_prompt, max_tokens, temperature)
            if ollama_response is not None:
                text = ollama_response
            else:
                # Server was reachable at init but failed now; degrade for this call only.
                logger.warning("Ollama generation failed; using extractive fallback for this call.")
                text = self._extractive_fallback(prompt)
            eval_tokens = len(text.split())
        else:
            # Deterministic extractive fallback when no LLM backend is loaded.
            time.sleep(0.05)  # simulate generation latency
            text = self._extractive_fallback(prompt)
            eval_tokens = len(text.split())

        elapsed_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "text": text,
            "tokens_generated": eval_tokens,
            "latency_ms": elapsed_ms,
            "ttft_ms": round(elapsed_ms * 0.25, 2),  # estimated TTFT
            "backend": self._active_backend,
            "is_real_llm": self.is_real_llm
        }

    def _ollama_generate(
        self,
        prompt: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float
    ) -> Optional[str]:
        """Generate via the local Ollama HTTP API. Returns None on any failure."""
        try:
            url = f"{self.ollama_host}/api/generate"
            payload = {
                "model": self.ollama_model,
                "system": system_prompt,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=self.request_timeout) as resp:
                if resp.status == 200:
                    res_data = json.loads(resp.read().decode("utf-8"))
                    return res_data.get("response", "").strip()
        except Exception as e:
            logger.warning(f"Ollama generation request failed: {e}")
        return None

    def _extractive_fallback(self, prompt: str) -> str:
        """Extract answer directly from document context lines matching query keywords."""
        query = ""
        context_part = ""

        if "QUESTION:" in prompt:
            parts = prompt.split("QUESTION:")
            query = parts[1].split("\n")[0].strip().lower()
            if "CONTEXT:" in parts[0]:
                context_part = parts[0].split("CONTEXT:")[1].strip()
        elif "QUERY:" in prompt:
            parts = prompt.split("QUERY:")
            query = parts[1].split("\n")[0].strip().lower()
            if "CONTEXT CHUNKS:" in parts[0]:
                context_part = parts[0].split("CONTEXT CHUNKS:")[1].strip()

        if not context_part:
            context_part = prompt

        lines = []
        for line in context_part.split("\n"):
            line_str = line.strip()
            # Filter out metadata lines, chunk headers, and page numbers
            if (line_str 
                and not line_str.startswith("Source ") 
                and not line_str.startswith("[Chunk") 
                and not line_str.startswith("CONTEXT") 
                and not line_str.startswith("QUESTION")
                and not line_str.startswith("--- Page")):
                lines.append(line_str)

        if not lines:
            return "No text context available to extract answer."

        # Special handling for document title / paper name queries
        is_title_query = any(k in query for k in ["title", "name of paper", "paper title", "name of the paper", "what paper", "paper name"])
        if is_title_query:
            # Check for title candidates (e.g. text before 'Abstract' or first non-meta sentence)
            for i, line in enumerate(lines[:10]):
                if "Abstract" in line:
                    # Title is usually the preceding line(s)
                    title_lines = [l for l in lines[:i] if len(l) > 10 and not l.startswith("School of") and not l.startswith("Department")]
                    if title_lines:
                        return " ".join(title_lines[:2])
                if ":" in line and len(line) > 15 and not line.lower().startswith("query"):
                    # Check if line looks like "Adaptive-RAG: Learning to Adapt..."
                    return line

            # Fallback to very first clean sentence of document context
            for line in lines:
                if len(line) > 15:
                    return line

        # Standard keyword density search for general queries
        stopwords = {"what", "is", "the", "of", "in", "and", "a", "an", "for", "to", "this", "that", "it", "how"}
        query_words = [w for w in query.replace("?", "").replace("!", "").replace(":", "").split() if len(w) > 1 and w not in stopwords]
        
        matching_lines = []
        for line in lines:
            line_lower = line.lower()
            match_score = sum(1 for w in query_words if w in line_lower)
            if match_score > 0:
                matching_lines.append((match_score, line))

        if matching_lines:
            matching_lines.sort(key=lambda x: x[0], reverse=True)
            seen = set()
            unique_lines = []
            for score, line in matching_lines:
                if line not in seen:
                    seen.add(line)
                    unique_lines.append(line)
                if len(unique_lines) >= 3:
                    break
            answer = " ".join(unique_lines)
        else:
            answer = " ".join(lines[:2])

        if len(answer) > 500:
            answer = answer[:500] + "..."

        return answer

    def count_tokens(self, text: str) -> int:
        if self.llm:
            return len(self.llm.tokenize(text.encode("utf-8")))
        return len(text.split())

    def health_check(self) -> bool:
        """True only when a real generation backend is active."""
        return self.is_real_llm

