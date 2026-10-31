from langchain_groq import ChatGroq

from app.core.logger import logger


class LLMGateway:
    """
    Central abstraction for LLM providers.

    Primary:
        Groq

    Optional fallback:
        Ollama
    """

    def __init__(
        self,
        groq_api_key: str | None,
        groq_model: str,
        temperature: float = 0.0,
        ollama_model: str = "llama3.2:3b",
    ):

        self.groq = None
        self.ollama = None

        # -----------------------------------------
        # Groq
        # -----------------------------------------

        if groq_api_key:

            self.groq = ChatGroq(
                api_key=groq_api_key,
                model=groq_model,
                temperature=temperature,
            )

            logger.info(
                f"Groq provider initialized: {groq_model}"
            )

        # -----------------------------------------
        # Ollama
        # -----------------------------------------

        try:

            from langchain_ollama import ChatOllama

            self.ollama = ChatOllama(
                model=ollama_model,
                temperature=temperature,
            )

            logger.info(
                f"Ollama fallback configured: {ollama_model}"
            )

        except Exception as exc:

            logger.warning(
                f"Ollama fallback unavailable: {exc}"
            )

    def generate(self, prompt: str) -> tuple[str, str]:

        # -----------------------------------------
        # Primary: Groq
        # -----------------------------------------

        if self.groq:

            try:

                response = self.groq.invoke(prompt)

                return response.content, "groq"

            except Exception as exc:

                logger.error(
                    f"Groq failed: {exc}"
                )

        # -----------------------------------------
        # Fallback: Ollama
        # -----------------------------------------

        if self.ollama:

            try:

                response = self.ollama.invoke(prompt)

                return response.content, "ollama"

            except Exception as exc:

                logger.error(
                    f"Ollama fallback failed: {exc}"
                )

        raise RuntimeError(
            "All configured LLM providers failed."
        )