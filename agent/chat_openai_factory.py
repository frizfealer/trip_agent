from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel


class ChatOpenAIFactory:
    def __init__(
        self,
        openai_api_key: str = "",
    ):
        self.openai_api_key = openai_api_key

    def create(
        self,
        temperature: float = 0.7,
        top_p: float = 1.0,
        model_name: str = "gpt-4-turbo-preview",
        streaming: bool = False,
    ) -> BaseChatModel:
        """Create a ChatOpenAI instance with the given parameters. Refer to https://platform.openai.com/docs/api-reference/chat/create for parameters."""
        return ChatOpenAI(
            openai_api_key=self.openai_api_key,
            temperature=temperature,
            model=model_name,
            top_p=top_p,
            streaming=streaming,
        )
