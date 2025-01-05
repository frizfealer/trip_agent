from langchain_openai import ChatOpenAI


class ChatOpenAIFactory:
    def __init__(
        self,
        openai_api_key: str = "",
    ):
        self.openai_api_key = openai_api_key

    def create(self, **kwargs) -> ChatOpenAI:
        """Create a ChatOpenAI instance with the given parameters. Refer to https://platform.openai.com/docs/api-reference/chat/create for parameters."""
        params = {"openai_api_key": self.openai_api_key}
        params.update(kwargs)
        return ChatOpenAI(
            **params,
        )
