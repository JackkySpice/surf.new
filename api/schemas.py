
from api.plugins import WebAgentType
from .utils.prompt import ClientMessage
from pydantic import BaseModel
from typing import List, Optional
from .models import ModelProvider
from .utils.types import AgentSettings, ModelSettings


class SessionRequest(BaseModel):
    agent_type: WebAgentType
    api_key: Optional[str] = None
    timeout: Optional[int] = 90000


class ChatRequest(BaseModel):
    session_id: str
    agent_type: WebAgentType
    provider: ModelProvider = ModelProvider.ANTHROPIC
    messages: List[ClientMessage]
    api_key: str = ""
    agent_settings: AgentSettings
    model_settings: ModelSettings