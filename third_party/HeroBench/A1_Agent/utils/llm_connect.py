import os

from openai import OpenAI


class LLMConnect:
    """
    Connection class for interacting with LLM models via OpenRouter API.
    
    This class provides a standardized interface for sending messages to
    LLM models through the OpenRouter API, handling authentication and
    response formatting.
    """
    
    def __init__(self,model):
        """
        Initialize the LLM connection with model configuration.
        
        Args:
            model: Name of the LLM model to use for communication
        """
        key = os.environ.get('OPENAI_KEY')
        self.model = model
        self.client = OpenAI(
            base_url='https://openrouter.ai/api/v1',
            api_key=key)

    def send_message(self, system_prompt: str , user_prompt: str) -> str:
        """
        Send a message to the LLM and get the response.
        
        This method sends both a system prompt and user prompt to the LLM
        and returns the generated response. It uses zero temperature for
        deterministic responses.
        
        Args:
            system_prompt: System prompt that defines the LLM's behavior
            user_prompt: User prompt containing the actual request
            
        Returns:
            String containing the LLM's response
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {'role':'system', 'content': system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature = 0
            )
        return response.choices[0].message.content