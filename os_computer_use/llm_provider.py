from openai import OpenAI

from os_computer_use.llama_utils import (
    parse_llama_tool_calls,
    create_llama_function_list,
)


class LLMProvider:
    """
    The LLM provider is used to make calls to an LLM given a provider and model name, with optional tool use support
    """

    # Class attributes for base URL and API key
    base_url = None
    api_key = None

    # Mapping of model aliases
    aliases = {}

    def __init__(self, model):
        # Validate base URL and API key
        if not self.base_url:
            raise ValueError("No base URL provided.")
        if not self.api_key:
            raise ValueError("No API key provided.")
        self.model = self.aliases.get(model, model)
        print(f"Using {self.__class__.__name__} with {self.model}")
        # Initialize OpenAI client
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def completion(self, messages, tools=None):
        # Create a chat completion using the OpenAI client
        if tools is not None:
            completion = self.client.chat.completions.create(
                messages=messages, model=self.model, tools=tools
            )
        else:
            completion = self.client.chat.completions.create(
                messages=messages, model=self.model
            )
        # Check for errors in the response
        if hasattr(completion, "error"):
            raise Exception("Error calling model: {}".format(completion.error))
        if not completion.choices:
            raise ValueError("Invalid response from model: {}".format(completion))
        return completion

    def call(self, messages, functions=None):
        # If functions are provided, only return actions
        if functions:
            completion = self.completion(
                messages, create_llama_function_list(functions)
            )
            return parse_llama_tool_calls(
                completion.choices[0].message.content,
                completion.choices[0].message.tool_calls or [],
            )
        # Otherwise, return the completion text
        else:
            return self.completion(messages).choices[0].message.content
