# Define the models to use in the agent

from os_computer_use import providers
from os_computer_use.search import ParallelSearch

grounding_model = providers.OSAtlasProvider()
# grounding_model = providers.ShowUIProvider()

# vision_model = providers.FireworksProvider("llama-3.2")
# vision_model = providers.OpenAIProvider("gpt-4o")
# vision_model = providers.AnthropicProvider("claude-3.5-sonnet")
# vision_model = providers.MoonshotProvider("moonshot-v1-vision")
# vision_model = providers.MistralProvider("pixtral")
#vision_model = providers.GroqProvider("llama-3.2")
vision_model = providers.OpenRouterProvider("qwen-2.5-vl")

# action_model = providers.FireworksProvider("llama-3.3")
# action_model = providers.OpenAIProvider("gpt-4o")
# action_model = providers.AnthropicProvider("claude-3.5-sonnet")
# vision_model = providers.MoonshotProvider("moonshot-v1-vision")
# action_model = providers.MistralProvider("mistral")
action_model = providers.GroqProvider("llama-3.3")

# Direct web search/page fetching; set WEB_SEARCH_ENABLED=false to disable.
# Set this to None to disable, or supply a provider with search/fetch methods.
search_provider = ParallelSearch.from_env()
