# Define the models to use in the agent

from os_computer_use import providers

grounding_model = providers.OSAtlasProvider()
# grounding_model = providers.ShowUIProvider()

# vision_model = providers.FireworksProvider("llama-3.2")
# vision_model = providers.OpenAIProvider("gpt-4o")
# vision_model = providers.AnthropicProvider("claude-3.5-sonnet")
# vision_model = providers.MoonshotProvider("moonshot-v1-vision")
# vision_model = providers.MistralProvider("pixtral")
#vision_model = providers.GroqProvider("llama-3.2")
# vision_model = providers.DashScopeProvider("qwen-vl-max")
vision_model = providers.OpenRouterProvider("qwen-2.5-vl")

# action_model = providers.FireworksProvider("llama-3.3")
# action_model = providers.OpenAIProvider("gpt-4o")
# action_model = providers.AnthropicProvider("claude-3.5-sonnet")
# vision_model = providers.MoonshotProvider("moonshot-v1-vision")
# action_model = providers.MistralProvider("mistral")
# action_model = providers.DashScopeProvider("qwen-max")
action_model = providers.GroqProvider("llama-3.3")