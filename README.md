# Open Computer Use

A secure cloud Linux computer powered by [E2B Desktop Sandbox](https://github.com/e2b-dev/desktop/) and controlled by open-source LLMs.

https://github.com/user-attachments/assets/3837c4f6-45cb-43f2-9d51-a45f742424d4

## Features

- Uses [E2B](https://e2b.dev) for secure [Desktop Sandbox](https://github.com/e2b-dev/desktop)
- Operates the computer via the keyboard, mouse, and shell commands
- Supports 10+ LLMs, [OS-Atlas](https://osatlas.github.io/)/[ShowUI](https://github.com/showlab/ShowUI) and [any other models you want to integrate](#llm-support)!
- Live streams the display of the sandbox on the client computer
- User can pause and prompt the agent at any time
- Uses Ubuntu, but designed to work with any operating system

## Design

![Open Computer Use Architecture](./assets/architecture.png#gh-dark-mode-only)
![Open Computer Use Architecture](./assets/architecture-light.png#gh-light-mode-only)

The details of the design are laid out in this article: [How I taught an AI to use a computer](https://blog.jamesmurdza.com/how-i-taught-an-ai-to-use-a-computer)

## LLM support

Open Computer Use is designed to make it easy to swap in and out new LLMs. The LLMs used by the agent are specified in [config.py](/os_computer_use/config.py) like this:

```
grounding_model = providers.OSAtlasProvider()
vision_model = providers.GroqProvider("llama3.2")
action_model = providers.GroqProvider("llama3.3")
```

The providers are imported from [providers.py](/os_computer_use/providers.py) and include:

- Fireworks, OpenRouter, Llama API:
  - Llama 3.2 (vision only), Llama 3.3 (action only)
- Groq:
  - Llama 3.2 (vision + action), Llama 3.3 (action only)
- DeepSeek:
  - DeepSeek (action only)
- Google:
  - Gemini 2.0 Flash (vision + action)
- OpenAI:
  - GPT-4o and GPT-4o mini (vision + action)
- Anthropic:
  - Claude (vision + action)
- HuggingFace Spaces:
  - OS-Atlas (grounding)
  - ShowUI (grounding)
- Moonshot
- DashScope
- Mistral AI (Pixtral for vision, Mistral Large for actions)

If you add a new model or provider, please [make a PR](../../pulls) to this repository with the updated providers.py!

## Get started

### Prerequisites

- Python 3.10 or later
- [git](https://git-scm.com/)
- [E2B API key](https://e2b.dev/dashboard?tab=keys)
- API key for an LLM provider (see above)

### 1. Install the prerequisites

In your terminal:

```sh
brew install poetry ffmpeg
```

### 2. Clone the repository

In your terminal:

```sh
git clone https://github.com/e2b-dev/open-computer-use/
```

### 3. Set the environment variables

Enter the project directory:

```
cd open-computer-use
```

Create a `.env` file in `open-computer-use` and set the following:

```sh
# Get your API key here: https://e2b.dev/
E2B_API_KEY="your-e2b-api-key"
```

Additionally, add API key(s) for any LLM providers you're using:
```
# You only need the API key for the provider(s) selected in config.py:
# Hugging Face Spaces do not require an API key.
FIREWORKS_API_KEY=...
OPENROUTER_API_KEY=...
LLAMA_API_KEY=...
GROQ_API_KEY=...
GEMINI_API_KEY=...
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
MOONSHOT_API_KEY=...
DASHSCOPE_API_KEY=...
# Required: Provide your Hugging Face token to bypass Gradio rate limits.
HF_TOKEN=...
```

### 4. Start the web interface

Run the following command to start the agent:

```sh
poetry install
```

```sh
poetry run start
```

The agent will open and prompt you for its first instruction.

To start the agent with a specified prompt, run:

```sh
poetry run start --prompt "use the web browser to get the current weather in sf"
```

The display stream should be visible a few seconds after the Python program starts.

