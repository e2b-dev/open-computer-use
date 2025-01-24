# Open Computer Use

A secure cloud Linux computer powered by [E2B Desktop Sandbox](https://github.com/e2b-dev/desktop/) and controlled by open-source LLMs.

https://github.com/user-attachments/assets/3837c4f6-45cb-43f2-9d51-a45f742424d4

## Features

- Uses [E2B](https://e2b.dev) for secure [Desktop Sandbox](https://github.com/e2b-dev/desktop)
- Supports [Meta Llama](https://www.llama.com/), [OS-Atlas](https://osatlas.github.io/)/[ShowUI](https://github.com/showlab/ShowUI) and [any LLM you want to integrate](#llm-support)!
- Operates the computer via the keyboard, mouse, and shell commands
- Live streams the display of the sandbox on the client computer
- User can pause and prompt the agent at any time
- Uses Ubuntu, but designed to work with any operating system

## Design

![Open Computer Use Architecture](./assets/architecture.png#gh-dark-mode-only)
![Open Computer Use Architecture](./assets/architecture-light.png#gh-light-mode-only)

The details of the design are laid out in this article: [How I taught an AI to use a computer](https://blog.jamesmurdza.com/how-i-taught-an-ai-to-use-a-computer)

## LLM support

Open Computer Use is designed to easily support new LLMs. The LLM and provider combinations are are defined in [models.py](https://github.com/e2b-dev/open-computer-use/blob/master/os_computer_use/models.py). Following the comments in this file, one can easily add any LLM and provider that adheres to the OpenAI API specification.

The list of tested models and providers currently includes:

| **Type**    | **Model**          | **Providers**                        |
|-------------|--------------------|---------------------------------------|
| Vision      | **Llama 3.2**      | **Fireworks**, OpenRouter, Llama API     |
| Vision      | Gemini 2.0 Flash   | Google                             |
| Action      | **Llama 3.3**      | **Fireworks**, Llama API                 |
| Action      | DeepSeek           | DeepSeek                             |
| Action      | Gemini 2.0 Flash   | Google                             |
| Grounding   | **OS-Atlas**       | **HuggingFace Spaces**                   |
| Grounding   | ShowUI             | HuggingFace Spaces                   |

The following lines of code in [models.py](/blob/master/os_computer_use/models.py) define the default LLMs and providers:

```
grounding_model = OSAtlasProvider()
vision_model = FireworksProvider("llama3.2")
action_model = FireworksProvider("llama3.3")
```

If you add a new model or provider, please make a PR to this repository!

## Get started

### Prerequisites

- Python 3.10 or later
- [git](https://git-scm.com/)
- [E2B API key](https://e2b.dev/dashboard?tab=keys)
- [Fireworks API key](https://fireworks.ai/account/api-keys)

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
# Get your API key here - https://e2b.dev/
E2B_API_KEY="your-e2b-api-key"
FIREWORKS_API_KEY="your-fireworks-api-key"
```

### 4. Start the web interface

Run the following command to start the agent:

```sh
poetry install
```

```sh
poetry run start
```

The agent will start and prompt you for its first instruction.
