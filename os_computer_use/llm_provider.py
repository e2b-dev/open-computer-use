from openai import OpenAI
from anthropic import Anthropic

from PIL import Image
import io
import json
import re
import base64


def Message(content, role="assistant"):
    return {"role": role, "content": content}


def Text(text):
    return {"type": "text", "text": text}


def parse_json(s):
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        print(f"Error decoding JSON for tool call arguments: {s}")
        return None


def extract_json_objects(s):
    """Extract all balanced JSON objects from a string."""
    objects = []
    brace_level = 0
    start_index = None
    for i, char in enumerate(s):
        if char == "{":
            if brace_level == 0:
                start_index = i
            brace_level += 1
        elif char == "}":
            brace_level -= 1
            if brace_level == 0 and start_index is not None:
                candidate = s[start_index : i + 1]
                try:
                    obj = json.loads(candidate)
                    objects.append(obj)
                except json.JSONDecodeError:
                    pass
                start_index = None
    return objects


class LLMProvider:
    """
    The LLM provider is used to make calls to an LLM given a provider and model name, with optional tool use support
    """

    # Class attributes for base URL and API key
    base_url = None
    api_key = None

    # Mapping of model aliases
    aliases = {}

    # Initialize the API client
    def __init__(self, model):
        self.model = self.aliases.get(model, model)
        print(f"Using {self.__class__.__name__} with {self.model}")
        self.client = self.create_client()

    # Convert our function schema to the provider's required format
    def create_function_schema(self, definitions):
        functions = []

        for name, details in definitions.items():
            properties = {}
            required = []

            for param_name, param_desc in details["params"].items():
                properties[param_name] = {"type": "string", "description": param_desc}
                required.append(param_name)

            # Add a dummy property if no parameters are provided, because providers like Gemini require a non-empty "properties" object.
            if not properties:
                properties["noop"] = {
                    "type": "string",
                    "description": "Dummy parameter for function with no parameters.",
                }

            function_def = self.create_function_def(name, details, properties, required)
            functions.append(function_def)

        return functions

    # Represent a tool call as an object
    def create_tool_call(self, name, parameters):
        return {
            "type": "function",
            "name": name,
            "parameters": parameters,
        }

    # Wrap a content block in a text or an image object
    def wrap_block(self, block):
        if isinstance(block, bytes):
            # Pass raw bytes so that imghdr can detect the image type properly.
            return self.create_image_block(block)
        else:
            return Text(block)

    # Wrap all blocks in a given input message
    def transform_message(self, message):
        content = message["content"]
        if isinstance(content, list):
            wrapped_content = [self.wrap_block(block) for block in content]
            return {**message, "content": wrapped_content}
        else:
            return message

    # Create a chat completion using the API client
    def completion(self, messages, **kwargs):
        # Skip the tools parameter if it's None
        filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        # Wrap content blocks in image or text objects if necessary
        new_messages = [self.transform_message(message) for message in messages]
        # Call the inference provider
        completion = self.client.create(
            messages=new_messages, model=self.model, **filtered_kwargs
        )
        # Check for errors in the response
        if hasattr(completion, "error"):
            raise Exception("Error calling model: {}".format(completion.error))
        return completion


class OpenAIBaseProvider(LLMProvider):

    def create_client(self):
        return OpenAI(base_url=self.base_url, api_key=self.api_key).chat.completions

    def create_function_def(self, name, details, properties, required):
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": details["description"],
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def create_image_block(self, image_data: bytes):
        # Use Pillow to detect the image type
        image_type = "png"  # Default to PNG if detection fails
        try:
            with Image.open(io.BytesIO(image_data)) as img:
                image_type = img.format.lower()
        except Exception as e:
            print(f"Error detecting image type: {e}")

        # Base64-encode the raw image bytes.
        encoded = base64.b64encode(image_data).decode("utf-8")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/{image_type};base64,{encoded}"},
        }

    def call(self, messages, functions=None):
        # If functions are provided, only return actions
        tools = self.create_function_schema(functions) if functions else None
        completion = self.completion(messages, tools=tools)
        message = completion.choices[0].message

        # Return response text and tool calls separately
        if functions:
            tool_calls = message.tool_calls or []
            combined_tool_calls = [
                self.create_tool_call(
                    tool_call.function.name, parse_json(tool_call.function.arguments)
                )
                for tool_call in tool_calls
                if parse_json(tool_call.function.arguments) is not None
            ]

            # Sometimes, function calls are returned unparsed by the inference provider. This code parses them manually.
            if message.content and not tool_calls:
                json_objs = extract_json_objects(message.content)
                for obj in json_objs:
                    parameters = obj.get("parameters", obj.get("arguments"))
                    if obj.get("name") and parameters is not None:
                        combined_tool_calls.append(
                            self.create_tool_call(obj.get("name"), parameters)
                        )
                if combined_tool_calls:
                    return None, combined_tool_calls

            return message.content, combined_tool_calls

        # Only return response text
        else:
            return message.content


class AnthropicBaseProvider(LLMProvider):

    def create_client(self):
        return Anthropic(api_key=self.api_key).messages

    def create_function_def(self, name, details, properties, required):
        return {
            "name": name,
            "description": details["description"],
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

    def create_image_block(self, base64_image):
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64_image,
            },
        }

    def call(self, messages, functions=None):
        tools = self.create_function_schema(functions) if functions else None

        # Move all messages with the system role to a system parameter
        system = "\n".join(
            msg.get("content") for msg in messages if msg.get("role") == "system"
        )
        messages = [msg for msg in messages if msg.get("role") != "system"]

        # Call the Anthropic API
        completion = self.completion(
            messages, system=system, tools=tools, max_tokens=4096
        )
        text = "".join(getattr(block, "text", "") for block in completion.content)

        # Return response text and tool calls separately
        if functions:
            tool_calls = [
                self.create_tool_call(block.name, block.input)
                for block in completion.content
                if block.type == "tool_use"
            ]
            return text, tool_calls

        # Only return response text
        else:
            return text


class MistralBaseProvider(OpenAIBaseProvider):
    def create_function_def(self, name, details, properties, required):
        # If description is wrapped in a dict, extract the inner string
        if isinstance(details.get("description"), dict):
            details["description"] = details["description"].get("description", "")
        return super().create_function_def(name, details, properties, required)

    def call(self, messages, functions=None):
        if messages and messages[-1].get("role") == "assistant":
            prefix = messages.pop()["content"]
            if messages and messages[-1].get("role") == "user":
                messages[-1]["content"] = (
                    prefix + "\n" + messages[-1].get("content", "")
                )
            else:
                messages.append({"role": "user", "content": prefix})
        return super().call(messages, functions)
