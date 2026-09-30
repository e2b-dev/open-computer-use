from PIL import Image, ImageDraw
import json
import re


def draw_big_dot(image, coordinates, color="red", radius=12):
    draw = ImageDraw.Draw(image)
    x, y = coordinates
    bounding_box = [x - radius, y - radius, x + radius, y + radius]
    draw.ellipse(bounding_box, fill=color, outline=color)
    return image


class VisionGrounding:
    """Locate a UI element with a vision model and return a pixel point."""

    def __init__(self, provider):
        self.provider = provider

    def call(self, prompt, image_path):
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
        with Image.open(image_path) as image:
            width, height = image.size
        text = self.provider.call(
            [
                {
                    "role": "user",
                    "content": [
                        image_data,
                        f"""\
The screenshot is {width} pixels wide and {height} pixels tall. The origin is the top-left corner.
Locate this UI element: {prompt}
Reply with JSON only, using integer pixel coordinates: {{"x": <x>, "y": <y>}}
""",
                    ],
                }
            ]
        )
        return parse_point(text, width, height)


def _number(value):
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise TypeError(f"expected one number, got {value!r}")
        return _number(value[0])
    if isinstance(value, bool) or value is None:
        raise TypeError(f"expected a number, got {value!r}")
    return float(value)


def _pair(value):
    if isinstance(value, str):
        match = re.search(r"\{.*\}|\[.*\]", value, re.S)
        if not match:
            raise ValueError(f"Grounding model did not return a point: {value}")
        value = json.loads(match.group(0))
    if isinstance(value, dict):
        if "x" not in value and "point" in value:
            return _pair(value["point"])
        x, y = value["x"], value["y"]
        if (
            isinstance(x, (list, tuple))
            and isinstance(y, (list, tuple))
            and len(x) >= 2
            and len(y) >= 2
        ):
            return (_number(x[0]) + _number(x[1])) / 2, (_number(y[0]) + _number(y[1])) / 2
        return _number(x), _number(y)
    if isinstance(value, (list, tuple)):
        if len(value) >= 4:
            return (_number(value[0]) + _number(value[2])) / 2, (
                _number(value[1]) + _number(value[3])
            ) / 2
        if len(value) >= 2:
            return _number(value[0]), _number(value[1])
    raise TypeError(f"expected a point, got {value!r}")


def coerce_point(value, width, height):
    """Turn a model point into in-bounds pixel coordinates."""
    x, y = _pair(value)
    if 0 <= x <= 1 and 0 <= y <= 1 and (width > 1 or height > 1):
        x, y = x * width, y * height
    elif (x > width or y > height) and 0 <= x <= 1000 and 0 <= y <= 1000:
        x, y = x / 1000 * width, y / 1000 * height
    return max(0, min(width - 1, int(round(x)))), max(0, min(height - 1, int(round(y))))


def parse_point(text, width, height):
    return coerce_point(text, width, height)


def extract_bbox_midpoint(bbox_response):
    match = re.search(r"<\|box_start\|>(.*?)<\|box_end\|>", bbox_response)
    inner_text = match.group(1) if match else bbox_response
    numbers = [float(num) for num in re.findall(r"\d+\.\d+|\d+", inner_text)]
    if len(numbers) == 2:
        return numbers[0], numbers[1]
    elif len(numbers) >= 4:
        return (numbers[0] + numbers[2]) // 2, (numbers[1] + numbers[3]) // 2
    else:
        return None
