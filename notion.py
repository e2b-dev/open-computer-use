import os
import re
from notion_client import Client
from typing import Optional, Dict, Any

class NotionService:
    def __init__(self, api_key: Optional[str] = None):
        self.client = Client(auth=api_key or os.environ["NOTION_API_KEY"])

    def extract_page_id_from_url(self, url: str) -> str:
        """Extract the page ID from a Notion URL."""
        # Pattern to match Notion URLs and extract the page ID
        pattern = r"notion\.so/(?:[^/]+/)?([a-f0-9]{32}|[a-f0-9]{8}-?[a-f0-9]{4}-?[a-f0-9]{4}-?[a-f0-9]{4}-?[a-f0-9]{12})"
        match = re.search(pattern, url)

        if not match:
            # Try alternative pattern for URLs with page titles
            pattern = r"notion\.so/[^/]*?-([a-f0-9]{32})(?:\?|$)"
            match = re.search(pattern, url)

            if not match:
                raise ValueError(f"Invalid Notion URL: {url}")

        page_id = match.group(1)
        # Remove any hyphens that might be in UUID format
        page_id = page_id.replace("-", "")

        return page_id

    def get_page_content(self, url: str) -> Dict[Any, Any]:
        """Retrieve the content of a Notion page by its URL."""
        page_id = self.extract_page_id_from_url(url)

        try:
            # Retrieve the page content
            page_content = self.client.pages.retrieve(page_id)

            # Get the block children (actual content)
            blocks = self.client.blocks.children.list(page_id)

            # Combine page metadata with its content
            result = {
                "page": page_content,
                "blocks": blocks
            }

            return result
        except Exception as e:
            raise Exception(f"Error retrieving Notion page: {str(e)}")

    def get_page_text(self, url: str) -> str:
        """Extract just the text content from a Notion page."""
        content = self.get_page_content(url)

        # Extract text from blocks
        text_content = []
        for block in content["blocks"]["results"]:
            block_type = block.get("type")
            if block_type and block_type in block:
                if "text" in block[block_type]:
                    for text_item in block[block_type]["text"]:
                        if "plain_text" in text_item:
                            text_content.append(text_item["plain_text"])
                elif "rich_text" in block[block_type]:
                    for text_item in block[block_type]["rich_text"]:
                        if "plain_text" in text_item:
                            text_content.append(text_item["plain_text"])

        return "\n".join(text_content)


# Example usage
if __name__ == "__main__":
    notion_service = NotionService()
    try:
        page_text = notion_service.get_page_text("https://www.notion.so/Onboarding-1be17eb1d9e18036b16aef573ac99934?pvs=4")
        print(page_text)
    except Exception as e:
        print(f"Error: {e}")