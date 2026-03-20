"""
Multi-turn conversation management module for the RAG assistant.

Maintains conversation history, builds structured prompts compatible with
Qwen2.5 message format, and formats retrieved documents into context.
"""

import logging
from typing import List, Dict, Any, Optional
from collections import deque

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent))
from src.config import Settings

logger = logging.getLogger(__name__)

settings = Settings()


class ConversationManager:
    """
    Manages multi-turn conversation state and builds RAG prompts
    in the Qwen2.5 messages format.
    """

    def __init__(
        self,
        max_turns: int = settings.MAX_HISTORY_TURNS,
        system_prompt: str = settings.SYSTEM_PROMPT,
    ):
        """
        Initialize the conversation manager.

        Args:
            max_turns: Maximum number of conversation turns (user+assistant pairs)
                       to keep in history. Older turns are dropped on overflow.
            system_prompt: Default system prompt for the assistant.
        """
        self.max_turns = max_turns
        self.system_prompt = system_prompt
        # Each element is {"role": str, "content": str}
        self._history: deque = deque(maxlen=max_turns * 2)

    def add_message(self, role: str, content: str) -> None:
        """
        Append a message to the conversation history.

        Args:
            role: Message role, one of "user", "assistant", or "system".
            content: Message content text.
        """
        if role not in ("user", "assistant", "system"):
            logger.warning("Unknown role '%s'; expected 'user', 'assistant', or 'system'.", role)

        self._history.append({"role": role, "content": content})

    def get_history(self) -> List[Dict[str, str]]:
        """
        Return a copy of the current conversation history.

        Returns:
            List of message dicts with "role" and "content" keys,
            ordered from oldest to newest.
        """
        return list(self._history)

    def clear(self) -> None:
        """Clear all conversation history."""
        self._history.clear()
        logger.info("Conversation history cleared.")

    @staticmethod
    def build_context_from_docs(docs: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents into a context string for the prompt.

        Each document dict should contain fields such as:
            - "title" or "Product_Title": product name
            - "price" or "Price": product price
            - "rating" or "Rating": product rating
            - "description" or "Description": product description
            - "text": pre-formatted text (used as fallback)

        Args:
            docs: List of document dicts from the retrieval pipeline.

        Returns:
            Formatted context string.
        """
        if not docs:
            return ""

        context_parts: List[str] = []

        for idx, doc in enumerate(docs, start=1):
            title = doc.get("title") or doc.get("Product_Title") or ""
            price = doc.get("price") or doc.get("Price") or ""
            rating = doc.get("rating") or doc.get("Rating") or ""
            description = doc.get("description") or doc.get("Description") or ""

            # If structured fields are available, format them nicely
            if title:
                lines = [f"[{idx}] {title}"]
                if price:
                    lines.append(f"    Price: {price}")
                if rating:
                    lines.append(f"    Rating: {rating}")
                if description:
                    desc_text = str(description)[:200]
                    lines.append(f"    Description: {desc_text}")
                context_parts.append("\n".join(lines))
            elif doc.get("text"):
                # Fallback: use the raw text field
                context_parts.append(f"[{idx}] {doc['text'][:300]}")
            else:
                logger.debug("Document at index %d has no usable content fields.", idx)

        return "\n\n".join(context_parts)

    def build_rag_prompt(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Build a complete RAG prompt in the Qwen2.5 messages format.

        Structure:
            1. System message (with instructions and retrieved context)
            2. Conversation history (previous turns)
            3. Current user query

        Args:
            query: The current user query.
            retrieved_docs: Documents retrieved for the current query.
            system_prompt: Override system prompt. Uses the default if None.

        Returns:
            List of message dicts compatible with Qwen2.5:
            [{"role": "system", ...}, {"role": "user", ...}, ...]
        """
        prompt = system_prompt or self.system_prompt
        context_text = self.build_context_from_docs(retrieved_docs)

        # Build the system message with retrieved context
        if context_text:
            system_content = (
                f"{prompt}\n\n"
                f"The following product information was retrieved based on the user's query. "
                f"Use this information to provide accurate answers:\n\n"
                f"{context_text}"
            )
        else:
            system_content = (
                f"{prompt}\n\n"
                f"No relevant product information was found for this query. "
                f"Please let the user know and try to help based on general knowledge."
            )

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_content},
        ]

        # Append conversation history
        for msg in self._history:
            messages.append({"role": msg["role"], "content": msg["content"]})

        # Append the current user query
        messages.append({"role": "user", "content": query})

        return messages
