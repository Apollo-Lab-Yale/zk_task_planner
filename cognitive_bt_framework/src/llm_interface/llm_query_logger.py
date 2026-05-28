"""
LLM Query Logger - Logs all LLM queries and responses with caller identification.

Strips base64 image data from logged prompts to keep logs readable.
Writes to a timestamped log file in the specified directory.
"""

import logging
import os
import re
from datetime import datetime
from typing import Any, List, Optional, Union


logger = logging.getLogger("llm_query_logger")


def _strip_image_data(obj: Any) -> Any:
    """
    Recursively strip base64 image embedding strings from prompt structures.
    Replaces image data URLs with a placeholder.
    """
    if isinstance(obj, str):
        # Replace data:image/...;base64,<long string> patterns
        return re.sub(
            r'data:image/[^;]+;base64,[A-Za-z0-9+/=]+',
            '[IMAGE_DATA_STRIPPED]',
            obj
        )
    elif isinstance(obj, dict):
        return {k: _strip_image_data(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_strip_image_data(item) for item in obj]
    return obj


def _extract_text_content(prompt: Any) -> str:
    """
    Extract a readable text representation from a prompt structure,
    with image data stripped.
    """
    cleaned = _strip_image_data(prompt)

    if isinstance(cleaned, str):
        return cleaned

    if isinstance(cleaned, list):
        parts = []
        for msg in cleaned:
            if isinstance(msg, dict):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                if isinstance(content, str):
                    parts.append(f"[{role}]: {content}")
                elif isinstance(content, list):
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                text_parts.append(item.get("text", ""))
                            elif item.get("type") == "image_url":
                                text_parts.append("[IMAGE]")
                        elif isinstance(item, str):
                            text_parts.append(item)
                    parts.append(f"[{role}]: {' '.join(text_parts)}")
                else:
                    parts.append(f"[{role}]: {content}")
            elif isinstance(msg, str):
                parts.append(msg)
        return "\n".join(parts)

    return str(cleaned)


class LLMQueryLogger:
    """
    Logger for LLM queries and responses.

    Logs are written to a timestamped file in the specified log directory.
    Each entry includes:
    - Timestamp
    - Caller identifier (who initiated the query)
    - Query content (with images stripped)
    - Response content
    """

    SEPARATOR = "=" * 80
    DEFAULT_LOG_DIR = "llm_logs"

    def __init__(self, enabled: bool = False, log_dir: Optional[str] = None):
        self.enabled = enabled
        self._query_count = 0
        self.log_file_path = None

        if enabled:
            self._setup_file_logging(log_dir)

    def _setup_file_logging(self, log_dir: Optional[str] = None):
        """Create log directory and configure file handler."""
        log_dir = log_dir or self.DEFAULT_LOG_DIR
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file_path = os.path.join(log_dir, f"llm_queries_{timestamp}.log")

        # Remove any existing handlers to avoid duplicates
        logger.handlers.clear()

        file_handler = logging.FileHandler(self.log_file_path)
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)

        # Prevent propagation to root logger (avoids duplicate stdout output)
        logger.propagate = False

        logger.info(f"LLM Query Log started at {datetime.now().isoformat()}")
        logger.info(f"Log file: {self.log_file_path}\n")

    def set_enabled(self, enabled: bool):
        self.enabled = enabled
        if enabled and self.log_file_path is None:
            self._setup_file_logging()

    def get_log_file_path(self) -> Optional[str]:
        """Return the path to the current log file, or None if not logging."""
        return self.log_file_path

    def log_query(self, caller: str, prompt: Any, response: Optional[str],
                  model: str = "", duration_ms: Optional[float] = None):
        """
        Log an LLM query and its response.

        Args:
            caller: Identifier for who initiated the query
                    (e.g., "TaskPlanner", "SkillGenerator.generate_skill")
            prompt: The prompt sent to the LLM (will have images stripped)
            response: The LLM response text
            model: Model name used
            duration_ms: Optional query duration in milliseconds
        """
        if not self.enabled:
            return

        self._query_count += 1
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        prompt_text = _extract_text_content(prompt)

        header = f"\n{self.SEPARATOR}\n"
        header += f"LLM QUERY #{self._query_count}\n"
        header += f"Timestamp: {timestamp}\n"
        header += f"Caller: {caller}\n"
        if model:
            header += f"Model: {model}\n"
        if duration_ms is not None:
            header += f"Duration: {duration_ms:.0f}ms\n"
        header += f"{self.SEPARATOR}"

        query_section = f"\n--- QUERY [{caller}] ---\n{prompt_text}"

        response_text = response if response else "[NO RESPONSE]"
        response_section = f"\n--- RESPONSE [{caller}] ---\n{response_text}"

        footer = f"\n{self.SEPARATOR}\n"

        logger.info(f"{header}{query_section}{response_section}{footer}")
