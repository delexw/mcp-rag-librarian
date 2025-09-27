"""
Configuration Manager - Single Responsibility for configuration handling.
"""

from typing import Dict, Any, Optional
from ..interfaces.knowledge_base_interfaces import ConfigManagerInterface


class ConfigManager(ConfigManagerInterface):
    """
    Manages application configuration following Single Responsibility Principle.

    Responsibilities:
    - Configuration loading and validation
    - Default value resolution
    - Parameter overriding logic
    """

    def __init__(self, default_values: Dict[str, Any]):
        """Initialize with default configuration values."""
        self._defaults = default_values.copy()
        self._overrides = {}

    def set_override(self, key: str, value: Any):
        """Set configuration override."""
        self._overrides[key] = value

    def set_overrides(self, overrides: Dict[str, Any]):
        """Set multiple configuration overrides."""
        self._overrides.update(overrides)

    def get_value(self, key: str, provided_value: Optional[Any] = None) -> Any:
        """
        Get configuration value with proper precedence.

        Precedence (highest to lowest):
        1. provided_value (tool call parameter)
        2. override value (command line argument)
        3. default value
        """
        if provided_value is not None:
            return provided_value

        if key in self._overrides:
            return self._overrides[key]

        if key in self._defaults:
            return self._defaults[key]

        raise ValueError(f"No configuration found for key: {key}")

    def get_all_values(self) -> Dict[str, Any]:
        """Get all configuration values with overrides applied."""
        result = self._defaults.copy()
        result.update(self._overrides)
        return result

    def validate_knowledge_base_path(self, provided_path: Optional[str] = None) -> str:
        """
        Validate and resolve knowledge base path.

        Args:
            provided_path: Path provided in tool call (optional)

        Returns:
            Resolved absolute knowledge base path (consistent format)

        Raises:
            ValueError: If no path is provided and no default is set
        """
        from pathlib import Path

        if provided_path:
            # Always resolve to absolute path for consistency
            return str(Path(provided_path).resolve())

        default_path = self._overrides.get("knowledge_base_path") or self._defaults.get("knowledge_base_path")
        if default_path:
            # Already absolute from server startup, but ensure it's resolved
            return str(Path(default_path).resolve())

        raise ValueError(
            "No knowledge base path provided and no default path configured. "
            "Either provide knowledge_base_path in the tool call or start the server with --knowledge-base"
        )

    def get_embedding_model(self, provided_model: Optional[str] = None) -> str:
        """Get embedding model with fallback."""
        return self.get_value("embedding_model", provided_model)

    def get_chunk_size(self, provided_size: Optional[int] = None) -> int:
        """Get chunk size with fallback."""
        return self.get_value("chunk_size", provided_size)

    def get_chunk_overlap(self, provided_overlap: Optional[int] = None) -> int:
        """Get chunk overlap with fallback."""
        return self.get_value("chunk_overlap", provided_overlap)

    def get_top_k(self, provided_top_k: Optional[int] = None) -> int:
        """Get top_k with fallback."""
        return self.get_value("top_k", provided_top_k)

    def is_persistence_enabled(self) -> bool:
        """Check if persistence is enabled."""
        return self.get_value("persist_cache", False)