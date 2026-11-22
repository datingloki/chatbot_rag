"""Реэкспорт базового узла, чтобы узлы импортировали его из одного места."""
from polybot.pipeline.base_node import BaseNode

__all__ = ["BaseNode"]
