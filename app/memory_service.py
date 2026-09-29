"""
Memory Service — backward-compatible wrapper.
Delegates to app.services.memory_service for all memory operations.

Existing code that imports `from app.memory_service import memory_service`
will continue to work without any changes.
"""

from app.services.memory_service import MemoryService, memory_service

__all__ = ["MemoryService", "memory_service"]