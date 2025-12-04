# src/memory.py
from typing import Any, Dict, List


class ShortTermMemory:
    """
    Simple in-process short-term memory.

    Stores the current conversation / workflow state.
    """

    def __init__(self) -> None:
        self._store: Dict[str, Any] = {}

    def get(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._store[key] = value

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._store)


class LongTermMemory:
    """
    Very simple long-term memory.

    In a real system this might be:
    - a database
    - a vector store
    - a document store

    Here we just keep it as an append-only list of records in memory.
    """

    def __init__(self) -> None:
        self._records: List[Dict[str, Any]] = []

    def add_record(self, record: Dict[str, Any]) -> None:
        self._records.append(record)

    def all_records(self) -> List[Dict[str, Any]]:
        return list(self._records)
