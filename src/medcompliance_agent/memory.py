# src/medcompliance_agent/memory.py

from typing import Any, Dict, List
import json
from pathlib import Path


class ShortTermMemory:
    """
    Stores state only during this run.
    Lives in RAM. Reset each time.
    """
    def __init__(self):
        self.store: Dict[str, Any] = {}

    def set(self, key: str, value: Any):
        self.store[key] = value

    def get(self, key: str, default=None):
        return self.store.get(key, default)

    def to_dict(self):
        return dict(self.store)


class LongTermMemory:
    """
    Persistent memory stored on disk as JSON.
    Good for saving device profiles, past checklists, etc.
    """

    def __init__(self, path: Path):
        self.path = path
        self.records: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                self.records = json.loads(self.path.read_text())
            except Exception:
                self.records = []
        else:
            self.records = []

    def add_record(self, record: Dict[str, Any]):
        self.records.append(record)
        self._save()

    def _save(self):
        self.path.write_text(json.dumps(self.records, indent=2))

    def all(self) -> List[Dict[str, Any]]:
        return list(self.records)


class CheckpointStore:
    """
    Saves agent's last known state.
    """

    def __init__(self, path: Path):
        self.path = path

    def save(self, state: Dict[str, Any]):
        self.path.write_text(json.dumps(state, indent=2))

    def load(self) -> Dict[str, Any]:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text())
            except Exception:
                return {}
        return {}
