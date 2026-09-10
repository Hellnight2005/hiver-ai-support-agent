import json
from pathlib import Path
from typing import List, Optional, Dict
from pydantic import BaseModel, Field


class IntentDefinition(BaseModel):
    name: str
    definition: str
    examples: List[str]
    positive_signals: List[str] = Field(default_factory=list)
    negative_signals: List[str] = Field(default_factory=list)
    boundary: Optional[str] = None


class IntentTaxonomy(BaseModel):
    intents: List[IntentDefinition]

    @classmethod
    def load_from_file(cls, filepath: str = "artifacts/intent_taxonomy.json") -> "IntentTaxonomy":
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Intent taxonomy file not found at {filepath}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    def get_intent_names(self) -> List[str]:
        return [i.name for i in self.intents]

    def get_intent(self, name: str) -> Optional[IntentDefinition]:
        for i in self.intents:
            if i.name.lower() == name.lower():
                return i
        return None

    def validate_intent(self, name: str) -> bool:
        return name.lower() in [i.name.lower() for i in self.intents]
