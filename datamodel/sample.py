from typing import Optional
from datetime import datetime
from dataclasses import dataclass

@dataclass
class Sample:
    """Class representing a penguin sample with its important attributes."""
    sample_id: str
    unique_id: str
    species: str
    colony_code: str
    collection_date: Optional[datetime]
    substrate: str
    substrate_group: str
    island: str
    area: Optional[str]
    breeding_period: str
    chick_age: Optional[float]
    notes: Optional[str]
    collaborator: str
    ad_chick: Optional[str] = None 