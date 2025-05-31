from enum import Enum
from dataclasses import dataclass

class SampleType(Enum):
    BLANK = "BLANK"
    ADULT = "ADULT"
    CHICK = "CHICK"

@dataclass
class Sample:
    sample_id: str
    unique_id: str
    colony_code: str
    sample_type: SampleType
