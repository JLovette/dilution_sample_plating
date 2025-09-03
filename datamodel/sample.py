from enum import Enum
from dataclasses import dataclass
from typing import Optional

class SampleType(Enum):
    ADULT = "ADULT"
    CHICK = "CHICK"
    BLANK = "BLANK"

@dataclass
class Sample:
    sample_id: str
    unique_id: str
    colony_code: str
    sample_type: Optional[SampleType] = None
    # Row index of this sample in the original CSV (for ordering down columns)
    source_row_index: Optional[int] = None
