import pandas as pd
from typing import List, Union
import io

from datamodel.sample import Sample, SampleType


def parse_samples(csv_file: Union[str, io.StringIO, io.BytesIO]) -> List[Sample]:
    """Parse the penguin sample data from a CSV file."""
    # Handle different input types
    if hasattr(csv_file, 'read'):  # File-like object (StringIO, BytesIO, StreamlitUploadedFile)
        df = pd.read_csv(csv_file)
    else:  # String path
        df = pd.read_csv(csv_file)
    
    samples = []
    for idx, row in df.iterrows():
        # Handle BLANK samples (no AD/CH designation required)
        if row['Species'] == 'BLANK':
            sample_type = SampleType.BLANK
        else:
            # Determine sample type based on AD/chick column for non-BLANK samples
            if row['AD/chick'] == 'AD':
                sample_type = SampleType.ADULT
            elif row['AD/chick'] == 'CH':
                sample_type = SampleType.CHICK
            else:
                raise ValueError(f"Unable to determine sample type for non-BLANK sample: {row}")

        sample = Sample(
            sample_id=row['Sample ID'],
            unique_id=row['Unique_ID'],
            colony_code=row['Colony_code'],
            sample_type=sample_type,
            source_row_index=int(idx)
        )
        samples.append(sample)
    return samples
