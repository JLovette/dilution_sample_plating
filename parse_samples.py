import pandas as pd
from typing import List

from datamodel.sample import Sample, SampleType


def parse_samples(csv_file: str) -> List[Sample]:
    """Parse the penguin sample data from a CSV file."""
    df = pd.read_csv(csv_file)
    samples = []
    for _, row in df.iterrows():
        # Determine sample type
        if row['Species'] == 'BLANK':
            sample_type = SampleType.BLANK
        elif row['AD/chick'] == 'AD':
            sample_type = SampleType.ADULT
        elif row['AD/chick'] == 'CH':
            sample_type = SampleType.CHICK
        else:
            raise ValueError("Unable to determine sample type...")

        sample = Sample(
            sample_id=row['Sample ID'],
            unique_id=row['Unique_ID'],
            colony_code=row['Colony_code'],
            sample_type=sample_type
        )
        samples.append(sample)
    return samples
