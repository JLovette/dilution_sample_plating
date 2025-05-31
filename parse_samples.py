from datetime import datetime, timedelta
from datamodel.sample import Sample
from typing import List
import pandas as pd

def parse_samples(csv_file: str) -> List[Sample]:
    """Parse the penguin sample data from a CSV file."""
    # Read the CSV file
    df = pd.read_csv(csv_file)
    # Convert DataFrame rows to Sample objects
    samples = []
    for _, row in df.iterrows():
        try:
            # Convert Excel date number to datetime
            try:
                date = datetime(1899, 12, 30) + timedelta(days=int(row['Date']))
            except (ValueError, TypeError):
                date = None
            # Convert chick age to float if possible
            try:
                chick_age = float(row['Chick age']) if row['Chick age'] else None
            except (ValueError, TypeError):
                chick_age = None
            sample = Sample(
                sample_id=row['Sample ID'],
                unique_id=row['Unique_ID'],
                species=row['Species'],
                colony_code=row['Colony_code'],
                collection_date=date,
                substrate=row['Substrate'],
                substrate_group=row['Substrate_group'],
                island=row['Island'],
                area=row['Area'] if row['Area'] else None,
                breeding_period=row['Breeding period (Incubation/ Chick-rearing/ Mixed)'],
                chick_age=chick_age,
                notes=row['Notes'] if row['Notes'] else None,
                collaborator=row['Collaborator name'],
                ad_chick=row['AD/chick'] if 'AD/chick' in row and pd.notnull(row['AD/chick']) else None
            )
            samples.append(sample)
        except Exception as e:
            print(f"Error parsing row: {row['Sample ID']} - {str(e)}")
    return samples
