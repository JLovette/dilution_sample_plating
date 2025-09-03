from typing import List, Optional

from datamodel.sample import Sample, SampleType

class Plate:
    """Laboratory plate for organizing and managing biological samples.

    Attributes:
        rows (int): Number of rows in the plate
        cols (int): Number of columns in the plate
        plate (List[List[Optional[Sample]]]): 2D grid storing Sample objects
    """

    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.plate: List[List[Optional[Sample]]] = [[None for _ in range(cols)] for _ in range(rows)]

    @property
    def num_blank_positions(self) -> int:
        """Count the number of designated blank positions on the plate."""
        # This would need to be passed in from the TestManager since the Plate doesn't know about blank_positions
        # For now, return 0 and this property can be calculated elsewhere
        return 0

    @property
    def num_adult_samples(self) -> int:
        """Count the number of adult samples on the plate."""
        return sum(1 for row in self.plate for sample in row if sample and sample.sample_type == SampleType.ADULT)

    @property
    def num_chick_samples(self) -> int:
        """Count the number of chick samples on the plate."""
        return sum(1 for row in self.plate for sample in row if sample and sample.sample_type == SampleType.CHICK)

    @property
    def num_blank_samples(self) -> int:
        """Count the number of blank samples on the plate."""
        return sum(1 for row in self.plate for sample in row if sample and sample.sample_type == SampleType.BLANK)

    def set_sample(self, row: int, col: int, sample: Optional[Sample]) -> None:
        if 0 <= row < self.rows and 0 <= col < self.cols:
            self.plate[row][col] = sample
        else:
            raise ValueError("Row or column index out of bounds")

    def get_sample(self, row: int, col: int) -> Optional[Sample]:
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self.plate[row][col]
        else:
            raise ValueError("Row or column index out of bounds")

    def __str__(self) -> str:
        return self.to_string()

    def to_string(self, plate_title: Optional[str] = None) -> str:
        row_labels = [chr(ord('A') + i) for i in range(self.rows)]
        col_headers = [str(i+1) for i in range(self.cols)]
        output = []
        if plate_title:
            output.append(plate_title)
        output.append("\t" + "\t".join(col_headers))

        for i, row in enumerate(self.plate):
            row_str = row_labels[i] + "\t" + "\t".join(cell.unique_id if cell else "" for cell in row)
            output.append(row_str)
        return "\n".join(output) 