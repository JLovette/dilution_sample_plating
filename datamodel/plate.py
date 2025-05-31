from typing import List, Optional

from datamodel.sample import Sample

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
            row_str = row_labels[i] + "\t" + "\t".join(cell.sample_id if cell else "" for cell in row)
            output.append(row_str)
        return "\n".join(output) 