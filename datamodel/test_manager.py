import io
import base64
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

from datamodel.plate import Plate
from datamodel.sample import Sample, SampleType

class TestManager:
    def __init__(self, samples: List[Sample], rows: int, cols: int, blank_positions: List[Tuple[int, int]]):
        self.samples = samples
        self.rows = rows
        self.cols = cols
        self.blank_positions = blank_positions
        self.plates: List[Plate] = []
        self.fill_plates()

    def fill_plates(self):
        # Sort samples by Sample ID (or any other desired order)
        samples_sorted = sorted(self.samples, key=lambda s: s.sample_id)
        # Separate BLANK samples from others
        blank_samples = [s for s in samples_sorted if s.sample_type == SampleType.BLANK]
        non_blank_samples = [s for s in samples_sorted if s.sample_type != SampleType.BLANK]
        blank_idx = 0
        sample_idx = 0
        num_cells = self.rows * self.cols
        num_blanks = len(self.blank_positions)
        samples_per_plate = num_cells - num_blanks
        # Calculate number of plates needed
        n_non_blank = len(non_blank_samples)
        n_blanks = len(blank_samples)
        n_plates = 0
        while n_non_blank > n_plates * samples_per_plate:
            n_plates += 1
        if n_plates == 0:
            n_plates = 1
        self.plates = [Plate(self.rows, self.cols) for _ in range(n_plates)]
        # Fill in BLANKs with actual BLANK samples if available
        total_blank_positions = n_plates * len(self.blank_positions)
        for i in range(total_blank_positions):
            plate_idx = i % n_plates
            pos_idx = i // n_plates
            if pos_idx < len(self.blank_positions):
                r, c = self.blank_positions[pos_idx]
                if blank_idx < len(blank_samples):
                    # Place the actual BLANK Sample object
                    self.plates[plate_idx].set_sample(r, c, blank_samples[blank_idx])
                    blank_idx += 1
                # If no more BLANK samples, the position remains None (empty)
        # Fill in non-blank samples breadth-first (round-robin)
        cell_positions = []
        for r in range(self.rows):
            for c in range(self.cols):
                # Only consider positions that are not designated as blank positions
                if (r, c) not in self.blank_positions:
                     # Also check if the cell is already filled by a BLANK sample object
                    if plate_idx < len(self.plates) and self.plates[plate_idx].get_sample(r, c) is None:
                         cell_positions.append((r, c))

        sample_idx = 0
        # We need to iterate through plates and then cell_positions on each plate
        for plate_idx in range(n_plates):
             for r, c in cell_positions:
                # Check if the cell is still empty before placing a sample
                if self.plates[plate_idx].get_sample(r, c) is None:
                    if sample_idx < len(non_blank_samples):
                        # Place the actual non-BLANK Sample object
                        self.plates[plate_idx].set_sample(r, c, non_blank_samples[sample_idx])
                        sample_idx += 1
                    else:
                        # No more samples to place
                        break
             if sample_idx >= len(non_blank_samples):
                 break # Stop filling plates if all samples are placed

    def get_plates(self) -> List[Plate]:
        return self.plates

    def print_plates(self):
        for i, plate in enumerate(self.plates):
            print(f"Plate {i+1}:")
            print(plate)
            print()

    def colony_balance(self):
        """Return a list of dicts, one per plate, with counts of each colony_code."""
        result = []
        for plate in self.plates:
            colony_counts = {}
            for r in range(plate.rows):
                for c in range(plate.cols):
                    sample_obj = plate.get_sample(r, c)
                    # Check if the cell contains a non-BLANK sample object
                    if sample_obj and sample_obj.sample_type != SampleType.BLANK:
                        colony = sample_obj.colony_code
                        colony_counts[colony] = colony_counts.get(colony, 0) + 1
            result.append(colony_counts)
        return result

    def blank_adult_chick_counts(self):
        """Return a list of dicts, one per plate, with counts of blanks, adults, and chicks."""
        result = []
        for plate in self.plates:
            counts = {"BLANK": 0, "ADULT": 0, "CHICK": 0}
            for r in range(plate.rows):
                for c in range(plate.cols):
                    sample_obj = plate.get_sample(r, c)
                    if sample_obj:
                        if sample_obj.sample_type == SampleType.BLANK:
                            counts["BLANK"] += 1
                        elif sample_obj.sample_type == SampleType.ADULT:
                            counts["ADULT"] += 1
                        elif sample_obj.sample_type == SampleType.CHICK:
                            counts["CHICK"] += 1
            result.append(counts)
        return result

    def contiguous_sample_measure(self):
        """Return a list of floats, one per plate, representing the average contiguous run length of non-blank samples."""
        result = []
        for plate in self.plates:
            runs = []
            # Check rows
            for r in range(plate.rows):
                run = 0
                for c in range(plate.cols):
                    sample_obj = plate.get_sample(r, c)
                    # Check if the cell contains a non-BLANK sample object
                    if sample_obj and sample_obj.sample_type != SampleType.BLANK:
                        run += 1
                    else:
                        if run > 0:
                            runs.append(run)
                        run = 0
                if run > 0:
                    runs.append(run)
            # Check columns
            for c in range(plate.cols):
                run = 0
                for r in range(plate.rows):
                    sample_obj = plate.get_sample(r, c)
                     # Check if the cell contains a non-BLANK sample object
                    if sample_obj and sample_obj.sample_type != SampleType.BLANK:
                        run += 1
                    else:
                        if run > 0:
                            runs.append(run)
                        run = 0
                if run > 0:
                    runs.append(run)
            avg_run = sum(runs) / len(runs) if runs else 0.0
            result.append(avg_run)
        return result

    def algorithm_metrics(self):
        """Return a dict of summary metrics for algorithm performance."""
        # 1. Colony balance score
        colony_counts_per_plate = self.colony_balance()
        all_colonies = set()
        for d in colony_counts_per_plate:
            all_colonies.update(d.keys())
        colony_stdevs = []
        for colony in all_colonies:
            counts = [d.get(colony, 0) for d in colony_counts_per_plate]
            if len(counts) > 1:
                colony_stdevs.append(np.std(counts))
        colony_balance_score = float(np.mean(colony_stdevs)) if colony_stdevs else 0.0

        # 2. Blank/adult/chick balance score
        type_counts_per_plate = self.blank_adult_chick_counts()
        for d in type_counts_per_plate:
            d.setdefault("BLANK", 0)
            d.setdefault("ADULT", 0)
            d.setdefault("CHICK", 0)
        type_stdevs = []
        for t in ["BLANK", "ADULT", "CHICK"]:
            counts = [d[t] for d in type_counts_per_plate]
            if len(counts) > 1:
                type_stdevs.append(np.std(counts))
        blank_adult_chick_balance_score = float(np.mean(type_stdevs)) if type_stdevs else 0.0

        # 3. Contiguity score
        contiguity_scores = self.contiguous_sample_measure()
        contiguity_score = float(np.mean(contiguity_scores)) if contiguity_scores else 0.0

        # 4. Plate utilization
        utilizations = []
        for plate in self.plates:
            total = plate.rows * plate.cols
            non_blank = sum(1 for r in range(plate.rows) for c in range(plate.cols) if plate.get_sample(r, c) and plate.get_sample(r, c) != "BLANK")
            utilizations.append(non_blank / total if total else 0)
        plate_utilization = float(np.mean(utilizations)) if utilizations else 0.0

        # 5. Overall adherence score (lower is better)
        # Normalize: balance scores by max possible (use total samples/plates as rough scale), contiguity by max possible (plate size), utilization by 1-utilization
        n_plates = len(self.plates)
        n_samples = len(self.samples)
        max_balance = n_samples / n_plates if n_plates else 1
        max_contiguity = max(self.rows, self.cols)
        norm_colony = colony_balance_score / max_balance if max_balance else 0
        norm_type = blank_adult_chick_balance_score / max_balance if max_balance else 0
        norm_contig = contiguity_score / max_contiguity if max_contiguity else 0
        norm_util = 1 - plate_utilization
        overall = float(np.mean([norm_colony, norm_type, norm_contig, norm_util]))

        return {
            "colony_balance_score": colony_balance_score,
            "blank_adult_chick_balance_score": blank_adult_chick_balance_score,
            "contiguity_score": contiguity_score,
            "plate_utilization": plate_utilization,
            "overall_adherence_score": overall
        } 

    def generate_plate_visualization(self) -> str:
        """Generate a PNG visualization of all plates and return it as a base64 string."""
        n_plates = len(self.plates)
        if n_plates == 0:
            return None

        # Calculate the maximum length of a sample ID for better sizing
        max_sid_length = 0
        if self.samples:
            # Assuming Unique_ID is used, if not, adjust field
            max_sid_length = max((len(s.unique_id) for s in self.samples), default=0)
        # Also consider the 'BLANK' string length
        max_sid_length = max(max_sid_length, len("BLANK"))

        # Estimate required cell size based on max text length and font size
        # Using a heuristic: char_width approx 0.6 * font_height
        fontsize = 7 # Current font size
        points_per_inch = 72
        char_width_inch = 0.6 * (fontsize / points_per_inch)
        # Add some padding (e.g., 2 characters worth)
        required_cell_width_text = (max_sid_length + 2) * char_width_inch

        # Base cell size (for visual spacing even with short text)
        cell_size_base = 0.5 # inches

        # Final cell size is the maximum of base size and text-required size
        cell_size = max(cell_size_base, required_cell_width_text)

        fig_width = self.cols * cell_size * n_plates
        fig_height = self.rows * cell_size # Assuming roughly square cells are okay

        # Cap figure size to prevent excessively large images
        max_fig_width = 50 # Increased cap
        max_fig_height = 30 # Increased cap
        fig_width = min(max_fig_width, fig_width)
        fig_height = min(max_fig_height, fig_height)

        fig, axes = plt.subplots(1, n_plates, figsize=(fig_width, fig_height))
        if n_plates == 1:
            axes = [axes]

        # Remove colony color logic
        # Create a sample map for coloring (no longer colony specific)
        # sample_map = {s.sample_id: s for s in self.samples}
        # colony_colors = {}
        # for sample in self.samples:
        #     if sample.colony_code not in colony_colors:
        #         colony_colors[sample.colony_code] = np.random.rand(3,)

        for i, (plate, ax) in enumerate(zip(self.plates, axes)):
            # Create a grid of cells
            grid = np.zeros((self.rows, self.cols, 3))
            
            # Fill in the colors based on sample type (BLANK, Empty, or Sample)
            for r in range(self.rows):
                for c in range(self.cols):
                    sample_obj = plate.get_sample(r, c) # Get the Sample object

                    if sample_obj is None:
                        grid[r, c] = [1, 1, 1]  # White for empty cells
                    elif sample_obj.sample_type == SampleType.BLANK:
                        grid[r, c] = [0.9, 0.9, 0.9]  # Light gray for blanks
                    else: # It's a non-BLANK Sample
                        # Use a fixed color for all non-blank samples
                        grid[r, c] = [0.6, 0.8, 0.6]  # Example: a shade of green

            # Display the grid
            ax.imshow(grid)
            ax.set_title(f'Plate {i+1}')
            
            # Add grid lines
            ax.set_xticks(np.arange(-.5, self.cols, 1), minor=True)
            ax.set_yticks(np.arange(-.5, self.rows, 1), minor=True)
            ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
            
            # Add row and column labels
            ax.set_xticks(np.arange(self.cols))
            ax.set_yticks(np.arange(self.rows))
            ax.set_xticklabels([str(i+1) for i in range(self.cols)])
            ax.set_yticklabels([chr(ord('A') + i) for i in range(self.rows)])

            # Add cell IDs (sample_id)
            for r in range(self.rows):
                for c in range(self.cols):
                    sample_obj = plate.get_sample(r, c)
                    if sample_obj:
                        display_text = sample_obj.unique_id if sample_obj.unique_id else ""
                        
                        # Calculate text color based on background
                        bg_color = grid[r, c]
                        # Simple luminance check for text color
                        luminance = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]
                        text_color = 'black' if luminance > 0.5 else 'white'
                        
                        # Add the cell ID
                        ax.text(c, r, display_text, 
                               ha='center', va='center',
                               color=text_color,
                               fontsize=7,
                               bbox=dict(facecolor='none', 
                                       edgecolor='none',
                                       alpha=0.7))

        plt.tight_layout()
        
        # Convert plot to base64 string
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        plt.close()
        
        return img_str

    def get_plate_visualization_html(self) -> str:
        """Return HTML for displaying the plate visualization."""
        img_str = self.generate_plate_visualization()
        if img_str:
            return f'<img src="data:image/png;base64,{img_str}" style="max-width:100%;">'
        return "No plates to visualize"

    def to_csv_string(self) -> str:
        """Generate a CSV string representation of the filled plates in a 2D grid format."""
        csv_data = []
        for i, plate in enumerate(self.plates):
            csv_data.append([f"Plate {i+1}"])

            # Add column headers row (empty first cell for row labels)
            header_row = [""] + [str(c + 1) for c in range(plate.cols)]
            csv_data.append(header_row)

            for r in range(plate.rows):
                row_label = chr(ord('A') + r)
                row_data = [row_label] # Start with row label
                for c in range(plate.cols):
                    sample_obj = plate.get_sample(r, c)
                    cell_value = sample_obj.unique_id if sample_obj else ""
                    row_data.append(cell_value)
                csv_data.append(row_data)

            # Add two blank rows after each plate
            csv_data.append([])
            csv_data.append([])

        # Join rows into a single CSV string
        # No need to handle None explicitly in join anymore as cell_value is always string or empty
        csv_string = "\n".join([",".join(map(str, row)) for row in csv_data])
        return csv_string 