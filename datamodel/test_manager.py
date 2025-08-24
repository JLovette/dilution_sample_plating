import io
import base64
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from typing import List, Tuple

from datamodel.plate import Plate
from datamodel.sample import Sample, SampleType

class TestManager:
    def __init__(self, samples: List[Sample], rows: int, cols: int, blank_positions: List[Tuple[int, int]]):
        self.samples = sorted(samples, key=lambda s: s.sample_id)
        self.rows = rows
        self.cols = cols
        self.blank_positions = blank_positions
        num_plates = self._num_required_plates()
        self.plates: List[Plate] = [Plate(self.rows, self.cols) for _ in range(num_plates)]

    def _num_required_plates(self) -> int:
        """
        Calculate the number of plates required to fit all the samples.
        """
        # Ceiling division to determine how many plates are needed for the blank samples
        blank_samples = [s for s in self.samples if s.sample_type == SampleType.BLANK]
        blank_required = (len(blank_samples) + len(self.blank_positions) - 1) // len(self.blank_positions)

        # Ceiling division to determine how many plates are needed for the non-blank samples
        non_blank_slots = self.rows * self.cols - blank_required
        non_blank_samples = [s for s in self.samples if s.sample_type != SampleType.BLANK]
        non_blank_required = (len(non_blank_samples) + non_blank_slots - 1) // non_blank_slots
        return max(blank_required, non_blank_required)

    def _place_blank_samples(self):
        """
        Place BLANK samples in designated positions across the plates.
        """
        blank_samples = [s for s in self.samples if s.sample_type == SampleType.BLANK]
        total_blank_positions = len(self.plates) * len(self.blank_positions)
        blank_idx = 0

        for i in range(total_blank_positions):
            plate_idx = i % len(self.plates)
            pos_idx = i // len(self.plates)
            if pos_idx < len(self.blank_positions):
                r, c = self.blank_positions[pos_idx]
                if blank_idx < len(blank_samples):
                    self.plates[plate_idx].set_sample(r, c, blank_samples[blank_idx])
                    blank_idx += 1

    def _get_available_cell_positions(self):
        """
        Get list of available cell positions that are not designated as blank positions.
        """
        cell_positions = []
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) not in self.blank_positions:
                    cell_positions.append((r, c))
        return cell_positions

    def _place_non_blank_samples(self):
        """
        Place non-BLANK samples in available positions across the plates.
        First balances colonies across plates, then balances adult/chick ratios.
        """
        non_blank_samples = [s for s in self.samples if s.sample_type != SampleType.BLANK]
        cell_positions = self._get_available_cell_positions()

        if not non_blank_samples:
            return

        # Group samples by colony and sample type for balanced distribution
        colony_groups = {}
        for sample in non_blank_samples:
            if sample.colony_code not in colony_groups:
                colony_groups[sample.colony_code] = {'ADULT': [], 'CHICK': []}
            if sample.sample_type == SampleType.ADULT:
                colony_groups[sample.colony_code]['ADULT'].append(sample)
            elif sample.sample_type == SampleType.CHICK:
                colony_groups[sample.colony_code]['CHICK'].append(sample)

        # Create a balanced distribution order
        balanced_samples = []
        max_samples_per_colony = max(len(colony_groups[colony]['ADULT']) + len(colony_groups[colony]['CHICK']) 
                                   for colony in colony_groups)

        # Distribute samples by colony and type in a round-robin fashion
        for i in range(max_samples_per_colony):
            for colony in sorted(colony_groups.keys()):
                # Add ADULT sample if available
                if i < len(colony_groups[colony]['ADULT']):
                    balanced_samples.append(colony_groups[colony]['ADULT'][i])
                # Add CHICK sample if available
                if i < len(colony_groups[colony]['CHICK']):
                    balanced_samples.append(colony_groups[colony]['CHICK'][i])

        # Place samples on plates in the balanced order
        sample_idx = 0
        for plate_idx in range(len(self.plates)):
            for r, c in cell_positions:
                if self.plates[plate_idx].get_sample(r, c) is None:
                    if sample_idx < len(balanced_samples):
                        self.plates[plate_idx].set_sample(r, c, balanced_samples[sample_idx])
                        sample_idx += 1
                    else:
                        break
            if sample_idx >= len(balanced_samples):
                break

    def fill_plates(self):
        """Fill plates with samples, placing BLANKs first then other samples."""        
        self._place_blank_samples()
        self._place_non_blank_samples()

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

    def get_plate_statistics(self):
        """Return detailed statistics for each plate including colony counts and sample type counts."""
        plate_stats = []
        
        for i, plate in enumerate(self.plates):
            # Get colony counts for this plate
            colony_counts = {}
            for r in range(plate.rows):
                for c in range(plate.cols):
                    sample_obj = plate.get_sample(r, c)
                    if sample_obj and sample_obj.sample_type != SampleType.BLANK:
                        colony = sample_obj.colony_code
                        colony_counts[colony] = colony_counts.get(colony, 0) + 1
            
            # Get sample type counts for this plate
            type_counts = {"BLANK": 0, "ADULT": 0, "CHICK": 0}
            for r in range(plate.rows):
                for c in range(plate.cols):
                    sample_obj = plate.get_sample(r, c)
                    if sample_obj:
                        if sample_obj.sample_type == SampleType.BLANK:
                            type_counts["BLANK"] += 1
                        elif sample_obj.sample_type == SampleType.ADULT:
                            type_counts["ADULT"] += 1
                        elif sample_obj.sample_type == SampleType.CHICK:
                            type_counts["CHICK"] += 1
            
            # Calculate total samples on this plate
            total_samples = type_counts["ADULT"] + type_counts["CHICK"]
            
            plate_stats.append({
                "plate_number": i + 1,
                "colony_counts": colony_counts,
                "type_counts": type_counts,
                "total_samples": total_samples,
                "utilization": total_samples / (plate.rows * plate.cols) if plate.rows * plate.cols > 0 else 0
            })
        
        return plate_stats

    def algorithm_metrics(self):
        """Return a dict of summary metrics for algorithm performance."""
        # Get detailed plate statistics
        plate_stats = self.get_plate_statistics()
        
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
        utilizations = [stats["utilization"] for stats in plate_stats]
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
            "overall_adherence_score": overall,
            "plate_statistics": plate_stats
        }

    def generate_plate_visualization_pdf(self) -> str:
        """Generate a PDF visualization of all plates with each plate on a separate page and return it as a base64 string."""
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

        # For PDF, we'll use a standard page size and fit each plate appropriately
        fig_width = self.cols * cell_size
        fig_height = self.rows * cell_size

        # Cap figure size to prevent excessively large pages
        max_fig_width = 12  # inches
        max_fig_height = 10  # inches
        fig_width = min(max_fig_width, fig_width)
        fig_height = min(max_fig_height, fig_height)

        # Create PDF with PdfPages
        buf = io.BytesIO()
        with PdfPages(buf) as pdf:
            for i, plate in enumerate(self.plates):
                # Create a new figure for each plate
                fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height), facecolor='white')
                ax.set_facecolor('white')

                # Create a grid of cells
                grid = np.zeros((self.rows, self.cols, 3))
                
                # Fill in the colors based on sample type (BLANK, Empty, or Sample)
                for r in range(self.rows):
                    for c in range(self.cols):
                        sample_obj = plate.get_sample(r, c) # Get the Sample object

                        if sample_obj is None:
                            grid[r, c] = [1, 1, 1]  # White for empty cells
                        elif sample_obj.sample_type == SampleType.BLANK:
                            grid[r, c] = [0.95, 0.95, 0.95]  # Very light gray for blanks
                        else: # It's a non-BLANK Sample
                            # Use a light blue color for samples
                            grid[r, c] = [0.7, 0.85, 0.95]  # Light blue for samples

                # Display the grid
                ax.imshow(grid)
                ax.set_title(f'Plate {i+1}', fontsize=12, fontweight='bold', color='black', pad=10)
                
                # Add grid lines
                ax.set_xticks(np.arange(-.5, self.cols, 1), minor=True)
                ax.set_yticks(np.arange(-.5, self.rows, 1), minor=True)
                ax.grid(which='minor', color='#e0e0e0', linestyle='-', linewidth=0.5)
                
                # Add row and column labels with better visibility
                ax.set_xticks(np.arange(self.cols))
                ax.set_yticks(np.arange(self.rows))
                ax.set_xticklabels([str(i+1) for i in range(self.cols)])
                ax.set_yticklabels([chr(ord('A') + i) for i in range(self.rows)])
                
                # Style the tick labels for better visibility
                ax.tick_params(axis='both', which='major', labelsize=10, colors='black')
                ax.tick_params(axis='both', which='minor', labelsize=8, colors='black')

                # Add cell IDs (sample_id)
                for r in range(self.rows):
                    for c in range(self.cols):
                        sample_obj = plate.get_sample(r, c)
                        if sample_obj:
                            display_text = sample_obj.unique_id if sample_obj.unique_id else ""
                            
                            # Always use black text for maximum contrast against white/light backgrounds
                            text_color = 'black'
                            
                            # Add the cell ID with maximum contrast
                            ax.text(c, r, display_text, 
                                   ha='center', va='center',
                                   color=text_color,
                                   fontsize=8,
                                   weight='bold',
                                   bbox=dict(facecolor='white', 
                                           edgecolor='#333333',
                                           alpha=1.0,
                                           boxstyle='round,pad=0.3',
                                           linewidth=1))

                plt.tight_layout()
                
                # Add the page to the PDF
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

        # Convert PDF to base64 string
        buf.seek(0)
        pdf_str = base64.b64encode(buf.read()).decode()
        
        return pdf_str

    def get_plate_visualization_pdf_html(self) -> str:
        """Return HTML for displaying the PDF visualization with download link."""
        pdf_str = self.generate_plate_visualization_pdf()
        if pdf_str:
            return f'''
            <div style="text-align: center; margin: 20px 0;">
                <a href="data:application/pdf;base64,{pdf_str}" 
                   download="plate_layouts.pdf" 
                   style="background-color: #007bff; color: white; padding: 12px 24px; 
                          text-decoration: none; border-radius: 5px; font-weight: bold;">
                    Download PDF
                </a>
            </div>
            '''
        return "No plates to visualize"

    def get_pdf_data(self) -> bytes:
        """Return the PDF data as bytes for direct download."""
        pdf_str = self.generate_plate_visualization_pdf()
        if pdf_str:
            return base64.b64decode(pdf_str)
        return None

    def get_detailed_csv_data(self) -> bytes:
        """Return the detailed CSV data as bytes for direct download."""
        csv_string = self.to_detailed_csv_string()
        return csv_string.encode('utf-8')

    def get_grid_csv_data(self) -> bytes:
        """Return the grid CSV data as bytes for direct download."""
        csv_string = self.to_csv_string()
        return csv_string.encode('utf-8')

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

    def to_detailed_csv_string(self) -> str:
        """Generate a detailed CSV string with sample information for lab work."""
        csv_data = []
        
        # Header row with sample details
        header = ["Plate", "Row", "Column", "Sample_ID", "Unique_ID", "Species", "Colony_Code", "Sample_Type", "Substrate", "Notes"]
        csv_data.append(header)
        
        for i, plate in enumerate(self.plates):
            for r in range(plate.rows):
                for c in range(plate.cols):
                    sample_obj = plate.get_sample(r, c)
                    if sample_obj:
                        row_data = [
                            f"Plate {i+1}",
                            chr(ord('A') + r),
                            str(c + 1),
                            sample_obj.sample_id if hasattr(sample_obj, 'sample_id') else "",
                            sample_obj.unique_id if hasattr(sample_obj, 'unique_id') else "",
                            sample_obj.species if hasattr(sample_obj, 'species') else "",
                            sample_obj.colony_code if hasattr(sample_obj, 'colony_code') else "",
                            sample_obj.sample_type.value if hasattr(sample_obj, 'sample_type') else "",
                            sample_obj.substrate if hasattr(sample_obj, 'substrate') else "",
                            sample_obj.notes if hasattr(sample_obj, 'notes') else ""
                        ]
                        csv_data.append(row_data)
                    else:
                        # Empty cell
                        row_data = [
                            f"Plate {i+1}",
                            chr(ord('A') + r),
                            str(c + 1),
                            "", "", "", "", "", "", ""
                        ]
                        csv_data.append(row_data)
        
        # Join rows into a single CSV string
        csv_string = "\n".join([",".join(map(str, row)) for row in csv_data])
        return csv_string