import io
import base64
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from typing import List, Tuple, Union
import pandas as pd

from datamodel.plate import Plate
from datamodel.sample import Sample, SampleType

class TestManager:
    def __init__(self, file_path: Union[str, io.StringIO, pd.DataFrame], rows: int, cols: int, blank_positions: List[Tuple[int, int]], 
                 colony_weight: float = 1.0, type_weight: float = 2.0, starting_plate_number: int = 1):
        """
        Initialize TestManager with an uploaded file instead of a list of samples.
        
        Args:
            file_path: Path to CSV file, StringIO object, or pandas DataFrame
            rows: Number of rows in each plate
            cols: Number of columns in each plate
            blank_positions: List of (row, col) tuples for BLANK positions
            colony_weight: Weight for colony balance (default: 1.0)
            type_weight: Weight for adult/chick balance (default: 2.0)
            starting_plate_number: Starting plate number (default: 1)
        """
        self.file_path = file_path
        self.rows = rows
        self.cols = cols
        self.blank_positions = blank_positions
        self.colony_weight = colony_weight
        self.type_weight = type_weight
        self.starting_plate_number = starting_plate_number
        
        # Load samples from the file
        self.samples = self._load_samples_from_file()
        self.original_df = self._load_original_dataframe()
        
        # Initialize plates
        num_plates = self._num_required_plates()
        self.plates: List[Plate] = [Plate(self.rows, self.cols) for _ in range(num_plates)]

    def _load_samples_from_file(self) -> List[Sample]:
        """Load samples from the uploaded file."""
        from parse_samples import parse_samples
        
        if isinstance(self.file_path, pd.DataFrame):
            # If it's already a DataFrame, convert to CSV string first
            temp_csv = io.StringIO()
            self.file_path.to_csv(temp_csv, index=False)
            temp_csv.seek(0)
            return parse_samples(temp_csv)
        elif hasattr(self.file_path, 'read'):  # StreamlitUploadedFile or file-like object
            # Reset file pointer to beginning
            self.file_path.seek(0)
            return parse_samples(self.file_path)
        else:
            # If it's a file path or StringIO, parse directly
            return parse_samples(self.file_path)

    def _load_original_dataframe(self) -> pd.DataFrame:
        """Load the original DataFrame from the uploaded file."""
        if isinstance(self.file_path, pd.DataFrame):
            return self.file_path.copy()
        elif hasattr(self.file_path, 'read'):  # StreamlitUploadedFile or file-like object
            # Reset file pointer to beginning
            self.file_path.seek(0)
            return pd.read_csv(self.file_path)
        else:
            return pd.read_csv(self.file_path)

    def _get_plate_assignment_for_sample(self, sample: Sample) -> Tuple[int, str, int]:
        """Get the plate number, row, and column for a given sample."""
        for plate_idx, plate in enumerate(self.plates):
            # Use column-first ordering for consistency with the new filling pattern
            for c in range(plate.cols):
                for r in range(plate.rows):
                    plate_sample = plate.get_sample(r, c)
                    if plate_sample and plate_sample.unique_id == sample.unique_id:
                        plate_num = self.starting_plate_number + plate_idx
                        row_label = chr(ord('A') + r)  # Convert to A, B, C, etc.
                        col_label = c + 1  # Convert to 1, 2, 3, etc.
                        return plate_num, row_label, col_label
        return None, None, None

    def export_original_csv_with_plates(self) -> str:
        """
        Export the original CSV with new columns showing plate assignment and cell position.
        
        Returns:
            CSV string with the original data plus plate assignments and cell positions
        """
        # Create a copy of the original DataFrame
        result_df = self.original_df.copy()

        # Add new columns for plate information
        result_df['Plate Number'] = ''
        result_df['Cell Row'] = ''
        result_df['Cell Column'] = ''

        # Fill in plate numbers and cell positions for each sample
        for idx, row in result_df.iterrows():
            # Find the corresponding sample object
            sample_found = False
            for sample in self.samples:
                if (sample.sample_id == str(row['Sample ID']) and 
                    sample.unique_id == str(row['Unique_ID'])):
                    plate_num, row_label, col_label = self._get_plate_assignment_for_sample(sample)
                    if plate_num:
                        result_df.at[idx, 'Plate Number'] = plate_num
                        result_df.at[idx, 'Cell Row'] = row_label
                        result_df.at[idx, 'Cell Column'] = col_label
                    sample_found = True
                    break
            
            # If no sample found, leave columns empty
            if not sample_found:
                result_df.at[idx, 'Plate Number'] = ''
                result_df.at[idx, 'Cell Row'] = ''
                result_df.at[idx, 'Cell Column'] = ''

        # Convert to CSV string
        output = io.StringIO()
        result_df.to_csv(output, index=False)
        output.seek(0)
        return output.getvalue()

    def get_original_csv_with_plates_data(self) -> bytes:
        """Return the original CSV with plate assignments as bytes for download."""
        csv_string = self.export_original_csv_with_plates()
        return csv_string.encode('utf-8')

    def _num_required_plates(self) -> int:
        """
        Calculate the number of plates required to fit all the samples.
        """
        # Calculate total samples (all samples are now treated equally)
        total_samples = len(self.samples)
        
        # Calculate available slots per plate (excluding designated blank positions)
        available_slots_per_plate = self.rows * self.cols - len(self.blank_positions)
        
        # Calculate number of plates needed
        if available_slots_per_plate > 0:
            num_plates = (total_samples + available_slots_per_plate - 1) // available_slots_per_plate
        else:
            num_plates = 1  # At least one plate
            
        return num_plates

    def _get_available_cell_positions(self):
        """
        Get list of available cell positions that are not designated as blank positions.
        Fills by column rather than by row - goes down each column before moving to the next column.
        """
        cell_positions = []
        for c in range(self.cols):
            for r in range(self.rows):
                if (r, c) not in self.blank_positions:
                    cell_positions.append((r, c))
        return cell_positions

    def _place_samples(self):
        """
        Place all samples in available positions across the plates.
        Uses a balanced distribution approach to ensure samples are spread evenly across plates.
        """
        # All samples are now treated equally (including former BLANK samples)
        samples_to_place = self.samples
        cell_positions = self._get_available_cell_positions()

        if not samples_to_place:
            return

        # Group samples by colony and sample type
        colony_groups = {}
        for sample in samples_to_place:
            if sample.colony_code not in colony_groups:
                colony_groups[sample.colony_code] = {'ADULT': [], 'CHICK': [], 'BLANK': []}
            if sample.sample_type == SampleType.ADULT:
                colony_groups[sample.colony_code]['ADULT'].append(sample)
            elif sample.sample_type == SampleType.CHICK:
                colony_groups[sample.colony_code]['CHICK'].append(sample)
            elif sample.sample_type == SampleType.BLANK:
                colony_groups[sample.colony_code]['BLANK'].append(sample)

        # Initialize plate tracking for balancing
        plate_colony_counts = [{} for _ in range(len(self.plates))]
        plate_type_counts = [{"ADULT": 0, "CHICK": 0, "BLANK": 0} for _ in range(len(self.plates))]

        # Calculate target distributions per plate
        total_samples = len(samples_to_place)
        available_slots_per_plate = self.rows * self.cols - len(self.blank_positions)
        
        # Ensure only the last plate can be non-full
        # All plates except the last should be completely filled
        if len(self.plates) > 1:
            # First N-1 plates get filled completely
            samples_per_full_plate = available_slots_per_plate
            # Last plate gets remaining samples
            samples_on_last_plate = total_samples - (samples_per_full_plate * (len(self.plates) - 1))
            
            plate_sample_counts = [samples_per_full_plate] * (len(self.plates) - 1)
            plate_sample_counts.append(max(0, samples_on_last_plate))
        else:
            # Only one plate - it gets all samples
            plate_sample_counts = [total_samples]

        # Create a balanced sample list that distributes samples more evenly
        balanced_samples = []
        
        # Get all colonies and types
        all_colonies = sorted(colony_groups.keys())
        all_types = ['ADULT', 'CHICK', 'BLANK']
        
        # Create a more sophisticated distribution that ensures better balance
        # First, calculate how many samples of each type we have total
        total_by_type = {}
        for sample_type in all_types:
            total_by_type[sample_type] = sum(len(colony_groups[colony][sample_type]) for colony in all_colonies)
        
        # Create a distribution that alternates between types and colonies
        # This ensures we don't put all samples of one type on the same plate
        max_samples = max(total_by_type.values())
        
        for i in range(max_samples):
            for sample_type in all_types:
                for colony in all_colonies:
                    if i < len(colony_groups[colony][sample_type]):
                        balanced_samples.append(colony_groups[colony][sample_type][i])

        # Now place samples using a smart distribution algorithm
        current_plate = 0
        samples_placed_on_current_plate = 0
        
        for sample in balanced_samples:
            # Find the best plate for this sample based on balancing criteria
            best_plate = self._find_best_balanced_plate(
                sample, plate_colony_counts, plate_type_counts, 
                plate_sample_counts, samples_placed_on_current_plate
            )
            
            if best_plate is not None:
                # Find available position on the best plate
                position_found = False
                for r, c in cell_positions:
                    if self.plates[best_plate].get_sample(r, c) is None:
                        self.plates[best_plate].set_sample(r, c, sample)
                        
                        # Update tracking
                        colony = sample.colony_code
                        plate_colony_counts[best_plate][colony] = plate_colony_counts[best_plate].get(colony, 0) + 1
                        if sample.sample_type == SampleType.ADULT:
                            plate_type_counts[best_plate]["ADULT"] += 1
                        elif sample.sample_type == SampleType.CHICK:
                            plate_type_counts[best_plate]["CHICK"] += 1
                        elif sample.sample_type == SampleType.BLANK:
                            plate_type_counts[best_plate]["BLANK"] += 1
                        
                        position_found = True
                        break
                
                if not position_found:
                    # This shouldn't happen, but just in case
                    print(f"Warning: Could not find position for sample {sample.unique_id} on plate {best_plate}")
            else:
                print(f"Warning: Could not find suitable plate for sample {sample.unique_id}")
        
        # Validate that only the last plate is non-full
        self._validate_plate_filling()

    def _find_best_balanced_plate(self, sample, plate_colony_counts, plate_type_counts, 
                                 plate_sample_counts, samples_placed_on_current_plate):
        """
        Find the best plate for a sample based on balancing criteria.
        Prioritizes colony balance and type balance while ensuring plates are filled evenly.
        """
        best_plate = None
        best_score = float('inf')
        
        for plate_idx in range(len(self.plates)):
            # Check if this plate has available positions
            has_available = False
            for c in range(self.cols):
                for r in range(self.rows):
                    if (r, c) not in self.blank_positions and self.plates[plate_idx].get_sample(r, c) is None:
                        has_available = True
                        break
                if has_available:
                    break
            
            if not has_available:
                continue
            
            # Calculate various balance scores (lower is better)
            
            # 1. Colony balance score - prefer plates with fewer samples from this colony
            current_colony_count = plate_colony_counts[plate_idx].get(sample.colony_code, 0)
            colony_score = current_colony_count * 3.0  # Weight colony balance very heavily
            
            # 2. Type balance score - prefer plates with fewer samples of this type
            current_type_count = plate_type_counts[plate_idx].get(sample.sample_type.value, 0)
            type_score = current_type_count * 2.5  # Weight type balance heavily
            
            # 3. Plate utilization score - prefer plates that are under their target
            current_plate_samples = sum(plate_colony_counts[plate_idx].values())
            target_samples = plate_sample_counts[plate_idx]
            utilization_score = max(0, current_plate_samples - target_samples) * 4.0  # Weight very heavily
            
            # 4. Sequential filling bonus - prefer earlier plates to avoid gaps
            # This ensures earlier plates are filled completely before moving to later plates
            sequential_bonus = plate_idx * 2.0  # Strong preference for earlier plates
            
            # Combined score
            combined_score = colony_score + type_score + utilization_score + sequential_bonus
            
            if combined_score < best_score:
                best_score = combined_score
                best_plate = plate_idx
        
        return best_plate

    def _validate_plate_filling(self):
        """
        Validate that only the last plate is non-full.
        This ensures our algorithm constraint is met.
        """
        if len(self.plates) <= 1:
            return
        
        available_slots_per_plate = self.rows * self.cols - len(self.blank_positions)
        
        for i in range(len(self.plates) - 1):  # All plates except the last
            sample_count = 0
            for r in range(self.rows):
                for c in range(self.cols):
                    if self.plates[i].get_sample(r, c) is not None:
                        sample_count += 1
            
            if sample_count < available_slots_per_plate:
                print(f"Warning: Plate {i+1} is not full ({sample_count}/{available_slots_per_plate} samples)")
                print("This violates the constraint that only the last plate can be non-full.")
        
        # Check the last plate
        last_plate_idx = len(self.plates) - 1
        sample_count = 0
        for r in range(self.rows):
            for c in range(self.cols):
                if self.plates[last_plate_idx].get_sample(r, c) is not None:
                    sample_count += 1
        
        print(f"Plate {last_plate_idx+1} (last): {sample_count}/{available_slots_per_plate} samples")

    def _sort_samples_by_id_within_plates(self):
        """
        Sort all samples by ID within each plate while preserving designated blank positions.
        This ensures samples are in numerical order for easier plate setup.
        """
        for plate in self.plates:
            # Collect all samples from this plate with their positions
            samples_with_positions = []
            for r in range(plate.rows):
                for c in range(plate.cols):
                    sample_obj = plate.get_sample(r, c)
                    if sample_obj:
                        samples_with_positions.append((r, c, sample_obj))
            
            if len(samples_with_positions) <= 1:
                continue  # No need to sort if 1 or fewer samples
            
            # Sort samples by original CSV row order to ensure stable down-column sequencing
            # Fall back to sample_id if source_row_index is not available
            def _order_key(item):
                sample_obj = item[2]
                if hasattr(sample_obj, 'source_row_index') and sample_obj.source_row_index is not None:
                    return (0, int(sample_obj.source_row_index))
                # Fallback: attempt numeric sample_id else lexicographic
                sid = getattr(sample_obj, 'sample_id', '')
                try:
                    return (1, int(sid))
                except (ValueError, TypeError):
                    return (2, str(sid))
            samples_with_positions.sort(key=_order_key)
            
            # Get available positions (excluding designated blank positions) in the same order as _get_available_cell_positions
            # Column-first ordering: go down each column before moving to the next column
            available_positions = []
            for c in range(plate.cols):
                for r in range(plate.rows):
                    if (r, c) not in self.blank_positions:
                        available_positions.append((r, c))
            
            # Clear all samples from the plate
            for r, c, _ in samples_with_positions:
                plate.set_sample(r, c, None)
            
            # Place sorted samples in available positions
            for i, (_, _, sample_obj) in enumerate(samples_with_positions):
                if i < len(available_positions):
                    r, c = available_positions[i]
                    plate.set_sample(r, c, sample_obj)
        
        # Verify ordering was applied based on source_row_index
        self._verify_sample_ordering()

    def _verify_sample_ordering(self):
        """
        Verify that all samples are properly ordered by ID within each plate.
        This is a debugging method to ensure the sorting worked correctly.
        """
        for plate_idx, plate in enumerate(self.plates):
            # Collect all samples in order they appear on the plate
            # Use column-first ordering for consistency
            plate_samples = []
            for c in range(plate.cols):
                for r in range(plate.rows):
                    sample_obj = plate.get_sample(r, c)
                    if sample_obj:
                        plate_samples.append(sample_obj)
            
            # Check if samples are in order by source_row_index (when present)
            if len(plate_samples) > 1:
                for i in range(1, len(plate_samples)):
                    prev = plate_samples[i-1]
                    curr = plate_samples[i]
                    prev_idx = getattr(prev, 'source_row_index', None)
                    curr_idx = getattr(curr, 'source_row_index', None)
                    if prev_idx is not None and curr_idx is not None and prev_idx > curr_idx:
                        print(
                            f"Warning: Ordering issue on plate {self.starting_plate_number + plate_idx}: "
                            f"row {prev_idx} comes before {curr_idx}"
                        )








    def fill_plates(self):
        """Fill plates with samples, leaving designated blank positions empty for negative controls."""        
        self._place_samples()
        self._sort_samples_by_id_within_plates()

    def get_plates(self) -> List[Plate]:
        return self.plates

    def print_plates(self):
        for i, plate in enumerate(self.plates):
            print(f"Plate {self.starting_plate_number + i}:")
            print(plate)
            print()

    def colony_balance(self):
        """Return a list of dicts, one per plate, with counts of each colony_code."""
        result = []
        for plate in self.plates:
            colony_counts = {}
            # Use column-first ordering for consistency
            for c in range(plate.cols):
                for r in range(plate.rows):
                    sample_obj = plate.get_sample(r, c)
                    # Check if the cell contains a sample object (all samples are now treated equally)
                    if sample_obj:
                        colony = sample_obj.colony_code
                        colony_counts[colony] = colony_counts.get(colony, 0) + 1
            result.append(colony_counts)
        return result

    def sample_type_counts(self):
        """Return a list of dicts, one per plate, with counts of adults, chicks, and blanks."""
        result = []
        for plate in self.plates:
            counts = {"ADULT": 0, "CHICK": 0, "BLANK": 0}
            # Use column-first ordering for consistency
            for c in range(plate.cols):
                for r in range(plate.rows):
                    sample_obj = plate.get_sample(r, c)
                    if sample_obj:
                        if sample_obj.sample_type == SampleType.ADULT:
                            counts["ADULT"] += 1
                        elif sample_obj.sample_type == SampleType.CHICK:
                            counts["CHICK"] += 1
                        elif sample_obj.sample_type == SampleType.BLANK:
                            counts["BLANK"] += 1
            result.append(counts)
        return result

    def contiguous_sample_measure(self):
        """Return a list of floats, one per plate, representing the average contiguous run length of samples."""
        result = []
        for plate in self.plates:
            runs = []
            # Check rows (left to right)
            for r in range(plate.rows):
                run = 0
                for c in range(plate.cols):
                    sample_obj = plate.get_sample(r, c)
                    # Check if the cell contains a sample object (all samples are now treated equally)
                    if sample_obj:
                        run += 1
                    else:
                        if run > 0:
                            runs.append(run)
                        run = 0
                if run > 0:
                    runs.append(run)
            # Check columns (top to bottom) - this is now the primary filling direction
            for c in range(plate.cols):
                run = 0
                for r in range(plate.rows):
                    sample_obj = plate.get_sample(r, c)
                     # Check if the cell contains a sample object (all samples are now treated equally)
                    if sample_obj:
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
            # Use column-first ordering for consistency
            for c in range(plate.cols):
                for r in range(plate.rows):
                    sample_obj = plate.get_sample(r, c)
                    if sample_obj:
                        colony = sample_obj.colony_code
                        colony_counts[colony] = colony_counts.get(colony, 0) + 1
            
            # Get sample type counts for this plate
            type_counts = {"ADULT": 0, "CHICK": 0, "BLANK": 0}
            # Use column-first ordering for consistency
            for c in range(plate.cols):
                for r in range(plate.rows):
                    sample_obj = plate.get_sample(r, c)
                    if sample_obj:
                        if sample_obj.sample_type == SampleType.ADULT:
                            type_counts["ADULT"] += 1
                        elif sample_obj.sample_type == SampleType.CHICK:
                            type_counts["CHICK"] += 1
                        elif sample_obj.sample_type == SampleType.BLANK:
                            type_counts["BLANK"] += 1
            
            # Calculate total samples on this plate
            total_samples = type_counts["ADULT"] + type_counts["CHICK"] + type_counts["BLANK"]
            
            plate_stats.append({
                "plate_number": self.starting_plate_number + i,
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

        # 2. Adult/chick/blank balance score
        type_counts_per_plate = self.sample_type_counts()
        for d in type_counts_per_plate:
            d.setdefault("ADULT", 0)
            d.setdefault("CHICK", 0)
            d.setdefault("BLANK", 0)
        type_stdevs = []
        for t in ["ADULT", "CHICK", "BLANK"]:
            counts = [d[t] for d in type_counts_per_plate]
            if len(counts) > 1:
                type_stdevs.append(np.std(counts))
        adult_chick_blank_balance_score = float(np.mean(type_stdevs)) if type_stdevs else 0.0

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
        norm_type = adult_chick_blank_balance_score / max_balance if max_balance else 0
        norm_contig = contiguity_score / max_contiguity if max_contiguity else 0
        norm_util = 1 - plate_utilization
        overall = float(np.mean([norm_colony, norm_type, norm_contig, norm_util]))

        return {
            "colony_balance_score": colony_balance_score,
            "adult_chick_blank_balance_score": adult_chick_blank_balance_score,
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
        # Consider minimum text length for cell sizing
        max_sid_length = max(max_sid_length, 10)
        
        # Ensure we have a reasonable minimum length for calculations
        max_sid_length = max(max_sid_length, 10)

        # Estimate required cell size based on max text length and font size
        # Using a heuristic: char_width approx 0.6 * font_height
        fontsize = 10  # Increased font size for better readability
        points_per_inch = 72
        char_width_inch = 0.6 * (fontsize / points_per_inch)
        
        # Calculate how many characters can fit in a reasonable cell width
        # We want cells to be wide enough for most IDs without being too wide
        target_cell_width_inch = 1.0  # Increased target cell width for better text fitting
        
        # Calculate how many characters can fit in the target width
        chars_per_line = int(target_cell_width_inch / char_width_inch) - 3  # Leave more padding for better readability
        
        # If the longest ID is longer than what fits in one line, we'll need to wrap
        if max_sid_length > chars_per_line:
            # Calculate how many lines we need
            lines_needed = (max_sid_length + chars_per_line - 1) // chars_per_line
            # Increase cell height to accommodate multiple lines
            cell_height_multiplier = max(1.0, lines_needed * 0.8)
        else:
            cell_height_multiplier = 1.0

        # Base cell size (for visual spacing even with short text)
        cell_width = target_cell_width_inch
        cell_height = 1.1 * cell_height_multiplier  # Increased base height for better text spacing

        # For PDF, we'll use a standard page size and fit each plate appropriately
        fig_width = self.cols * cell_width
        fig_height = self.rows * cell_height

        # Cap figure size to prevent excessively large pages
        max_fig_width = 14  # inches - increased for better readability
        max_fig_height = 12  # inches - increased for better readability
        fig_width = min(max_fig_width, fig_width)
        fig_height = min(max_fig_height, fig_height)

        # Create PDF with PdfPages
        buf = io.BytesIO()
        with PdfPages(buf) as pdf:
            for i, plate in enumerate(self.plates):
                # Create a new figure for each plate
                fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height), facecolor='white')
                ax.set_facecolor('white')
                
                # Set the axis limits to ensure proper cell positioning
                ax.set_xlim(-0.5, self.cols - 0.5)
                ax.set_ylim(-0.5, self.rows - 0.5)
                
                # Invert y-axis so that row A is at the top
                ax.invert_yaxis()

                # Create a grid of cells
                grid = np.zeros((self.rows, self.cols, 3))
                
                # Fill in the colors based on sample type (Empty or Sample)
                for r in range(self.rows):
                    for c in range(self.cols):
                        sample_obj = plate.get_sample(r, c) # Get the Sample object

                        if sample_obj is None:
                            # Check if this is a designated blank position
                            if (r, c) in self.blank_positions:
                                grid[r, c] = [0.95, 0.95, 0.95]  # Very light gray for designated blank positions
                            else:
                                grid[r, c] = [1, 1, 1]  # White for empty cells
                        else: # It's a Sample
                            # Use a light blue color for samples
                            grid[r, c] = [0.7, 0.85, 0.95]  # Light blue for samples

                # Display the grid
                ax.imshow(grid)
                ax.set_title(f'Plate {self.starting_plate_number + i}', fontsize=16, fontweight='bold', color='black', pad=20)
                
                # Add grid lines
                ax.set_xticks(np.arange(-.5, self.cols, 1), minor=True)
                ax.set_yticks(np.arange(-.5, self.rows, 1), minor=True)
                ax.grid(which='minor', color='#b0b0b0', linestyle='-', linewidth=2.0)
                
                # Add row and column labels with better visibility
                ax.set_xticks(np.arange(self.cols))
                ax.set_yticks(np.arange(self.rows))
                ax.set_xticklabels([str(i+1) for i in range(self.cols)], fontsize=12, fontweight='bold')
                ax.set_yticklabels([chr(ord('A') + i) for i in range(self.rows)], fontsize=12, fontweight='bold')
                
                # Style the tick labels for better visibility
                ax.tick_params(axis='both', which='major', labelsize=12, colors='black', width=2)
                ax.tick_params(axis='both', which='minor', labelsize=10, colors='black')
                
                # Make tick marks more visible
                ax.tick_params(axis='both', which='major', length=6, width=2)
                ax.tick_params(axis='both', which='minor', length=4, width=1)
                
                # Remove axis spines for cleaner look
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)

                # Add cell IDs (sample_id)
                for r in range(self.rows):
                    for c in range(self.cols):
                        sample_obj = plate.get_sample(r, c)
                        if sample_obj:
                            display_text = sample_obj.unique_id if sample_obj.unique_id else ""
                            
                            # Always use black text for maximum contrast against white/light backgrounds
                            text_color = 'black'
                            
                            # Skip empty text
                            if not display_text.strip():
                                continue
                            
                            # Wrap text if it's too long for the cell
                            if len(display_text) > chars_per_line:
                                # Split text into multiple lines, trying to break at word boundaries
                                wrapped_lines = []
                                current_line = ""
                                
                                # Split by common delimiters first (dashes, underscores, etc.)
                                words = display_text.replace('-', ' - ').replace('_', ' _ ').split()
                                
                                for word in words:
                                    if len(current_line) + len(word) + 1 <= chars_per_line:
                                        if current_line:
                                            current_line += " " + word
                                        else:
                                            current_line = word
                                    else:
                                        if current_line:
                                            wrapped_lines.append(current_line)
                                        current_line = word
                                
                                if current_line:
                                    wrapped_lines.append(current_line)
                                
                                # If we still have lines that are too long, force break them
                                final_lines = []
                                for line in wrapped_lines:
                                    if len(line) <= chars_per_line:
                                        final_lines.append(line)
                                    else:
                                        # Force break long lines, but try to break at better positions
                                        remaining = line
                                        while len(remaining) > chars_per_line:
                                            # Try to break at a dash or underscore if possible
                                            break_pos = chars_per_line
                                            for i in range(min(chars_per_line, len(remaining)), 0, -1):
                                                if remaining[i-1] in ['-', '_', ' ']:
                                                    break_pos = i
                                                    break
                                            final_lines.append(remaining[:break_pos])
                                            remaining = remaining[break_pos:]
                                        if remaining:
                                            final_lines.append(remaining)
                                
                                # Join lines with newline characters
                                wrapped_text = '\n'.join(final_lines)
                                
                                # Add the wrapped text
                                ax.text(c, r, wrapped_text, 
                                       ha='center', va='center',
                                       color=text_color,
                                       fontsize=fontsize-1,  # Slightly smaller font for wrapped text
                                       weight='bold',
                                       family='monospace',  # Use monospace font for better alignment
                                       bbox=dict(facecolor='white', 
                                               edgecolor='#333333',
                                               alpha=0.95,
                                               boxstyle='round,pad=0.5',
                                               linewidth=2.0))
                            else:
                                # Add the single-line text
                                ax.text(c, r, display_text, 
                                       ha='center', va='center',
                                       color=text_color,
                                       fontsize=fontsize,
                                       weight='bold',
                                       family='monospace',  # Use monospace font for better alignment
                                       bbox=dict(facecolor='white', 
                                               edgecolor='#333333',
                                               alpha=0.95,
                                               boxstyle='round,pad=0.4',
                                               linewidth=2.0))

                # Adjust layout to ensure text doesn't get cut off
                plt.tight_layout(pad=2.5)
                
                # Add the page to the PDF
                pdf.savefig(fig, bbox_inches='tight', dpi=300)
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
            csv_data.append([f"Plate {self.starting_plate_number + i}"])

            # Add column headers row (empty first cell for row labels)
            header_row = [""] + [str(c + 1) for c in range(plate.cols)]
            csv_data.append(header_row)

            # Use column-first ordering for consistency with the new filling pattern
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
            # Use column-first ordering for consistency with the new filling pattern
            for c in range(plate.cols):
                for r in range(plate.rows):
                    sample_obj = plate.get_sample(r, c)
                    if sample_obj:
                        row_data = [
                            f"Plate {self.starting_plate_number + i}",
                            chr(ord('A') + r),
                            str(c + 1),
                            sample_obj.sample_id if hasattr(sample_obj, 'sample_id') else "",
                            sample_obj.unique_id if hasattr(sample_obj, 'unique_id') else "",
                            sample_obj.species if hasattr(sample_obj, 'species') else "",
                            sample_obj.colony_code if hasattr(sample_obj, 'colony_code') else "",
                            sample_obj.sample_type.value if (hasattr(sample_obj, 'sample_type') and sample_obj.sample_type) else "",
                            sample_obj.substrate if hasattr(sample_obj, 'substrate') else "",
                            sample_obj.notes if hasattr(sample_obj, 'notes') else ""
                        ]
                        csv_data.append(row_data)
                    else:
                        # Empty cell
                        row_data = [
                            f"Plate {self.starting_plate_number + i}",
                            chr(ord('A') + r),
                            str(c + 1),
                            "", "", "", "", "", "", ""
                        ]
                        csv_data.append(row_data)
        
        # Join rows into a single CSV string
        csv_string = "\n".join([",".join(map(str, row)) for row in csv_data])
        return csv_string