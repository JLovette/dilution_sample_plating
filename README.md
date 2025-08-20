# Dilution Sample Plating

A tool for automatically arranging samples and dilutions on assay plates for use by the [Lovette Lab](https://lovette.eeb.cornell.edu/).

## Features

- Automatically places samples across multiple plates based on lab requirements
- Balances colonies and sample types (AD, CH, BLANK) evenly across plates
- Keeps samples in order and fills plates left-to-right, top-to-bottom
- Preserves fixed BLANK positions in the layout
- Generates PDF output with each plate on a separate page for easy printing and sharing
- Provides PNG visualization as an alternative option

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/dilution_sample_plating.git
cd dilution_sample_plating
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
pip install streamlit  # For web interface
```

## Usage

### Web Interface (Streamlit)

To launch the web interface:

```bash
streamlit run app.py
```

The web interface provides a step-by-step workflow:

1. **Upload CSV**: Upload your sample data in CSV format
2. **BLANK Selection**: Mark positions for BLANK samples on the plate (8x12 grid)
3. **View Results**: See the generated plate layouts and download results as PDF (recommended) or PNG

### Command Line Interface

The CLI version can be used for batch processing or automation:

```bash
python main.py --input_file samples.csv --blank 0,0 --blank 0,1
```

**Note**: Plates are fixed at 8 rows × 12 columns (96 wells per plate).

## Input CSV Format

Input CSV file format was based on the provided example `/example_data/sample_data.csv`.

The input CSV file must contain the following columns:

- `Sample ID`: Unique identifier for each sample
- `Unique_ID`: Additional unique identifier for tracking
- `Colony_code`: Code identifying the colony of origin
- `AD/chick`: Designation of sample type, must be one of:
  - `CH`: For chick samples
  - `AD`: For adult samples
  - `NA,""`: For BLANK samples
- `Species`: `BLANK` for controls, species code otherwise (eg. `BLCK`)

Example CSV format:
```csv
Sample ID,Unique_ID,Colony_code,AD/chick,Species
S001,U123,COL1,CH,BLCK
S002,U124,COL1,AD,BLCK
S003,U125,COL2,NA,BLANK
``` 