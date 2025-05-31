import io
import base64
import streamlit as st
import pandas as pd

from datamodel.test_manager import TestManager
from parse_samples import parse_samples

st.set_page_config(page_title="Dilution Sample Plating", layout="wide")
st.title("Dilution Sample Plating")

st.markdown(
    """<style>
    div[data-testid="stHorizontalBlock"] > div {
        gap: 0.1rem !important;
    }
    div[data-testid="stCheckbox"] {
        margin-bottom: 0.1rem !important;
    }
    </style>""",
    unsafe_allow_html=True,
)

# Initialize session state variables if they don't exist
if 'step' not in st.session_state:
    st.session_state['step'] = 1
if 'rows' not in st.session_state:
    st.session_state['rows'] = 8
if 'cols' not in st.session_state:
    st.session_state['cols'] = 12
if 'samples' not in st.session_state:
    st.session_state['samples'] = []
if 'blank_positions' not in st.session_state:
    st.session_state['blank_positions'] = []

# --- Step 1: Upload CSV ---
def step1_upload_csv():
    st.header("Step 1: Upload Sample CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        temp_csv = io.StringIO()
        df.to_csv(temp_csv, index=False)
        temp_csv.seek(0)
        samples = parse_samples(temp_csv)
        st.session_state["samples"] = samples
        st.success(f"Loaded {len(samples)} samples from file.")
    elif "samples" in st.session_state:
        st.info(f"{len(st.session_state['samples'])} samples loaded.")
    else:
        st.info("No file uploaded yet.")
    if (uploaded_file or "samples" in st.session_state):
        if st.button("Next"):
            st.session_state['step'] = 2
            st.rerun()

# --- Step 2: Plate Dimensions ---
def step2_plate_dimensions():
    st.header("Step 2: Specify Plate Dimensions")
    
    samples = st.session_state.get("samples")
    col1, col2 = st.columns(2)
    with col1:
        st.number_input("Number of rows", min_value=1, max_value=26, value=st.session_state.get('rows', 8), key="rows")
    with col2:
        st.number_input("Number of columns", min_value=1, max_value=24, value=st.session_state.get('cols', 12), key="cols")

    back_clicked = st.button("Back")
    next_clicked = st.button("Next")

    if back_clicked:
        st.session_state['step'] = 1
        st.rerun()
    if next_clicked:
        rows = st.session_state.get("rows")
        cols = st.session_state.get("cols")
        if not samples:
             st.warning("Please upload a CSV file first (Step 1).")
        elif not rows or not cols:
             st.warning("Please specify both rows and columns.")
        else:
            st.session_state['step'] = 3
            st.rerun()

# --- Step 3: BLANK Selection & Summary ---
def step3_blank_selection():
    st.header("Step 3: Set BLANK Sample Locations")

    samples = st.session_state.get("samples")
    rows = st.session_state.get("rows")
    cols = st.session_state.get("cols")

    if samples is None and "samples" in st.session_state:
         samples = st.session_state["samples"]
    if rows is None and "rows" in st.session_state:
         rows = st.session_state["rows"]
    if cols is None and "cols" in st.session_state:
         cols = st.session_state["cols"]

    if not all([samples, rows, cols]):
        st.warning("Please complete the previous steps first.")
        if st.button("Go to Step 1"):
            st.session_state['step'] = 1
            st.rerun()
        st.stop()

    if 'blank_positions' not in st.session_state:
        st.session_state.blank_positions = []

    header_cols = st.columns(cols + 1)
    for c in range(cols):
        header_cols[c + 1].markdown(f"**{c + 1}**")

    def update_blank_positions(r, c):
        # Checkbox callback to update persisted BLANK positioning
        pos_tuple = (r, c)
        checkbox_key = f"cell_{r}_{c}"
        is_checked = st.session_state[checkbox_key]

        if is_checked and pos_tuple not in st.session_state.blank_positions:
            st.session_state.blank_positions.append(pos_tuple)
        elif not is_checked and pos_tuple in st.session_state.blank_positions:
            st.session_state.blank_positions.remove(pos_tuple)

    # BLANK location checkbox grid display
    for r in range(rows):
        row_cols = st.columns(cols + 1)
        row_label = chr(ord('A') + r)
        row_cols[0].markdown(f"**{row_label}**")
        for c in range(cols):
            key = f"cell_{r}_{c}"
            initial_value = (r, c) in st.session_state.blank_positions
            
            row_cols[c + 1].checkbox(
                "BLANK",
                value=initial_value,
                key=key,
                label_visibility="collapsed",
                on_change=update_blank_positions,
                args=(r, c)
            )

    if 'blank_mask' in st.session_state:
        del st.session_state.blank_mask
    if 'mask_rows' in st.session_state:
        del st.session_state.mask_rows
    if 'mask_cols' in st.session_state:
        del st.session_state.mask_cols

    st.caption("Check any cell to mark as BLANK. BLANK cells are highlighted.")

    if st.button("Back"):
        st.session_state['step'] = 2
        st.rerun()
    if st.button("Next"):
        st.session_state['step'] = 4
        st.rerun()

# --- Step 4: Display Filled Plates ---
def step4_display_plates():
    samples = st.session_state.get("samples")
    rows = st.session_state.get("rows")
    cols = st.session_state.get("cols")
    blank_positions = st.session_state.get("blank_positions", [])

    if not all([samples, rows, cols, blank_positions]):
        st.warning("Missing data to display plates. Please go back to Step 1.")
        if st.button("Go to Step 1"):
            st.session_state['step'] = 1
            st.rerun()
    manager = TestManager(samples, rows, cols, blank_positions)
    manager.fill_plates()


    # Image display and download
    st.markdown(manager.get_plate_visualization_html(), unsafe_allow_html=True)
    img_str = manager.generate_plate_visualization()
    if img_str:
        st.download_button(
            label="Download Plate Visualization (PNG)",
            data=base64.b64decode(img_str),
            file_name="assay_plates.png",
            mime="image/png"
        )

    # CSV display and download
    csv_string = manager.to_csv_string()
    st.download_button(
        label="Download Plate Data (CSV)",
        data=csv_string,
        file_name="assay_plates.csv",
        mime="text/csv"
    )

    with st.expander("Show Plates"):
        for i, plate in enumerate(manager.plates):
            st.text(plate.to_string(f"PLATE {i+1}"))

    if st.button("Back"):
        st.session_state['step'] = 3
        st.rerun()

# Top-level app sequencing
if st.session_state['step'] == 1:
    step1_upload_csv()
elif st.session_state['step'] == 2:
    step2_plate_dimensions()
elif st.session_state['step'] == 3:
    step3_blank_selection()
elif st.session_state['step'] == 4:
    step4_display_plates()
