import streamlit as st
import pandas as pd
from datamodel.test_manager import TestManager
from parse_samples import parse_samples
import io
import json
import base64

st.set_page_config(page_title="Assay Plate BLANK Selector", layout="wide")
st.title("Assay Plate BLANK Selector")

# Add CSS to reduce spacing between checkboxes and columns
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

# Use session state to track the current step
if 'step' not in st.session_state:
    st.session_state['step'] = 1

# Initialize critical session state variables if they don't exist
# This helps ensure they are present from the start of every rerun
if 'rows' not in st.session_state:
    st.session_state['rows'] = 8 # Default value
if 'cols' not in st.session_state:
    st.session_state['cols'] = 12 # Default value
if 'samples' not in st.session_state:
    st.session_state['samples'] = [] # Default empty list
if 'blank_positions' not in st.session_state:
    st.session_state['blank_positions'] = [] # Default empty list

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
    
    # Get samples from session state for validation, but allow entering dimensions first
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
        # Perform validation before moving to the next step
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
    st.header("Step 3: Select BLANK Locations")
    
    # Get values from session state with validation
    # Attempt to get values, provide None if not found initially
    samples = st.session_state.get("samples") # Use get without default initially
    rows = st.session_state.get("rows") # Use get without default initially
    cols = st.session_state.get("cols") # Use get without default initially

    # If values are None but keys exist, try getting them directly (more strict)
    if samples is None and "samples" in st.session_state:
         samples = st.session_state["samples"]
    if rows is None and "rows" in st.session_state:
         rows = st.session_state["rows"]
    if cols is None and "cols" in st.session_state:
         cols = st.session_state["cols"]

    # If any required data is still missing, display warning and stop
    if not all([samples, rows, cols]):
        st.warning("Please complete the previous steps first.")
        if st.button("Go to Step 1"):
            st.session_state['step'] = 1
            st.rerun()
        st.stop() # Stop execution here to prevent the TypeError

    # Now that we are sure samples, rows, and cols are not None, we can safely use them.

    # Initialize blank_positions if not in session state
    if 'blank_positions' not in st.session_state:
        st.session_state.blank_positions = []

    print(f"rows: {rows}, cols: {cols}, len(samples): {len(samples)}")

    # Column headers
    header_cols = st.columns(cols + 1)
    # The first column is for row labels, leave it empty in the header
    for c in range(cols):
        header_cols[c + 1].markdown(f"**{c + 1}**")

    # Create a callback to update the blank positions
    def update_blank_positions(r, c):
        # The values of rows, cols, samples should be available in the scope
        # due to the checks at the beginning of step3_blank_selection
        
        pos_tuple = (r, c)
        # Check the current state of the checkbox to determine action
        # Streamlit passes the *new* value to the callback implicitly via session_state[key]
        # Get the key used for the checkbox
        checkbox_key = f"cell_{r}_{c}"
        is_checked = st.session_state[checkbox_key]

        if is_checked and pos_tuple not in st.session_state.blank_positions:
            st.session_state.blank_positions.append(pos_tuple)
        elif not is_checked and pos_tuple in st.session_state.blank_positions:
            st.session_state.blank_positions.remove(pos_tuple)

        # Optional: Keep the blank positions list sorted for consistency/easier debugging
        # st.session_state.blank_positions.sort()

    # Display the grid
    for r in range(rows):
        row_cols = st.columns(cols + 1)
        row_label = chr(ord('A') + r)
        row_cols[0].markdown(f"**{row_label}**")
        for c in range(cols):
            key = f"cell_{r}_{c}"
            # Determine initial value based on blank_positions list
            initial_value = (r, c) in st.session_state.blank_positions
            
            row_cols[c + 1].checkbox(
                "BLANK",
                value=initial_value,
                key=key,
                label_visibility="collapsed",
                on_change=update_blank_positions,
                args=(r, c) # Pass row and col to the callback
            )

    # Remove the old blank_mask if it exists in session state (for cleanup after refactoring)
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
        # Run the plate filling algorithm and store the result
        # Use the validated samples, rows, cols, and the blank_positions list
        manager = TestManager(samples, rows, cols, st.session_state.blank_positions)
        st.session_state.filled_plates = manager.get_plates()
        st.session_state['step'] = 4
        st.rerun()

# --- Step 4: Display Filled Plates ---
def step4_display_plates():
    st.header("Filled Assay Plates")
    if 'filled_plates' in st.session_state:
        # Create the plate manager for visualization
        # Need samples, rows, cols, blank_positions to recreate manager for visualization
        samples = st.session_state.get("samples")
        rows = st.session_state.get("rows")
        cols = st.session_state.get("cols")
        blank_positions = st.session_state.get("blank_positions", [])

        if not all([samples, rows, cols]):
             st.warning("Missing data to display plates. Please go back to Step 1.")
             if st.button("Go to Step 1"):
                 st.session_state['step'] = 1
                 st.rerun()
        else:
            manager = TestManager(
                samples,
                rows,
                cols,
                blank_positions
            )
            manager.plates = st.session_state.filled_plates
            
            # Display the visualization
            st.markdown(manager.get_plate_visualization_html(), unsafe_allow_html=True)
            
            # Add download button for visualization
            img_str = manager.generate_plate_visualization()
            if img_str:
                st.download_button(
                    label="Download Plate Visualization (PNG)",
                    data=base64.b64decode(img_str),
                    file_name="assay_plates.png",
                    mime="image/png"
                )

            # Add download button for CSV
            csv_string = manager.to_csv_string()
            st.download_button(
                label="Download Plate Data (CSV)",
                data=csv_string,
                file_name="assay_plates.csv",
                mime="text/csv"
            )
            
            # Display the text representation
            with st.expander("Show Plates"):
                for i, plate in enumerate(st.session_state.filled_plates):
                    st.text(plate.to_string(f"PLATE {i+1}"))
    else:
        st.warning("No filled plates found in session state.")

    if st.button("Back"):
        st.session_state['step'] = 3
        st.rerun()

# Main app logic based on step
if st.session_state['step'] == 1:
    step1_upload_csv()
elif st.session_state['step'] == 2:
    step2_plate_dimensions()
elif st.session_state['step'] == 3:
    step3_blank_selection()
elif st.session_state['step'] == 4:
    step4_display_plates()

            