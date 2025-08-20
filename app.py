import io
import base64
import streamlit as st
import pandas as pd

from datamodel.test_manager import TestManager
from parse_samples import parse_samples

st.set_page_config(page_title="Dilution Sample Plating", layout="wide", page_icon="🧬")

# Custom CSS for white background and improved styling
st.markdown(
    """<style>
    /* Main background */
    .main .block-container {
        background-color: white;
        padding-top: 2rem;
        padding-bottom: 2rem;
        box-shadow: 0 0 20px rgba(0,0,0,0.05);
        border-radius: 10px;
        margin: 1rem;
    }
    
    /* Page background */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Improve overall spacing and readability */
    .stMarkdown {
        line-height: 1.6;
    }
    
    /* Add subtle borders to improve element separation */
    .stButton, .stDownloadButton, .stSelectbox, .stTextInput, .stNumberInput {
        border-radius: 6px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Header styling */
    h1 {
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Horizontal block spacing */
    div[data-testid="stHorizontalBlock"] > div {
        gap: 0.1rem !important;
    }
    
    /* Checkbox styling */
    div[data-testid="stCheckbox"] {
        margin-bottom: 0.1rem !important;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    .stButton > button:hover {
        background-color: #0d5aa7;
    }
    
    /* Download button styling - ensure good contrast on white background */
    .stDownloadButton > button {
        background-color: #28a745 !important;
        color: white !important;
        border: 2px solid #28a745 !important;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 600 !important;
        text-shadow: 0 1px 2px rgba(0,0,0,0.3);
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stDownloadButton > button:hover {
        background-color: #218838 !important;
        border-color: #1e7e34 !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .stDownloadButton > button:active {
        background-color: #1e7e34 !important;
        transform: translateY(1px);
    }
    
    /* Success and info boxes */
    .stSuccess {
        background-color: #d4edda;
        border-color: #c3e6cb;
        color: #155724;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stInfo {
        background-color: #d1ecf1;
        border-color: #bee5eb;
        color: #0c5460;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Warning boxes */
    .stWarning {
        background-color: #fff3cd;
        border-color: #ffeaa7;
        color: #856404;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Error boxes */
    .stError {
        background-color: #f8d7da;
        border-color: #f5c6cb;
        color: #721c24;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Caption styling */
    .caption {
        color: #333;
        font-style: italic;
        font-weight: 500;
    }
    
    /* Ensure all text has good contrast */
    .stMarkdown, .stText, .stCaption {
        color: #2c3e50 !important;
    }
    
    /* Header text styling */
    h2, h3, h4 {
        color: #1f77b4;
        font-weight: 600;
        text-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    
    /* Improve contrast for all text elements */
    p, span, div, label {
        color: #2c3e50 !important;
    }
    
    /* Ensure form elements have good contrast */
    .stSelectbox, .stTextInput, .stNumberInput {
        color: #2c3e50 !important;
    }
    
    /* Improve checkbox visibility */
    .stCheckbox > label {
        color: #2c3e50 !important;
        font-weight: 500 !important;
    }
    
    /* Success and info text */
    .stSuccess, .stInfo {
        color: #155724 !important;
    }
    
    /* Warning text */
    .stWarning {
        color: #856404 !important;
    }
    
    /* Button text */
    .stButton > button {
        color: white !important;
        font-weight: 600;
    }
    
    /* Global text color override for all Streamlit elements */
    .stMarkdown, .stText, .stCaption, .stAlert, .stSuccess, .stInfo, .stWarning, .stError {
        color: #2c3e50 !important;
    }
    
    /* Ensure all div text is visible */
    div {
        color: #2c3e50 !important;
    }
    
    /* Override any Streamlit default text colors */
    .stMarkdown p, .stMarkdown div, .stMarkdown span {
        color: #2c3e50 !important;
    }
    
    /* Improve table text contrast */
    .stDataFrame {
        color: #2c3e50 !important;
    }
    
    /* Ensure expander text is visible */
    .streamlit-expanderHeader {
        color: #2c3e50 !important;
        font-weight: 600 !important;
    }
    
    /* Improve file uploader text */
    .stFileUploader > div > div {
        color: #2c3e50 !important;
    }
    
    /* Quick action buttons styling */
    .stButton > button[data-testid="baseButton-primary"] {
        background-color: #28a745 !important;
        border-color: #28a745 !important;
        font-weight: 600 !important;
    }
    
    .stButton > button[data-testid="baseButton-primary"]:hover {
        background-color: #218838 !important;
        border-color: #1e7e34 !important;
    }
    
    /* Reduce spacing between buttons in quick actions */
    .quick-actions-buttons .stButton {
        margin: 0 2px !important;
    }
    
    /* Tighter column spacing for button groups */
    div[data-testid="stHorizontalBlock"] > div {
        gap: 0.1rem !important;
    }
    
    /* Reduce white space around quick actions section */
    .quick-actions-section {
        margin: 0.5rem 0 !important;
        padding: 0.5rem !important;
    }
    
    /* Reduce spacing between rows */
    .quick-actions-section .stButton {
        margin: 0.2rem 0 !important;
    }
    
    /* Reduce margins around the Quick Actions heading */
    .quick-actions-section h4 {
        margin: 0.5rem 0 0.5rem 0 !important;
    }
    
    /* Reduce spacing between horizontal blocks */
    .quick-actions-section div[data-testid="stHorizontalBlock"] {
        margin: 0.2rem 0 !important;
    }
    
    /* Target Streamlit's default spacing more aggressively */
    .quick-actions-section .stMarkdown {
        margin: 0.2rem 0 !important;
    }
    
    /* Reduce spacing around buttons specifically */
    .quick-actions-section button {
        margin: 0.1rem !important;
        padding: 0.3rem 0.6rem !important;
    }
    
    /* Reduce spacing between columns even more */
    .quick-actions-section [data-testid="stHorizontalBlock"] > div {
        gap: 0.05rem !important;
        padding: 0 0.1rem !important;
    }
    </style>""",
    unsafe_allow_html=True,
)

st.title("Dilution Sample Plating")

# Initialize session state variables if they don't exist
if 'step' not in st.session_state:
    st.session_state['step'] = 1
if 'samples' not in st.session_state:
    st.session_state['samples'] = []
if 'blank_positions' not in st.session_state:
    st.session_state['blank_positions'] = []

# Fixed plate dimensions
ROWS = 8
COLS = 12

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
        st.markdown(f'<div style="background-color: #d4edda; border: 1px solid #c3e6cb; color: #155724; padding: 10px; border-radius: 5px; margin: 10px 0;"><strong>✅ Success:</strong> Loaded {len(samples)} samples from file.</div>', unsafe_allow_html=True)
    # elif "samples" in st.session_state:
    #     st.markdown(f'<div style="background-color: #d1ecf1; border: 1px solid #bee5eb; color: #0c5460; padding: 10px; border-radius: 5px; margin: 10px 0;"><strong>ℹ️ Info:</strong> {len(st.session_state["samples"])} samples loaded.</div>', unsafe_allow_html=True)
    # else:
    #     st.markdown('<div style="background-color: #d1ecf1; border: 1px solid #bee5eb; color: #0c5460; padding: 10px; border-radius: 5px; margin: 10px 0;"><strong>ℹ️ Info:</strong> No file uploaded yet.</div>', unsafe_allow_html=True)
    if (uploaded_file or "samples" in st.session_state):
        if st.button("Next"):
            st.session_state['step'] = 2
            st.rerun()



# --- Step 2: BLANK Selection & Summary ---
def step2_blank_selection():
    st.header("Step 2: Set BLANK Sample Locations")

    samples = st.session_state.get("samples")

    if not samples:
        st.markdown('<div style="background-color: #fff3cd; border: 1px solid #ffeaa7; color: #856404; padding: 10px; border-radius: 5px; margin: 10px 0;"><strong>⚠️ Warning:</strong> Please upload a CSV file first (Step 1).</div>', unsafe_allow_html=True)
        if st.button("Go to Step 1"):
            st.session_state['step'] = 1
            st.rerun()
        st.stop()

    if 'blank_positions' not in st.session_state:
        st.session_state.blank_positions = []

    header_cols = st.columns(COLS + 1)
    for c in range(COLS):
        header_cols[c + 1].markdown(f"**{c + 1}**", help=f"Column {c + 1}")

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
    for r in range(ROWS):
        row_cols = st.columns(COLS + 1)
        row_label = chr(ord('A') + r)
        row_cols[0].markdown(f"**{row_label}**", help=f"Row {row_label}")
        for c in range(COLS):
            key = f"cell_{r}_{c}"
            initial_value = (r, c) in st.session_state.blank_positions
            
            # Create checkbox for BLANK position selection
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

    st.markdown("**Check any cell to mark as BLANK.**")
    
    # Add default BLANK positions button
    st.markdown("---")
    
    # Apply custom CSS class for tighter spacing
    st.markdown('<div class="quick-actions-section">', unsafe_allow_html=True)
    st.markdown("**Quick Actions:**")
    
    # First row: Use Defaults and Clear buttons
    row1_col1, row1_col2 = st.columns([1, 1])
    
    with row1_col1:
        if st.button("Use Default Positioning", type="primary"):
            default_positions = [(0,0), (1,0), (1,1), (3,3), (6,5), (7,7), (4,9), (3,11), (7,11)]
            st.session_state.blank_positions = default_positions
            st.rerun()
    
    with row1_col2:
        if st.button("Clear"):
            st.session_state.blank_positions = []
            st.success("🧹 Cleared all BLANK positions!")
            st.rerun()
    
    # Second row: Back and Next buttons
    row2_col1, row2_col2 = st.columns([1, 1])
    
    with row2_col1:
        if st.button("Back"):
            st.session_state['step'] = 1
            st.rerun()
    
    with row2_col2:
        if st.button("Next"):
            st.session_state['step'] = 3
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# --- Step 3: Display Filled Plates ---
def step3_display_plates():
    samples = st.session_state.get("samples")
    blank_positions = st.session_state.get("blank_positions", [])

    if not all([samples, blank_positions]):
        st.markdown('<div style="background-color: #fff3cd; border: 1px solid #ffeaa7; color: #856404; padding: 10px; border-radius: 5px; margin: 10px 0;"><strong>⚠️ Warning:</strong> Missing data to display plates. Please go back to Step 1.</div>', unsafe_allow_html=True)
        if st.button("Go to Step 1"):
            st.session_state['step'] = 1
            st.rerun()
    manager = TestManager(samples, ROWS, COLS, blank_positions)
    manager.fill_plates()


    # Image display and download
    st.markdown(manager.get_plate_visualization_pdf_html(), unsafe_allow_html=True)

    with st.expander("Show Plates"):
        for i, plate in enumerate(manager.plates):
            st.text(plate.to_string(f"PLATE {i+1}"))

    if st.button("Back"):
        st.session_state['step'] = 2
        st.rerun()

# Top-level app sequencing
if st.session_state['step'] == 1:
    step1_upload_csv()
elif st.session_state['step'] == 2:
    step2_blank_selection()
elif st.session_state['step'] == 3:
    step3_display_plates()
