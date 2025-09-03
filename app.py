import io
import base64
from datamodel.sample import SampleType
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
if 'colony_weight' not in st.session_state:
    st.session_state['colony_weight'] = 1.0
if 'type_weight' not in st.session_state:
    st.session_state['type_weight'] = 2.0
if 'starting_plate_number' not in st.session_state:
    st.session_state['starting_plate_number'] = 1

# Fixed plate dimensions
ROWS = 8
COLS = 12

# --- Step 1: Upload CSV ---
def step1_upload_csv():
    st.header("1: Upload Sample CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file:
        # Store the uploaded file in session state for later use
        st.session_state["uploaded_file"] = uploaded_file
        
        df = pd.read_csv(uploaded_file)
        temp_csv = io.StringIO()
        df.to_csv(temp_csv, index=False)
        temp_csv.seek(0)
        samples = parse_samples(temp_csv)
        adult_samples = [s for s in samples if s.sample_type == SampleType.ADULT]
        chick_samples = [s for s in samples if s.sample_type == SampleType.CHICK]
        blank_samples = [s for s in samples if s.sample_type == SampleType.BLANK]
        st.session_state["samples"] = samples
        colonies = set(s.colony_code for s in samples if s.colony_code)
        st.markdown(f'<div style="background-color: #d4edda; border: 1px solid #c3e6cb; color: #155724; padding: 10px; border-radius: 5px; margin: 10px 0;">Loaded {len(samples)} total samples for {len(colonies)} colonies ({len(adult_samples)} ADULT, {len(chick_samples)} CHICK, {len(blank_samples)} BLANK)</div>', unsafe_allow_html=True)

    if (uploaded_file):
        if st.button("Next"):
            st.session_state['step'] = 2
            st.rerun()



# --- Step 2: Blank Position Selection & Summary ---
def step2_blank_selection():
    st.header("2: Set Blank Position Locations")

    samples = st.session_state.get("samples")

    if not samples:
        st.markdown('<div style="background-color: #fff3cd; border: 1px solid #ffeaa7; color: #856404; padding: 10px; border-radius: 5px; margin: 10px 0;"><strong>⚠️ Warning:</strong> Please upload a CSV file first (Step 1).</div>', unsafe_allow_html=True)
        if st.button("Go to Step 1"):
            st.session_state['step'] = 1
            st.rerun()
        st.stop()

    if 'blank_positions' not in st.session_state:
        st.session_state.blank_positions = []
    
    # Calculate and display blank position recommendations
    total_samples = len(samples)
    num_plates = (total_samples + 95) // 96  # Ceiling division by 96 (8x12 plate capacity)
    # Recommend 2-4 blank positions per plate for negative controls
    recommended_blanks_per_plate = min(4, max(2, num_plates))

    header_cols = st.columns(COLS + 1)
    for c in range(COLS):
        header_cols[c + 1].markdown(f"**{c + 1}**", help=f"Column {c + 1}")

    def update_blank_positions(r, c):
        # Checkbox callback to update persisted blank positioning
        pos_tuple = (r, c)
        checkbox_key = f"cell_{r}_{c}"
        is_checked = st.session_state[checkbox_key]

        if is_checked and pos_tuple not in st.session_state.blank_positions:
            st.session_state.blank_positions.append(pos_tuple)
        elif not is_checked and pos_tuple in st.session_state.blank_positions:
            st.session_state.blank_positions.remove(pos_tuple)


    # Blank position checkbox grid display
    for r in range(ROWS):
        row_cols = st.columns(COLS + 1)
        row_label = chr(ord('A') + r)
        row_cols[0].markdown(f"**{row_label}**", help=f"Row {row_label}")
        for c in range(COLS):
            key = f"cell_{r}_{c}"
            initial_value = (r, c) in st.session_state.blank_positions
            
            # Create checkbox for blank position selection
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

    st.markdown("**Check any cell to mark as a blank position for negative controls.**")
    
    # Add default blank positions button
    st.markdown("---")

    # Alert if insufficient blank positions
    current_blanks = len(st.session_state.blank_positions)
    # if current_blanks < recommended_blanks_per_plate:
    #     st.markdown(f"""
    #     <div style="background-color: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; padding: 15px; border-radius: 8px; margin: 15px 0;">
    #     ⚠️ You currently have {current_blanks} blank positions selected. {recommended_blanks_per_plate} is recommended for negative controls during DNA extraction. 
    #     </div>
    #     """, unsafe_allow_html=True)

    st.markdown("---")
    
    # Apply custom CSS class for tighter spacing
    st.markdown('<div class="quick-actions-section">', unsafe_allow_html=True)
    # st.markdown("**Quick Actions:**")
    
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
                st.success("🧹 Cleared all blank positions!")
                st.rerun()


    # Algorithm Weight Settings
    st.markdown("---")
    st.subheader("Plate Numbering")
    st.markdown("Set the starting plate number for your assay:")
    
    starting_plate_number = st.number_input(
        "Starting Plate Number",
        min_value=1,
        max_value=9999,
        value=st.session_state.starting_plate_number,
        step=1,
        help="The first plate will be labeled with this number, subsequent plates will increment from here."
    )
    
    # Update session state when starting plate number changes
    if starting_plate_number != st.session_state.starting_plate_number:
        st.session_state.starting_plate_number = starting_plate_number
        st.rerun()
    
    st.markdown("---")
    st.subheader("Algo Settings")
    st.markdown("Adjust how the algorithm prioritizes colony distribution vs. adult/chick ratio balancing:")
    
    weight_cols = st.columns(2)
    
    with weight_cols[0]:
        st.markdown("**Colony Balance Weight**")
        st.markdown("Higher values prioritize even colony distribution across plates")
        colony_weight = st.slider(
            "Colony Weight", 
            min_value=0.1, 
            max_value=5.0, 
            value=st.session_state.colony_weight, 
            step=0.1,
            help="Weight for colony balance (default: 1.0). Higher values = more emphasis on spreading colonies evenly."
        )
    
    with weight_cols[1]:
        st.markdown("**Adult/Chick Ratio Weight**")
        st.markdown("Higher values prioritize mixing adult and chick samples on each plate")
        type_weight = st.slider(
            "Type Weight", 
            min_value=0.1, 
            max_value=5.0, 
            value=st.session_state.type_weight, 
            step=0.1,
            help="Weight for adult/chick balance (default: 2.0). Higher values = more emphasis on preventing segmentation."
        )
    
    # Update session state when weights change
    if colony_weight != st.session_state.colony_weight:
        st.session_state.colony_weight = colony_weight
        st.rerun()
    
    if type_weight != st.session_state.type_weight:
        st.session_state.type_weight = type_weight
        st.rerun()
    
    # Show current balance preview
    st.markdown("**Current Balance Preview:**")
    if colony_weight > type_weight:
        st.info("**Colony-focused**: Colonies will be distributed more evenly, but adult/chick ratios may be less balanced.")
    elif type_weight > colony_weight:
        st.info("**Ratio-focused**: Adult and chick samples will be mixed better, but colony distribution may be less even.")
    else:
        st.info("**Balanced**: Equal emphasis on both colony distribution and adult/chick ratios.")
    
    st.markdown(f"**Colony Weight:** {colony_weight:.1f} | **Type Weight:** {type_weight:.1f}")
    
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
    
    # Get the uploaded file from session state
    uploaded_file = st.session_state.get("uploaded_file")
    if not uploaded_file:
        st.error("No uploaded file found. Please go back to Step 1.")
        if st.button("Go to Step 1"):
            st.session_state['step'] = 1
            st.rerun()
        st.stop()
    
    # Create TestManager with the uploaded file
    manager = TestManager(uploaded_file, ROWS, COLS, blank_positions, 
                         st.session_state.colony_weight, st.session_state.type_weight,
                         starting_plate_number=st.session_state.starting_plate_number)
    manager.fill_plates()
    st.markdown("""
    <div style="text-align: center; margin: 20px 0;">
        <a href="data:application/pdf;base64,""" + base64.b64encode(manager.get_pdf_data()).decode() + """" 
           download="plate_layouts.pdf" 
           style="background-color: #dc3545; color: white; padding: 12px 24px; margin: 0 10px; 
                  text-decoration: none; border-radius: 5px; font-weight: bold; display: inline-block;">
            Download PDF
        </a>
        <a href="data:text/csv;base64,""" + base64.b64encode(manager.get_grid_csv_data()).decode() + """" 
           download="plate_layouts_grid.csv" 
           style="background-color: #007bff; color: white; padding: 12px 24px; margin: 0 10px; 
                  text-decoration: none; border-radius: 5px; font-weight: bold; display: inline-block;">
            Download Plate Layouts CSV
        </a>
        <a href="data:text/csv;base64,""" + base64.b64encode(manager.get_original_csv_with_plates_data()).decode() + """" 
           download="samples_with_plate_assignments.csv" 
           style="background-color: #28a745; color: white; padding: 12px 24px; margin: 0 10px; 
                  text-decoration: none; border-radius: 5px; font-weight: bold; display: inline-block;">
            Download Original CSV with Plate Assignments
        </a>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Show Plates"):
        for i, plate in enumerate(manager.plates):
                                        st.text(plate.to_string(f"PLATE {st.session_state.starting_plate_number + i}"))

    with st.expander("Plate Statistics", expanded=False):
        plate_stats = manager.get_plate_statistics()
        
        # Create a more visually appealing layout using Streamlit components
        # Use columns to create a grid layout
        num_plates = len(plate_stats)
        cols_per_row = min(3, num_plates)  # Max 3 plates per row
        
        for i in range(0, num_plates, cols_per_row):
            row_cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                if i + j < num_plates:
                    stats = plate_stats[i + j]
                    with row_cols[j]:
                        # Create a styled container for each plate
                        with st.container():
                            # Plate header with custom styling
                            st.markdown(f"""
                            <div style="
                                background: linear-gradient(90deg, #1f77b4, #0d5aa7);
                                color: white;
                                padding: 1rem;
                                border-radius: 10px 10px 0 0;
                                text-align: center;
                                font-weight: bold;
                                font-size: 1.1rem;
                                margin-bottom: 0;
                            ">
                            Plate {stats['plate_number']}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Plate content with border and background
                            st.markdown(f"""
                            <div style="
                                border: 2px solid #dee2e6;
                                border-top: none;
                                border-radius: 0 0 10px 10px;
                                padding: 1rem;
                                background: linear-gradient(135deg, #f8f9fa, #e9ecef);
                                margin-bottom: 1rem;
                            ">
                            """, unsafe_allow_html=True)
                            
                            # Sample Types Section
                            st.markdown("**📊 Sample Types**")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(f"**ADULT:** {stats['type_counts']['ADULT']}")
                                st.markdown(f"**CHICK:** {stats['type_counts']['CHICK']}")
                            with col2:
                                # No more BLANK samples - they are now treated like normal samples
                                st.markdown(f"**Total:** {stats['total_samples']}")
                            
                            st.markdown("---")
                            
                            # Colonies Section
                            st.markdown("**🏝️ Colonies**")
                            if stats['colony_counts']:
                                for colony, count in sorted(stats['colony_counts'].items()):
                                    st.markdown(f"• **{colony}:** {count}")
                            else:
                                st.markdown("*No colonies*")
                            
                            st.markdown("</div>", unsafe_allow_html=True)

    # Display current algorithm settings
    st.header("Algorithm Settings")
    settings_cols = st.columns(3)
    
    with settings_cols[0]:
        st.metric("Colony Weight", f"{st.session_state.colony_weight:.1f}")
    with settings_cols[1]:
        st.metric("Type Weight", f"{st.session_state.type_weight:.1f}")
    with settings_cols[2]:
        balance_type = "Colony-focused" if st.session_state.colony_weight > st.session_state.type_weight else \
                      "Ratio-focused" if st.session_state.type_weight > st.session_state.colony_weight else "Balanced"
        st.metric("Balance Type", balance_type)

    
    # Display detailed balance information in styled container
    st.header("Balance Analysis")
    
    # Create a styled balance analysis container
    with st.container():        
        # Colony balance details
        st.subheader("Colony Distribution Across Plates")
        colony_balance = manager.colony_balance()
        colony_df = pd.DataFrame(colony_balance).fillna(0)
        colony_df.index = [f"Plate {st.session_state.starting_plate_number + i}" for i in range(len(colony_df))]
        st.dataframe(colony_df, use_container_width=True)
        
        # Adult/Chick ratio details
        st.subheader("Adult/Chick Sample Distribution")
        type_counts = manager.sample_type_counts()
        type_df = pd.DataFrame(type_counts)
        type_df.index = [f"Plate {st.session_state.starting_plate_number + i}" for i in range(len(type_df))]
        
        # Calculate ratios
        type_df['ADULT_RATIO'] = (type_df['ADULT'] / (type_df['ADULT'] + type_df['CHICK'])).fillna(0)
        type_df['CHICK_RATIO'] = (type_df['CHICK'] / (type_df['ADULT'] + type_df['CHICK'])).fillna(0)
        
        st.dataframe(type_df, use_container_width=True)
        
        # Summary statistics
        st.subheader("Balance Summary")
        summary_cols = st.columns(3)
        
        with summary_cols[0]:
            metrics = manager.algorithm_metrics()
            st.metric("Colony Balance (Std Dev)", f"{metrics['colony_balance_score']:.2f}")
            st.metric("Type Balance (Std Dev)", f"{metrics['adult_chick_blank_balance_score']:.2f}")
        
        with summary_cols[1]:
            # Calculate ideal distribution
            total_adult = sum(type_df['ADULT'])
            total_chick = sum(type_df['CHICK'])
            ideal_adult_per_plate = total_adult / len(manager.plates) if len(manager.plates) > 0 else 0
            ideal_chick_per_plate = total_chick / len(manager.plates) if len(manager.plates) > 0 else 0
            
            st.metric("Ideal ADULT/Plate", f"{ideal_adult_per_plate:.1f}")
            st.metric("Ideal CHICK/Plate", f"{ideal_chick_per_plate:.1f}")
        
        with summary_cols[2]:
            # Show worst case deviations
            max_colony_dev = max([max(d.values()) if d else 0 for d in colony_balance]) - min([min(d.values()) if d else 0 for d in colony_balance])
            max_type_dev = max(type_df['ADULT'] + type_df['CHICK']) - min(type_df['ADULT'] + type_df['CHICK'])
            
            st.metric("Max Colony Deviation", f"{max_colony_dev:.0f}")
            st.metric("Max Type Deviation", f"{max_type_dev:.0f}")
        
        st.markdown("</div>", unsafe_allow_html=True)

    # Navigation button
    st.markdown("---")
    if st.button("New Plating", type="primary", use_container_width=True):
        st.session_state['step'] = 1
        st.rerun()

# Top-level app sequencing
if st.session_state['step'] == 1:
    step1_upload_csv()
elif st.session_state['step'] == 2:
    step2_blank_selection()
elif st.session_state['step'] == 3:
    step3_display_plates()
