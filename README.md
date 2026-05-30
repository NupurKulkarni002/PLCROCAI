# 🏭 Electroplating Plant Hoist Scheduler - Streamlit GUI

An interactive web-based GUI for scheduling wagon movements through electroplating stations with real-time analysis and visualization.

## Features

### ⚙️ Configuration Tab
- Upload three input CSV files:
  - `input_tanks_csv.csv` - Station/tank configuration
  - `input_wagon_new.csv` - Wagon/hoist parameters
  - `input_Crosstrolley.csv` (optional)
- Set simulation parameters:
  - Total loads to process
  - Project ID
  - Program ID
  - Wagon number
- Generate sequence with one click

### 📊 Execution Tab
- View complete generated instruction sequence
- Download OUTPUT_sequence.csv
- Summary metrics:
  - Total instructions
  - Number of loads processed
  - Total cycle time
  - Pick-up operations count

### 📈 Analysis Tab
Complete analysis with multiple visualizations:

1. **Dip Time Audit Log**
   - Detailed table of all dip operations
   - Entry/exit times
   - Pass/fail status
   - Download capability

2. **Cycle Time Analysis**
   - Per-load cycle time bar chart
   - Average, max, min metrics
   - Load-by-load comparison

3. **Processing Time by Station**
   - Average processing time per station
   - Operation count
   - Station-wise comparison chart

4. **Wagon Movement Graph**
   - Station-to-station movement over time
   - Load tracking
   - Time-based visualization
   - Movement statistics

5. **Dip Compliance Monitoring**
   - Compliance rate percentage
   - Pass/fail counts
   - Failed dips detail view
   - Dip time heatmap (Load vs Station)

## Installation

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Setup Steps

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify installation:**
   ```bash
   streamlit --version
   ```

## Running the Application

### Start the GUI:
```bash
streamlit run streamlit_app.py
```

The application will open in your default browser at `http://localhost:8501`

### Alternative (if port 8501 is busy):
```bash
streamlit run streamlit_app.py --server.port 8502
```

## Input File Formats

### input_tanks_csv.csv
Required columns:
| Column | Description |
|--------|-------------|
| station_no | Station number (integer) |
| process_name | Name of the station/process |
| distance_mm | Position of station along track |
| dip_time_sec | Target dip time |
| max_dip_time_sec | Maximum acceptable dip time |
| station_type | Type of station |
| Process_NO | Process number (for identifying alternating tanks) |

### input_wagon_new.csv
Required columns:
| Column | Description |
|--------|-------------|
| Fast Speed Mtrs/Min | Wagon speed when loaded (meters/min) |
| Superfast SpeedMtrs/Min | Wagon speed when empty (meters/min) |
| Lift Time Seconds | Time to lift load |
| Lower Time Seconds | Time to lower load |

## Output Files

### Generated Outputs (downloadable from GUI)

1. **OUTPUT_sequence.csv**
   - Contains the generated instruction sequence
   - Columns: PROJECT ID, Program ID, Wagon Number, Instruction, Instruction Sr No, Instruction Value, LOAD_NO, ACCUMULATED TIME

2. **DIP_TIME_OUTPUT.csv**
   - Audit log of all dip operations
   - Columns: Load ID, Station No, Assigned Tank, Entry Time, Exit Time, Target Dip, Actual Dip, Status

## Usage Workflow

1. **Configuration Tab:**
   - Select and upload your three CSV input files
   - Adjust simulation parameters as needed
   - Click "Generate Sequence" button
   - Wait for confirmation message

2. **Execution Tab:**
   - Review the generated instruction sequence
   - Check total cycle time
   - Download sequence if needed

3. **Analysis Tab:**
   - Examine dip times for compliance
   - Analyze cycle and processing times
   - View wagon movement patterns
   - Check compliance metrics
   - Export data for further analysis

## Performance Notes

- Handles up to 100 loads per simulation
- Real-time calculation and visualization
- Typical generation time: 1-5 seconds depending on load complexity
- Session state is maintained during the browser session

## Troubleshooting

### Port Already in Use
```bash
streamlit run streamlit_app.py --server.port 8502
```

### File Upload Issues
- Ensure CSV files are properly formatted
- Check column names match expected format (case-sensitive)
- Verify no special characters in file paths

### Performance Issues
- Reduce total loads if simulation is slow
- Close other applications
- Refresh the browser page

## Architecture

The application uses:
- **Streamlit**: Web framework and UI
- **Pandas**: Data processing and analysis
- **Plotly**: Interactive visualizations
- **Session State**: Client-side data persistence

## File Structure
```
d:\STREAMLIT_PLANT290426\
├── streamlit_app.py          # Main GUI application
├── scheduler.py              # Original scheduler (reference)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── input_tanks_csv.csv       # Sample tanks data
├── input_wagon_new.csv       # Sample wagon data
├── input_Crosstrolley.csv    # Sample crosstrolley data
├── OUTPUT_sequence.csv       # Sample output (generated)
└── DIP_TIME_OUTPUT.csv       # Sample output (generated)
```

## Tips for Best Results

1. **Data Quality:**
   - Ensure all numeric fields contain valid numbers
   - Avoid NaN or empty values in critical columns
   - Use consistent decimal separators

2. **Simulation Parameters:**
   - Start with 5-10 loads for testing
   - Increase gradually to test larger batches
   - Monitor compliance rates

3. **Analysis:**
   - Use heatmap to identify problematic station combinations
   - Check wagon movement graph for bottlenecks
   - Review failed dips for corrective actions

## Support

For issues or questions:
1. Check the Help section in the sidebar
2. Verify input file formats
3. Review the error messages in the application

## License

Internal use only - Electroplating Plant Scheduling System

## Version
1.0 - April 2026
