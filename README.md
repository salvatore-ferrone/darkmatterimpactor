# Dark Matter Impactor: Prediction Pipeline for Subhalo Gaps in Stellar Streams

A modular Python pipeline for predicting dark matter subhalo-induced gaps in cold globular cluster stellar streams.

## Overview

This project implements a hierarchical prediction workflow for dark matter subhalo gaps in Milky Way stellar streams, structured around four data levels:

- **Level 0**: Input data (MW model, streams, subhalo population)
- **Level 1**: Computed orbits and intersections
- **Level 2**: Stream simulations with gaps
- **Level 3**: Analysis and predictions

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/darkmatterimpactor.git
cd darkmatterimpactor
```

### 2. Create Conda Environment
```bash
conda env create -f environment.yml
conda activate darkmatter-streams
```

### 3. Install tstrippy (Custom Package)
tstrippy is a custom tidal stripping package that needs to be built from source. Clone and build it:

```bash
# Clone tstrippy (replace with actual repo URL)
git clone https://github.com/salvatore-ferrone/tstrippy.git
cd tstrippy

# Build the package
meson setup builddir
meson compile -C builddir/
meson install -C builddir/

# Return to main project
cd ../darkmatterimpactor
```

### 4. Verify Installation
```bash
python -c "import tstrippy; import galpy; import astropy; print('All dependencies installed successfully')"
```

## Project Structure

```
darkmatterimpactor/
├── code/
│   ├── mw_model.py          # Milky Way potential model
│   ├── streams.py           # Globular cluster streams
│   ├── subhalo_population.py # DM subhalo population
│   ├── gap_prediction.py    # Gap prediction algorithms
│   └── analysis.py          # Analysis and visualization
├── inputdata/               # Input datasets
├── outputdata/              # Computed results
├── plots/                   # Generated plots
├── frames/                  # Animation frames
├── videos/                  # Rendered animations
└── environment.yml          # Conda environment specification
```

## Usage

### Basic Workflow
```python
from code.mw_model import MilkyWayModel
from code.streams import StreamGenerator
from code.gap_prediction import GapPredictor

# Initialize MW model
mw = MilkyWayModel()

# Generate stream orbits
streams = StreamGenerator(mw_potential=mw.potential)

# Predict gaps
predictor = GapPredictor(mw_model=mw, streams=streams)
gaps = predictor.predict_gaps(subhalo_population)
```

### Running Tests
```bash
# Test MW model against existing data
python code/test_mw_model.py

# Run full pipeline
python code/run_pipeline.py
```

## Dependencies

- **tstrippy**: Custom tidal stripping package (built from source)
- **galpy**: Galactic dynamics library
- **astropy**: Astronomical utilities and units
- **numpy/scipy**: Numerical computing
- **matplotlib/plotly**: Visualization
- **h5py**: HDF5 data I/O

## Development

### Code Style
This project uses:
- **black** for code formatting
- **flake8** for linting

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make changes with proper tests
4. Submit a pull request

## License

[Add your license here]

## Citation

[Add citation information when ready]