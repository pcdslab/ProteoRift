# ProteoRift Python Package

End-to-end machine learning pipeline for peptide database search in mass spectrometry proteomics.


## Installation

```bash
pip install proteorift
```

## Quick Start

### Using Sample Data

```python
from proteorift import ProteoRiftSearch

# Initialize and run with sample data
searcher = ProteoRiftSearch()
results = searcher.search_with_sample_data()

print(f"Results saved to: {results['output_dir']}")
```

### Using Your Own Data

```python
from proteorift import ProteoRiftSearch

# Initialize search
searcher = ProteoRiftSearch()

# Run peptide database search
results = searcher.search(
    mgf_dir="path/to/your/spectra",      # Directory with MGF files
    peptide_db="path/to/your/database",  # Directory with FASTA files
    output_dir="./results"
)
```

### Custom Parameters

```python
searcher = ProteoRiftSearch(
    precursor_tolerance=10,
    precursor_tolerance_type="ppm",
    charge=3,
    length_filter=True,
    device="cuda"  # or "cpu", "auto"
)

results = searcher.search(mgf_dir="...", peptide_db="...")
```

## Command Line Interface

```bash
# Run search with sample data
proteorift search-sample --output-dir ./results

# Run search with your data
proteorift search \
    --mgf-dir path/to/spectra \
    --peptide-db path/to/database \
    --output-dir ./results \
    --tolerance 10 \
    --charge 3

# Download models only
proteorift download-models
```

## Output

ProteoRift generates Percolator-compatible PIN files:
- `target.pin` - Target peptide-spectrum matches
- `decoy.pin` - Decoy peptide-spectrum matches


## System Requirements

- **Python:** 3.8 or higher
- **GPU:** 12GB+ VRAM recommended (CPU also supported)
- **OS:** Linux, macOS, or Windows

