# Find Raman/SERS Datasets

Search for and collect datasets for machine learning related to Raman spectroscopy and SERS.

## Usage

Provide a search topic as the argument, or leave blank for the default Raman/SERS search.

## Instructions

Run the dataset search agent:

```bash
cd /Users/dtsyplyackov/Documents/data_for_ml/data_for_ml && python -m agent $ARGUMENTS
```

If dependencies are not installed, first run:

```bash
pip install -r requirements.txt
```

The agent will:
1. Search Kaggle, HuggingFace, and the web for datasets
2. Present found datasets in a formatted table
3. Ask which datasets to download (human-in-the-loop)
4. Download selected datasets to `./downloads/`

After the agent finishes, review the downloaded data and suggest preprocessing steps.
