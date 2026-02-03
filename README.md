# Algospeak Detection Project

This project implements a comprehensive algospeak (intentional coded/obfuscated language) detection system for social media posts. It combines rule-based pattern matching with advanced LLM labeling to identify and categorize algospeak content.

## Features

- **Rule-based Detection**: Fast initial screening using dictionary matching and regex patterns
- **LLM Labeling**: Advanced classification using OpenAI GPT models with structured outputs
- **Data Pipeline**: Complete workflow from raw posts to labeled datasets
- **Export Tools**: Convert results to CSV for analysis in spreadsheets
- **Analysis Scripts**: Extract and examine detected algospeak patterns

## Project Structure

```
algospeak_data/
├── generated_data/          # Output files (auto-created, gitignored)
├── algospeak_dictionary.json # Dictionary of algospeak terms
├── posts.txt               # Input post data
├── posts_10000.txt         # Larger input dataset
├── pyproject.toml          # Project dependencies
├── uv.lock                 # Lockfile for reproducible installs
├── .python-version         # Python version specification
├── .env                    # Environment variables (API keys)
├── .gitignore             # Git ignore rules
└── *.py                   # Python scripts
```

## Setup Instructions

### 1. Clone and Install Dependencies

```bash
# Clone the repository
git clone <your-repo-url>
cd algospeak_data

# Install uv package manager (if not already installed)
# On Windows: winget install astral-sh.uv
# Or download from: https://github.com/astral-sh/uv

# Install dependencies
uv sync
```

### 2. Set Up Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys).

### 3. Verify Installation

```bash
# Activate virtual environment
uv run python --version

# Test basic import
uv run python -c "import openai, pydantic; print('Dependencies OK')"
```

## Usage

All commands should be run from the project root directory.

### Data Collection

Collect posts from Bluesky:

```bash
uv run python test_pull.py
```

This creates `posts_10000.txt` with English posts.

### Rule-Based Labeling

Apply initial rule-based detection:

```bash
uv run python data_processing.py
```

Outputs:
- `generated_data/clean_posts.txt`
- `generated_data/questionable_posts.txt`
- `generated_data/moderated_posts.txt`
- `generated_data/labeled_posts.jsonl`

### LLM Labeling

Apply advanced LLM classification:

```bash
# Basic run
uv run python llm_label_posts.py --input posts_10000.txt --model gpt-5-nano

# Limited run for testing
uv run python llm_label_posts.py --input posts.txt --model gpt-4o-mini --limit 100

# Full production run
uv run python llm_label_posts.py --input posts_10000.txt --model gpt-5-nano --limit 500 --overwrite
```

Outputs:
- `generated_data/llm_labeled_posts.jsonl`
- `generated_data/clean_posts.txt`
- `generated_data/algospeak_yes.txt`
- `generated_data/needs_manual_review.txt`
- `generated_data/error_posts.txt`

### Analysis and Export

Extract algospeak posts for analysis:

```bash
uv run python extract_algospeak_posts.py
```

Convert to CSV for spreadsheet analysis:

```bash
uv run python jsonl_to_csv.py
```

Filter for algospeak detections:

```bash
uv run python inspect_algospeek.py
```

## Script Reference

### Core Scripts

- **`data_processing.py`**: Rule-based algospeak and moderation detection
- **`llm_label_posts.py`**: LLM-powered detailed labeling with OpenAI API
- **`test_pull.py`**: Collect posts from Bluesky firehose

### Analysis Scripts

- **`extract_algospeak_posts.py`**: Extract and display LLM-labeled algospeak posts
- **`jsonl_to_csv.py`**: Convert JSONL results to CSV format
- **`inspect_algospeek.py`**: Filter rule-based results for algospeak hits

### Utility Scripts

- **`main.py`**: Project entry point (placeholder)

## Data Flow

```
Raw Posts → Rule-based Labeling → LLM Labeling → Analysis/Export
     ↓             ↓                    ↓             ↓
posts.txt → labeled_posts.jsonl → llm_labeled_posts.jsonl → CSV/Reports
```

## Configuration

### Models

- **gpt-5-nano**: Fast, cost-effective primary model
- **gpt-4o-mini**: Fallback model for reliability

### File Paths

All output files are automatically created in `generated_data/` directory. Input files should be placed in the project root.

### Environment

- Python 3.12+
- Dependencies managed by uv
- Virtual environment auto-created by uv

## Troubleshooting

### Common Issues

1. **OpenAI API Errors**: Check `.env` file and API key validity
2. **Missing Dependencies**: Run `uv sync` to install packages
3. **Path Errors**: Ensure running commands from project root
4. **Rate Limits**: LLM script includes automatic retries and fallbacks

### Generated Data

The `generated_data/` folder contains all output files and is gitignored. Files are created automatically when scripts run.

## Contributing

1. Follow the existing code style
2. Add docstrings to new functions
3. Update this README for new features
4. Test scripts before committing

## License

[Add your license information here]