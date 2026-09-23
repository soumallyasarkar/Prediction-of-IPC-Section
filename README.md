# Prediction of IPC Sections

A command-line legal text-matching tool that predicts relevant sections of the Indian Penal Code (IPC) from a plain-language crime description.

The project supports English and several Indian languages. For multilingual input, it detects the language with fastText, translates the query to English with NLLB-200, searches the IPC dataset, and translates the matching results back into the input language.

## Features

- Hybrid IPC search using semantic embeddings and TF-IDF keyword similarity.
- Severity calibration that compares the estimated crime severity with the punishment severity.
- Top-three section recommendations with offense, punishment, cognizability, bail, and court details.
- Language detection and translation for English, Hindi, Telugu, Tamil, Malayalam, Kannada, Marathi, Bengali, Gujarati, Punjabi, Odia, and Urdu.
- Cached embeddings and TF-IDF vectors to make later runs faster.

## Project Structure

| File | Purpose |
| --- | --- |
| `translator.py` | Multilingual CLI entry point. |
| `ipc.py` | English IPC search CLI and matching implementation. |
| `merged.json` | IPC section data used by the search engine. |
| `download_lid176.py` | Downloads the fastText language-identification model. |
| `requirements.txt` | Python dependencies. |

## Requirements

- Python 3.8 or newer
- Internet access on the first run to download Hugging Face models
- Enough disk space for the translation and embedding models

## Installation

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies and the spaCy English model:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Download the fastText language detector if `lid.176.bin` is not already present:

```powershell
python download_lid176.py
```

## Usage

### Multilingual assistant

Use this mode for English or one of the supported Indian languages:

```powershell
python translator.py
```

Enter a crime description when prompted. Type `quit`, `exit`, or `q` to stop.

Example:

```text
Enter your legal query: Someone threatened me with a knife and demanded money
```

### English-only IPC search

Use the underlying search engine directly:

```powershell
python ipc.py
```

This mode displays additional similarity and severity information for each result.

## How It Works

1. IPC records from `merged.json` are combined into searchable text.
2. `all-mpnet-base-v2` creates semantic embeddings for each IPC record.
3. A TF-IDF vectorizer measures keyword overlap, including unigrams and bigrams.
4. The scores are combined with a 60% semantic and 40% keyword weighting.
5. A rule-based severity estimate adjusts results whose punishment severity differs substantially from the query.
6. The highest-ranked sections are displayed.

On the first run, the project creates `ipc_embeddings_all-mpnet-base-v2.pkl` and `ipc_tfidf.pkl`. These files are reused on later runs. Delete them if the dataset or search configuration changes and you want to rebuild the caches.

## Configuration

Search settings are defined near the top of `ipc.py`:

- `MODEL_NAME`: sentence-transformer model used for embeddings.
- `SIMILARITY_THRESHOLD`: minimum result score.
- `HYBRID_WEIGHT_EMBEDDING`: semantic score weight.
- `HYBRID_WEIGHT_KEYWORD`: TF-IDF score weight.

The two hybrid weights should normally add up to `1.0`.

## Limitations

- This is a retrieval and ranking tool, not a legal advisor or a substitute for professional legal counsel.
- Predictions depend on the quality and coverage of `merged.json`; verify every result against current law and the facts of the case.
- The project uses IPC data. Indian criminal law has changed over time, so users should confirm whether the relevant offence is governed by current legislation.
- Language detection and translation can be inaccurate for short, ambiguous, or mixed-language input.
- The first model download and embedding generation may take several minutes and require significant memory.
# Future Plans

A dashboard is planned as the next major extension of this project. The dashboard will provide a user-friendly interface for:

- Entering crime descriptions without using the command line.
- Viewing ranked IPC section matches, confidence scores, punishment details, and severity comparisons.
- Submitting queries in supported Indian languages and viewing translated results.
- Exploring IPC sections and filtering results by offense, punishment, bailability, cognizability, and court.
- Displaying clear warnings that results are informational and should be verified by a qualified legal professional.

The dashboard may be implemented as a web application backed by the existing Python search and translation modules. The current command-line interface will remain useful for development, testing, and batch experiments.

## License

See [LICENSE](LICENSE) for the project license.

## Author

**Soumallya Sarkar**  
Undergraduate B.Tech student in Computer Science and Engineering (CSE)  
Indian Institute of Information Technology, Kottayam
