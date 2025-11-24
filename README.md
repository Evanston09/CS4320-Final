# CS4320 Final: Speaker Identification

Speaker identification system using logistic regression implemented from scratch without ML-specific libraries (e.g., scikit-learn).

## Features

- **Custom Logistic Regression**: Implementation without scikit-learn dependencies
- **Voice Activity Detection**: Uses Silero VAD for audio preprocessing
- **Interactive TUI**: Record samples, train models, and test speaker identification
- **Audio Processing**: MFCC feature extraction for voice analysis

## Quick Start

1. Install dependencies:
   ```bash
  uv run main.py 
   ```

2. Menu options:
   - **[M]** Modify speakers - Add or manage speaker profiles
   - **[T]** Train Model - Train on recorded samples
   - **[R]** Run Model - Test speaker identification
   - **[Q]** Quit

## Project Structure

- `main.py` - Interactive TUI for training and testing
- `utils.py` - Model training and evaluation logic
- `audio_processing.py` - Audio feature extraction
- `predict_speaker.py` - CLI tool for speaker prediction on audio files
- `kfold_validation.py` - Stratified K-fold cross-validation script
- `hyperparameter_tuning.py` - Grid search for optimal hyperparameters
- `clean_processed_data.py` - Utility to clean processed audio data
- `data/speakers/` - Speaker audio samples

## Author

Evan Kim
