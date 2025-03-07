# Bach Chorale Generator

This project implements a deep learning model to generate Bach-style chorales using LSTM neural networks. It explores how different data augmentation techniques affect the model's performance and creativity.

## Dataset

The dataset consists of chorales composed by Johann Sebastian Bach from this Kaggle dataset https://www.kaggle.com/datasets/pranjalsriv/bach-chorales-2/data, represented as piano note indices:
- Each chorale is 100 to 640 time steps long
- Each time step contains 4 integers representing the notes played by the four voices (soprano, alto, tenor, bass)
- The dataset includes 382 chorales (229 training, 76 validation, 77 test)

## Features

- **Data Exploration**: Analyze the Bach chorales dataset to gain insights into the musical structures
- **Data Augmentation**: Implement four augmentation strategies:
  1. The original dataset (no augmentation)
  2. All chorales transposed to C major/A minor
  3. Chorales in C, G, and F major with tempo variations
  4. Chorales transposed to all 12 keys
- **LSTM Model**: Train a sequence-to-sequence LSTM model to predict the next chord
- **Chorale Generation**: Generate new Bach-style chorales from trained models
- **MIDI Conversion**: Convert generated chorales to MIDI files for playback

## Project Structure

```
bach-chorale-generator/
├── README.md                 # Project overview, setup instructions, results
├── notebooks/                # Exploratory analysis and visualization
│   ├── 01_data_exploration.ipynb
│   └── 02_model_training.ipynb
├── src/                      # Core Python modules
│   ├── data/                 # Data processing
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   └── augmentation.py
│   ├── models/               # Model architectures
│   │   ├── __init__.py
│   │   └── lstm_model.py
│   └── utils/                # Helper functions
│       ├── __init__.py
│       └── midi_conversion.py
├── scripts/                  # Training/inference scripts
│   ├── train.py
│   └── generate.py
├── tests/                    # Unit tests
└── requirements.txt          # Dependencies
```

## Getting Started

### Prerequisites

- Python 3.8+
- TensorFlow 2.6+
- Other dependencies listed in `requirements.txt`

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/bach-chorale-generator.git
   cd bach-chorale-generator
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the package:
   ```
   # For development (editable mode)
   pip install -e .
   
   # OR for regular installation
   pip install .
   ```

4. Download the Bach chorales dataset and place it in a directory with the following structure:
   ```
   data/bach_chorales/
   ├── train/         # 229 CSV files
   ├── validation/    # 76 CSV files
   └── test/          # 77 CSV files
   ```

### Usage

#### Data Exploration

Explore the dataset using the provided Jupyter notebook:
```
jupyter notebook notebooks/01_data_exploration.ipynb
```

#### Training a Model

Train a model with one of the augmentation strategies:
```
# Using the script directly
python scripts/train.py --data_dir data/bach_chorales --augmentation_version 3 --epochs 50

# OR using the installed command-line tool
train-bach --data_dir data/bach_chorales --augmentation_version 3 --epochs 50
```

#### Generating Chorales

Generate new chorales using a trained model:
```
# Using the script directly
python scripts/generate.py --model_path outputs/bach_lstm_v3_TIMESTAMP/best_model.h5 --data_dir data/bach_chorales --num_chorales 5

# OR using the installed command-line tool
generate-bach --model_path outputs/bach_lstm_v3_TIMESTAMP/best_model.h5 --data_dir data/bach_chorales --num_chorales 5
```

## Experimental Results

The project compares the performance of models trained on different augmentation strategies:

| Augmentation Version | Description | Training Size | Validation Loss | Note Accuracy |
|----------------------|-------------|---------------|-----------------|---------------|
| 1 | Original dataset | 229 chorales | - | - |
| 2 | C major/A minor | 229 chorales | - | - |
| 3 | C, G, F + tempo | 1,374 chorales | - | - |
| 4 | All 12 keys | 2,748 chorales | - | - |

*Note: Fill in the results after training the models*
