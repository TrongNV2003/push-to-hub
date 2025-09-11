# Hugging Face Model & Dataset Upload Tools
This repository contains Python scripts to easily upload your trained models and datasets to Hugging Face Hub.

## Features

- 📤 Upload local models to Hugging Face Hub
- 📊 Upload local datasets to Hugging Face Hub
- 🔐 Support for private and public repositories
- 🚀 Simple and straightforward Python scripts
- 📋 Automatic conversion to standard formats (Parquet for datasets)

## Prerequisites

Before using these tools, ensure you have:

1. **Python environment** with required packages installed
2. **Hugging Face account** and authentication token
3. **Local model or dataset** ready for upload

## Installation

Install the required packages:

```bash
pip install huggingface_hub datasets transformers
```

## Authentication

Before uploading, authenticate with Hugging Face:

```bash
huggingface-cli login
```

Enter your Hugging Face token when prompted.

## Usage

### 1. Upload Models (`upload_model.py`)

This script uploads a locally saved model and tokenizer to Hugging Face Hub.

#### Configuration

Edit the following variables in `upload_model.py`:

```python
LOCAL_MODEL_DIR = "./your-local-model-directory"  # Path to your saved model
REPO_ID = "your-username/model-name"              # Hub repository ID
```

#### Run

```bash
python upload_model.py
```

### 2. Upload Datasets (`upload_dataset.py`)

This script uploads local JSONL files to Hugging Face Hub as a dataset.

#### Expected Data Structure

Your local dataset directory should contain:

```
your-dataset-dir/
├── train.jsonl
├── validation.jsonl
└── test.jsonl
```

#### Configuration

Edit the following variables in `upload_dataset.py`:

```python
LOCAL_DATASET_DIR = "your-local-dataset-directory"  # Path to your dataset folder
REPO_ID = "your-username/dataset-name"              # Hub repository ID
```

#### Run

```bash
python upload_dataset.py
```

The dataset will be automatically converted to Parquet format on Hugging Face Hub:

```
data/
├── train-00000-of-00001.parquet
├── validation-00000-of-00001.parquet
└── test-00000-of-00001.parquet
```

## Configuration Options

### Privacy Settings

Both scripts support private and public repositories:

```python
private=True   # Private repository (default)
private=False  # Public repository
```

### Custom Data Files

For datasets with different file names or formats, modify the `data_files` dictionary in `upload_dataset.py`:

```python
data_files = {
    "train": os.path.join(local_data_dir, "custom_train.jsonl"),
    "validation": os.path.join(local_data_dir, "custom_val.jsonl"),
    "test": os.path.join(local_data_dir, "custom_test.jsonl")
}
```

## Troubleshooting

### Common Issues

1. **Authentication Error**
   ```
   Solution: Run `huggingface-cli login` and enter your token
   ```

2. **Permission Denied**
   ```
   Solution: Ensure you have write access to the repository
   ```

3. **File Not Found**
   ```
   Solution: Check your local file paths and ensure files exist
   ```

4. **Model Loading Error**
   ```
   Solution: Verify your model directory contains all required files:
   - config.json
   - pytorch_model.bin (or model.safetensors)
   - tokenizer files
   ```

### Debug Tips

- Check file paths are correct and absolute
- Ensure your local model was saved properly using `trainer.save_model()` or `model.save_pretrained()`
- Verify JSONL files are properly formatted
- Check internet connection and Hugging Face Hub status

## File Structure

```
push-to-hub/
├── README.md           # This documentation
├── upload_model.py     # Model upload script
└── upload_dataset.py   # Dataset upload script
```

## Example Workflow

1. **Train your model** using Transformers Trainer or custom training loop
2. **Save model locally**:
   ```python
   trainer.save_model("./my-model")
   # or
   model.save_pretrained("./my-model")
   tokenizer.save_pretrained("./my-model")
   ```
3. **Prepare dataset** in JSONL format
4. **Configure scripts** with your paths and repository names
5. **Authenticate** with Hugging Face
6. **Run upload scripts**

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is open source. Feel free to use and modify as needed.

---

**Note**: Always test with private repositories first before making your models/datasets public. 