# Cross-Lingual-Sentiment-Analysis

This project involves fine-tuning pre-trained multilingual transformer models to classify text data. The pipeline includes data preprocessing, model training, evaluation, and visualization of performance metrics.

## Project Overview

### Key Features:
- **Multi-level Data Categorization:**
  - The code organizes files into `train`, `dev`, and `test` datasets, with further granularity into `sentence`- and `text`-level categories for test data.

- **Data Processing and Tokenization:**
  - Text data is processed into tokenized datasets using a pre-trained tokenizer for transformer models.
  - Labels are mapped to integer values for compatibility with the model.

- **Model Fine-Tuning:**
  - Pre-trained transformer models (e.g., BERT) are fine-tuned using the `Trainer` API from Hugging Face Transformers.
  - Training employs features like gradient accumulation, mixed precision (FP16), and early stopping.

- **Evaluation:**
  - Models are evaluated on test datasets with metrics such as F1 Score and Accuracy.

- **Visualization:**
  - Performance metrics are visualized through bar plots, comparing models across different levels (sentence/text) and languages.
 

### Code Breakdown

- **Data Processing**  
 - *categorize_files(directory):*
  - Categorizes files into train, dev, and test datasets.
  - Organizes test datasets by granularity (sentence/text) and language.  
 - *process_data(filepaths, tokenizer, label_map):*
  - Reads files, tokenizes text, and maps labels to integers.
  - Processes data in batches to reduce memory overhead.
  - Converts the data to Hugging Face Dataset format.
  
- **Model Fine-Tuning**

- *fine_tune_model(model_name, train_dataset, dev_dataset):*


  - Fine-tunes a pre-trained transformer model on the training and validation datasets.
  - Utilizes Hugging Face's Trainer API with features like gradient accumulation and FP16 precision.
  
- **Evaluation**

 - evaluate_model(model_name, model, test_data):
  - Evaluates the fine-tuned model on test datasets.
  - Computes metrics like F1 Score and Accuracy using sklearn.
- **Visualization**

 - plot_overall_performance(results):
  - Compares overall performance (F1 Score and Accuracy) of different models.
  
 - plot_multilevel_performance(results):
  - Visualizes performance differences across sentence and text levels.

 - plot_language_performance(results):
  - Highlights model performance for different languages in the dataset.
