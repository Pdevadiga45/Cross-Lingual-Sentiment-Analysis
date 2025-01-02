# Cross Lingual Sentiment Analysis

MultiLingual Dataset Used: MultiEmo, a new benchmark dataset for multilingual sentiment analysis tasks, includes 11 languages: `Polish`, `English`, `Chinese`, `Italian`, `Japanese`, `Russian`, `German`, `Spanish`, `French`, `Dutch`, and `Portuguese`. The collection contains consumer reviews from four domains: medicine, hotels, products, and universities.
(https://clarin-pl.eu/dspace/handle/11321/798)

### Code Breakdown

- **Data Processing**  
  - `categorize_files(directory):`
    - Categorizes files into train, dev, and test datasets.
    - Organizes test datasets by granularity (sentence/text) and language.

  - `process_data(filepaths, tokenizer, label_map):`
    - Reads files, tokenizes text, and maps labels to integers.
    - Processes data in batches to reduce memory overhead.
    - Converts the data to Hugging Face Dataset format.

- **Model Fine-Tuning**  
  - `fine_tune_model(model_name, train_dataset, dev_dataset):`
    - Fine-tunes a pre-trained transformer model on the training and validation datasets.
    - Utilizes Hugging Face's Trainer API with features like gradient accumulation and FP16 precision.

- **Evaluation**  
  - `evaluate_model(model_name, model, test_data):`
    - Evaluates the fine-tuned model on test datasets.
    - Computes metrics like F1 Score and Accuracy using sklearn.

- **Visualization**  
  - `plot_overall_performance(results):`
    - Compares overall performance (F1 Score and Accuracy) of different models.

  - `plot_multilevel_performance(results):`
    - Visualizes performance differences across sentence and text levels.

  - `plot_language_performance(results):`
    - Highlights model performance for different languages in the dataset.
