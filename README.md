# SMS Text Classification

## Overview
This project focuses on SMS Text Classification, where the goal is to classify SMS messages into predefined categories such as spam, ham (not spam), or other relevant labels. This project demonstrates the application of Natural Language Processing (NLP) and machine learning techniques to handle text data effectively.

## Features
- Preprocessing SMS messages (tokenization, stemming, removing stop words, etc.)
- Feature extraction using techniques like TF-IDF or Bag of Words (BoW)
- Implementation of machine learning algorithms (e.g., Naive Bayes, Support Vector Machines, Logistic Regression)
- Performance evaluation using metrics like accuracy, precision, recall, and F1-score
- Visualization of results and feature importance

## Requirements
To run this project, ensure you have the following installed:

- Python 3.8+
- Libraries:
  - pandas
  - numpy
  - scikit-learn
  - nltk
  - matplotlib (optional for visualization)
  - seaborn (optional for visualization)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Apratim23/SMS_Text.git
   cd sms-text-classification
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset
The project uses a labeled dataset of SMS messages, typically containing categories like `spam` and `ham`.

### Example Datasets
- [UCI SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)

Place the dataset in the `data` folder and ensure the structure aligns with the project's expectations (e.g., a CSV file with `label` and `message` columns).

## Usage

1. Preprocess the dataset:
   ```bash
   python preprocess.py
   ```

2. Train the model:
   ```bash
   python train.py
   ```

3. Evaluate the model:
   ```bash
   python evaluate.py
   ```

4. Make predictions on new SMS messages:
   ```bash
   python predict.py --message "Your free gift awaits!"
   ```


Visualizations and insights are stored in the `notebooks` folder and can be reproduced using the Jupyter Notebook.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your improvements.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
