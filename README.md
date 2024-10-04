# 🎬 IMDb Dataset Sentiment Analysis

This project focuses on performing **sentiment analysis** on movie reviews from the IMDb dataset, as part of the **Artificial Intelligence coursework**. We utilize three popular machine learning algorithms (Naive Bayes, Random Forest & Logistic Regression) to classify reviews as **positive** or **negative**, and compare the predicted results with the actual labels from the dataset.

## Algorithms Implemented

1. **Naive Bayes Classifier**: A probabilistic classifier based on applying Bayes' Theorem.
2. **Random Forest**: An ensemble learning method that constructs multiple decision trees (using ID3 algorithm) to improve prediction accuracy.
3. **Logistic Regression**: A binary classification algorithm using stochastic gradient ascent, enhanced with feature normalization.

## Dataset

We use the **IMDb Sentiment Dataset** provided by Stanford University, which contains **50,000 movie reviews** split evenly between **positive** and **negative** sentiments. You can find the dataset [here](https://ai.stanford.edu/~amaas/data/sentiment/).

## Project Workflow

1. **Preprocessing**:
   - We utilize the **Keras API** to extract and preprocess the reviews and labels.
   - Each review is transformed into a **binary vector** based on the presence or absence of certain words in the vocabulary.
   
2. **Binary Representation**:
   - **x_train_binary** and **x_test_binary**: Each review is represented as a sequence of **1000 binary digits**, where `1` indicates the presence of a word in the vocabulary, and `0` indicates its absence.
   - **y_train_list** and **y_test_list**: These contain the actual classification of each review (1 for **positive**, 0 for **negative**).

3. **Parameter Configuration**:
   - **n = 50**: The most frequent 50 words are omitted from consideration.
   - **m = 1000**: The remaining 1000 frequent words make up the vocabulary used in the algorithms.
   - **k = 38,839**: Rare words (appearing only once) are excluded.
   - We set the default number of trees in the **Random Forest** to **9**.

## Sentiment Classification Process

1. Each review is classified as either **positive** (<span style="color:green;">positive</span>) or **negative** (<span style="color:red;">negative</span>) based on its word content using our chosen algorithms.
2. The predicted classifications are then compared with the actual labels from the dataset.
3. Finally, the **success rate** (accuracy) of each algorithm is calculated.

## Results and Analysis

For each algorithm, we measure the performance based on accuracy, precision, recall, and F1-score. We aim to determine which algorithm performs best on this dataset and under the given conditions.

## Tools and Libraries

- **Keras API (TensorFlow)**: For dataset preprocessing and vector representation.
- **Python**: For implementing the machine learning algorithms.
- **Pandas & NumPy**: For data manipulation and handling.
- **Scikit-learn**: For calculating performance metrics and Random Forest implementation

## Contributors
- **Odysseas Spyropoulos**
- **Lydia Christina Wallace**
- **Miltiadis Tsolkas**