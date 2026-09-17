# Agent Assist – Intent Classification & Tracking

A customer-service conversation analysis project that uses NLP and machine learning to identify customer intents and track how those intents change during a conversation.

The project was built around a dataset of customer-service call transcripts and includes data cleaning, intent discovery, text classification, model evaluation, and an intent-tracking component for use in an agent-assist application.

## What the project does

The project takes customer-service conversations and:

1. Cleans and preprocesses the conversation text
2. Removes common call-centre noise and irrelevant phrases
3. Segments conversations into meaningful sections
4. Uses sentence embeddings to explore conversation clusters
5. Maps clusters to potential customer intents
6. Creates a labelled intent dataset
7. Trains a Logistic Regression classifier
8. Evaluates the classifier on unseen test data
9. Tracks intent and confidence throughout a conversation
10. Provides the results through a dashboard

## Dataset

The original dataset contains **5,459 customer conversations**.

After cleaning and removing empty or very short conversations:

* 5,431 conversations contained usable text
* 4,314 conversations remained after applying a 20-word minimum
* 3,191 conversations were used in the final labelled dataset
* 17 intent types were identified

The intent categories include examples such as:

* Warranty claims
* Credit enquiries
* Invoice enquiries
* Insurance enquiries
* Lodge claim
* Billing enquiries
* Claim updates
* Complaints
* Refunds
* Service enquiries
* Job-order enquiries
* Cancellation

## NLP Processing

The conversation data is cleaned before modelling.

The preprocessing includes:

* Lowercasing
* Removing numbers and punctuation
* Removing English stopwords
* Removing common call-centre phrases and noise
* Lemmatization using spaCy
* Removing very short conversations

Some domain-specific noise was also removed, including common terms such as `agent`, `calling`, `customer`, `service`, `phone`, and `help`.

## Intent Discovery

Before training the classifier, the project explores the conversations using sentence embeddings.

The `all-mpnet-base-v2` Sentence Transformer model is used to create embeddings for meaningful conversation segments.

UMAP is then used to reduce the embedding dimensions, followed by Agglomerative Clustering to identify groups of similar conversations.

Different cluster counts were tested from 5 to 10. Six clusters produced the highest silhouette score of **0.292** in this experiment.

The resulting clusters were then inspected using common words and manually mapped to potential business intents.

## Intent Classification

After the intent labels were created, the conversations were split into:

* **70% training data:** 2,233 conversations
* **20% validation data:** 638 conversations
* **10% test data:** 320 conversations

TF-IDF was used to convert the conversation text into features.

The vectorizer was configured with:

* Maximum of 5,000 features
* Unigrams, bigrams and trigrams
* Minimum document frequency of 3
* Maximum document frequency of 0.8
* Sublinear TF scaling

A **Logistic Regression** model was then trained using class balancing and GridSearchCV to tune the model parameters.

The best configuration used:

```text
C = 2.0
solver = liblinear
max_iter = 1000
```

The model was saved as `intent_model.pkl`, while the fitted TF-IDF vectorizer was saved as `vectorizer.pkl`.

## Model Results

The classifier achieved:

**Test accuracy: 76.2%**

On the 320-sample test set:

| Metric             | Score |
| ------------------ | ----: |
| Accuracy           | 0.762 |
| Weighted Precision |  0.75 |
| Weighted Recall    |  0.76 |
| Weighted F1        |  0.75 |
| Macro F1           |  0.51 |

Performance varied considerably between intent classes, partly because some intents had very small numbers of test examples.

The stronger-performing classes included:

* `credit_enquiry` – 0.89 accuracy
* `billing_enquiry` – 0.89 accuracy
* `warranty_claim` – 0.84 accuracy
* `invoice_enquiry` – 0.79 accuracy
* `Lodge_claim` – 0.78 accuracy

This also highlighted areas for improvement, particularly for less frequent intents.

## Conversation Intent Tracking

The project goes beyond classifying individual conversations by including a `ConversationIntentTracker`.

For each conversation segment, the tracker:

* Predicts the current intent
* Calculates the model's confidence
* Applies a confidence threshold
* Records the intent history
* Detects when the current intent changes
* Produces an overall intent flow

For example:

```text
Insurance Enquiry
        ↓
Claim Enquiry
        ↓
Claim Follow-up
```

This is useful in an agent-assist setting because a customer's reason for contacting support can change as the conversation develops.

## Project Structure

```text
Agent_-Assist/
│
├── Bert_model.ipynb
├── script.py
├── intent_tracker_dashboard.py
└── README.md
```

### `Bert_model.ipynb`

Main notebook covering:

* Data loading
* Text preprocessing
* Conversation cleaning
* Conversation segmentation
* Sentence embeddings
* UMAP dimensionality reduction
* Clustering
* Intent labelling
* Dataset preparation
* Logistic Regression training
* Model evaluation

### `script.py`

Contains the supporting intent-processing/classification logic used by the project.

### `intent_tracker_dashboard.py`

Dashboard component for viewing and interacting with the intent-tracking results.

### `README.md`

Project documentation and setup information.

## Technologies

**Python**

**NLP / Machine Learning**

* spaCy
* Sentence Transformers
* Scikit-learn
* Logistic Regression
* TF-IDF
* UMAP
* Agglomerative Clustering

**Data**

* Pandas
* NumPy
* CSV

**Visualisation / Application**

* Matplotlib
* Streamlit

## Running the Project

Install the required Python packages and run the relevant notebook:

```bash
jupyter notebook Bert_model.ipynb
```

For the dashboard:

```bash
streamlit run intent_tracker_dashboard.py
```

## What I learned

This project gave me practical experience working with unstructured conversation data and taking it through the full machine-learning workflow — from preprocessing and exploratory NLP through to creating labelled data, training a classifier, evaluating predictions, and using the model in a conversation-level tracking workflow.

It also showed the challenges that come with real-world intent classification, particularly **class imbalance and limited examples for some intents**.

## Future Improvements

Potential next steps include:

* Increasing the number of labelled examples for underrepresented intents
* Improving conversation segmentation
* Testing transformer-based classification models
* Comparing TF-IDF and embedding-based classifiers
* Improving intent labelling
* Adding more robust evaluation
* Testing the tracker on longer multi-intent conversations
* Adding confidence visualisation to the dashboard
* Integrating the intent tracker with the RAG Agent Assist application
