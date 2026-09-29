# Question Difficulty Prediction (QDP)

An AI-based system that predicts whether a question is **Easy, Moderate, or Hard** using only the information present in the question text.

---

## About the Project

I built this project to study a simple but interesting question:

> **Can we predict how difficult a question is just by analyzing the question itself?**

Instead of using student performance, response time, submission statistics, or answer keys, my system uses only the **intrinsic properties of the question text**.

The project combines **handcrafted linguistic and structural features** with machine learning models to identify patterns associated with question difficulty.

The main idea is to represent a question from multiple perspectives:

```text
Question Text
     │
     ├── Linguistic Features
     ├── Readability Features
     ├── Syntactic Features
     ├── Task / Cognitive Features
     ├── MCQ Features
     ├── TF-IDF Features
     ├── Semantic Features
     └── Programming Features
              │
              ↓
       Feature Fusion
              │
              ↓
        ML Classifier
              │
              ↓
    Easy / Moderate / Hard
```

---

# 1. Problem Formulation and Originality

The problem is formulated as a **three-class text classification problem**:

```text
Input  → Question text
Output → Easy / Moderate / Hard
```

The important restriction in my project is that the prediction must be based on the **question itself**.

I deliberately do not use:

* Student success rate
* Number of attempts
* Response time
* Submission statistics
* Answer keys
* Student performance
* External metadata that directly reveals difficulty

My research question is:

> **How much information about question difficulty is contained in the linguistic, structural, semantic, and programming properties of the question text?**

My approach is different from a simple bag-of-words classifier because I extract multiple feature families and study their individual contribution through **ablation experiments**.

---

# 2. AI Concepts and Algorithm Implementation

I implemented a complete text-processing and machine-learning pipeline.

## Preprocessing

Before feature extraction, the question text goes through normalization and structural analysis.

The preprocessing stage handles:

* Unicode normalization
* Newline normalization
* Blank-line handling
* Tokenization
* Sentence detection
* Natural-language views
* Code detection
* Mathematical content
* MCQ structure detection

The purpose is to create consistent representations while preserving useful information from the original question.

---

# 3. Features Extracted

I use **35 logical engineered features**, divided into meaningful feature families.

## 3.1 Lexical Features

These describe the basic structure and vocabulary of the question.

Examples include:

* Character count
* Word count
* Token count
* Unique-token statistics
* Vocabulary-related measures

### Why?

A difficult question often contains more information, terminology, conditions, or instructions. Lexical features capture this **surface-level complexity**.

---

## 3.2 Readability Features

These estimate how difficult the text is to read.

Examples include:

* Flesch-style readability measures
* Sentence complexity
* Word/syllable-related measures

### Why?

A question may become harder because its instructions are linguistically more complex.

Readability features therefore capture **linguistic complexity**.

---

## 3.3 Syntactic Features

These describe the grammatical structure of the question.

Examples include:

* Sentence structure
* Dependency-related statistics
* Part-of-speech patterns
* Syntactic depth-related information

### Why?

Two questions can have similar lengths but very different grammatical complexity.

Syntactic features help capture **structural complexity** that simple word counts cannot.

---

## 3.4 Task / Cognitive Features

These identify the type of task being requested.

Examples include question/task indicators such as:

* Define
* Explain
* Calculate
* Compare
* Prove
* Analyze
* Derive

### Why?

The action requested by a question is often strongly related to the kind of reasoning required.

For example:

```text
"Define..."       → recall
"Calculate..."    → application
"Prove..."        → higher-level reasoning
"Analyze..."      → deeper reasoning
```

These features attempt to capture the **cognitive/task structure** of the question.

---

## 3.5 MCQ Features

When the question contains multiple-choice options, the system detects their structure.

Examples include:

* Whether options are detected
* Number of options
* Option-length characteristics
* Structural statistics of the options

### Why?

The structure of an MCQ can provide information about its difficulty.

For example, questions with multiple detailed or structurally similar options may require more reasoning.

The important point is that the system does **not** use the correct answer as a feature.

---

## 3.6 TF-IDF Features

TF-IDF identifies words that are particularly informative within the dataset.

Conceptually:

```text
TF  → How frequently a word appears
IDF → How rare the word is across questions
```

### Why?

Certain terms can be strongly associated with particular kinds of questions.

TF-IDF therefore provides a representation of **informative vocabulary** rather than treating every word equally.

---

## 3.7 Semantic Features

I also include semantic information to move beyond simple word counts.

The project supports a semantic representation using:

```text
all-MiniLM-L6-v2
        ↓
384-dimensional embedding
        ↓
L2 normalization
        ↓
PCA
        ↓
Reduced semantic representation
```

### Why?

Two questions can use different words while expressing similar concepts.

Semantic embeddings allow the system to capture **meaning and conceptual similarity** that lexical features alone may miss.

---

## 3.8 Programming Features

When programming content or constraints can be detected, additional features are extracted.

Examples include:

* Programming/code presence
* Input-size information
* Complexity-related structural indicators
* Code-related characteristics

### Why?

For programming questions, difficulty is often related to constraints, algorithmic requirements, and code-related structure.

These features allow the model to capture **programming-specific complexity**.

---

# 4. Why Combine All These Features?

A question is not difficult for just one reason.

It may be difficult because of:

```text
Vocabulary
   +
Sentence structure
   +
Task type
   +
Conceptual complexity
   +
Mathematical/code structure
   +
Question format
```

Therefore, instead of relying on a single representation, I combine several complementary views.

The central idea is:

> **Different feature families capture different dimensions of question difficulty.**

---

# 5. Complete Pipeline

The complete pipeline is:

```text
                 QUESTION
                    │
                    ↓
             DATA VALIDATION
                    │
                    ↓
             PREPROCESSING
                    │
        ┌───────────┴───────────┐
        ↓                       ↓
   Text Analysis           Structure Detection
        │                       │
        ├── Lexical             ├── MCQ
        ├── Readability         ├── Code
        ├── Syntax              └── Constraints
        ├── Task
        ├── TF-IDF
        ├── Semantic
        └── Programming
                 │
                 ↓
          FEATURE ASSEMBLY
                 │
                 ↓
        TRANSFORMATIONS
     (imputation / encoding /
      PCA / correlation filter)
                 │
                 ↓
          MODEL TRAINING
                 │
                 ↓
        MODEL SELECTION
                 │
                 ↓
          EVALUATION
                 │
                 ↓
       EASY / MODERATE / HARD
```

---

# 6. AI Algorithms Compared

I implemented multiple machine-learning approaches instead of relying on a single model.

## Logistic Regression

I use Multinomial Logistic Regression as an interpretable baseline.

It provides a simple linear reference point for the engineered features.

---

## Random Forest

Random Forest provides a nonlinear tree-based comparison.

It can capture interactions between features that a linear model cannot easily represent.

---

## XGBoost

XGBoost is the main model used in the project.

It builds an ensemble of decision trees sequentially, where later trees focus on correcting the errors made by earlier trees.

I use it because the final representation is a structured numerical feature matrix containing heterogeneous feature types.

---

# 7. Experimental Methodology

I use a leakage-safe experimental pipeline.

The data is divided into:

```text
70% Training
15% Validation
15% Test
```

Questions belonging to the same duplicate/near-duplicate family are kept within the same split.

This prevents nearly identical questions from appearing in both training and testing.

Data-dependent transformations such as:

* TF-IDF
* PCA
* imputation
* feature filtering
* model selection

are fitted using the training data according to the experimental protocol.

The test set is reserved for final evaluation.

---

# 8. Feature Ablation

One of the main research components of my project is **feature ablation**.

Instead of only asking:

> "How accurate is the final model?"

I also ask:

> "How much does each feature family contribute?"

The cumulative experiment looks like:

```text
A0 → Baseline
 ↓
A1 → Lexical + Readability
 ↓
A2 → + Syntax
 ↓
A3 → + Task
 ↓
A4 → + MCQ
 ↓
A5 → + TF-IDF
 ↓
A6 → + Semantic Scalars
 ↓
A7 → + Programming
 ↓
A8 → + Semantic Embedding
```

This allows me to study the contribution of different feature families.

---

# 9. Evaluation

I evaluate the models using multiple metrics instead of relying only on accuracy.

The main metrics include:

* Accuracy
* Macro-F1
* Balanced Accuracy
* Ordinal MAE
* Confusion Matrix
* Per-class Precision / Recall / F1

I use Macro-F1 because it gives equal importance to all three difficulty classes.

I also use Ordinal MAE because the labels naturally have an order:

```text
Easy < Moderate < Hard
```

Therefore, confusing Easy with Moderate is different from confusing Easy with Hard.

---

# 10. Results

### Initial Baseline

My original 5K experiment achieved approximately:

| Metric           |    Result |
| ---------------- | --------: |
| Test Accuracy    | **49.7%** |
| Test Macro-F1    | **49.1%** |
| Test Ordinal MAE | **0.619** |

The initial experiment helped identify important limitations in the dataset and representation.

---

# 11. My Student-Designed Improvement

One of my main contributions is a **data-centric improvement strategy**.

Instead of simply increasing model complexity, I focused on improving the information available to the model.

### Improvement 1 — Dataset Expansion

I expanded the question corpus to increase the amount and diversity of training data.

### Improvement 2 — Ambiguity and Artifact Removal

I cleaned and normalized problematic question text, including source-format inconsistencies and ambiguous artifacts.

The goal was to ensure that the model learns from the **actual difficulty-related content**, rather than irrelevant formatting noise.

### Improvement 3 — Controlled Re-experimentation

I then reran the experimental pipeline with the improved dataset/configuration and compared it against the original baseline.

### Observed result

```text
Original baseline       → 49.7%
Improved experiment     → 55.0%

Absolute improvement    → +5.3 percentage points
```

This improvement is important because it suggests that **data quality and representation can matter substantially, not only model complexity**.

---

# 12. Interpretability

I also study which features the model relies on.

The project supports:

* XGBoost feature importance
* Permutation importance
* SHAP-based interpretation

These help answer questions such as:

```text
Which features does the model rely on?
Which feature families contribute most?
What characteristics distinguish the predicted classes?
```

Feature importance is interpreted as **model reliance**, not proof that a feature causes human-perceived difficulty.

---

# 13. Working Demo

The trained model can be used for inference on a new question.

Example:

```text
Input:
"Prove that every finite subgroup of the
multiplicative group of a field is cyclic."

                ↓

       Question Difficulty Predictor

                ↓

           Prediction:
              HARD
```

The inference pipeline reuses the fitted preprocessing, feature transformations, and trained model rather than retraining the system for each prediction.

---

# 14. Technology Stack

### Programming

* Python

### NLP / Text Processing

* spaCy
* text processing and tokenization utilities
* TF-IDF

### Semantic Representation

* Sentence Transformers
* `all-MiniLM-L6-v2`
* PCA

### Machine Learning

* Scikit-learn
* XGBoost

### Data Processing

* NumPy
* Pandas
* JSONL / Parquet

### Testing

* Pytest

---

# 15. Project Structure

```text
QDP2/
│
├── src/
│   └── qdp/
│       ├── data/
│       ├── preprocessing/
│       ├── features/
│       ├── semantic/
│       ├── transform/
│       ├── models/
│       ├── evaluation/
│       ├── artifacts/
│       └── utils/
│
├── scripts/
│   ├── train.py
│   ├── predict.py
│   ├── evaluate.py
│   └── run_ablations.py
│
├── config/
│
├── tests/
│
├── data/
│
├── artifacts/
│
└── README.md
```

---

# 16. What I Learned

Through this project, I worked with:

* NLP preprocessing
* Feature engineering
* Semantic embeddings
* Dimensionality reduction
* Text classification
* Ensemble learning
* Experimental design
* Ablation studies
* Model interpretability
* Data leakage prevention
* Reproducible ML pipelines

More importantly, I learned that improving an ML model is not always about choosing a more complex algorithm. **The quality of the data, the representation of the problem, and the experimental methodology can be equally important.**

---

# 17. Current Limitations

The current project has several limitations.

The initial dataset is strongly concentrated toward programming-style questions, so broader domain coverage is still an important area for improvement.

The semantic branch also requires the full P0 environment and the specified MiniLM model for the complete hybrid experiment.

Therefore, future experiments will focus on:

* Larger and more diverse datasets
* Better semantic representations
* More robust generalization across domains
* Additional controlled model comparisons
* Further analysis of the Moderate class
* A richer interactive application

---

# 18. Future Work

My planned extensions include:

```text
Larger / more diverse dataset
          ↓
Improved semantic representation
          ↓
Better generalization
          ↓
More extensive experimentation
          ↓
Interactive deployment
```

An interesting future direction is to investigate **ordinal classification**, since the target classes naturally follow:

```text
Easy → Moderate → Hard
```

---

# 19. Conclusion

This project demonstrates that question difficulty can be approached as a **text-based AI classification problem**.

My approach combines:

```text
Linguistic information
        +
Structural information
        +
Task information
        +
Semantic information
        +
Programming information
        ↓
   Machine Learning
        ↓
Easy / Moderate / Hard
```

Rather than treating the problem as a simple text-classification task, I designed the system to investigate **which characteristics of a question carry difficulty information**.

The project also demonstrates a practical lesson from experimentation: improving the **dataset quality and representation** produced a measurable increase from the original **49.7% baseline to 55.0%** in my improved experiment.

---

## Author

**Lokesh P. Patle**

Birla Institute of Technology, Mesra

GitHub: [@lokeshpatle](https://github.com/lokeshpatle)

---

## Project Goal

> **Build an explainable and reproducible AI system that predicts question difficulty from the question text itself, while understanding which linguistic, structural, semantic, and programming features contribute to that prediction.**
