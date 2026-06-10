# CEGO — Chat Room Monitoring

A chat moderation system built on a simple principle: **no complexity for the sake of complexity**.

The system's key passing criterion is that it must never wrongfully categorise a high-severity gambling addiction signal as clean.

---

## System Overview

<img width="633" height="522" alt="Pipeline overview" src="https://github.com/user-attachments/assets/075587e4-6e76-461d-b970-5a0f3deacea2" />

The pipeline works in four steps:

1. **TF-IDF** (Term Frequency-Inverse Document Frequency) translates each message into numbers
2. **5 handcrafted features** add signal that pure word frequency misses:
   - `msg_length`
   - `exclamations`
   - `has_big_numbers`
   - `offensive_word_count`
   - `gambling_word_count`
3. **Two Logistic Regression classifiers** — one for category, one for severity
4. **A lookup table** maps category + severity to the appropriate action

---

## Dataset

<img width="1935" height="735" alt="Dataset distributions" src="https://github.com/user-attachments/assets/7e7b7827-56d8-4a35-85ae-a52fd43b831f" />

---

## Top Features

<img width="2085" height="1485" alt="Feature importances" src="https://github.com/user-attachments/assets/580268df-0a3d-4eb3-bffe-bcd9771f9a6c" />

---

## Top Words per Category

<img width="2226" height="1703" alt="Top words per category" src="https://github.com/user-attachments/assets/bce53beb-f050-41b1-9b27-94014d0ff86a" />

---

## Confusion Matrix

<img width="2161" height="810" alt="Confusion matrices" src="https://github.com/user-attachments/assets/0efbd7cc-05b9-450c-bb16-16b580ca33e1" />

---
## Results
<img width="511" height="686" alt="image" src="https://github.com/user-attachments/assets/819a042c-3f50-4d75-847c-69cad7f83a5f" />

## Error analysis
<img width="679" height="323" alt="image" src="https://github.com/user-attachments/assets/e03d046b-7659-40f5-85c6-450d225632d1" />


## Reflections

The model is not perfect, but it is transparent and straightforward to improve. Natural next steps include adding more handcrafted features, swapping TF-IDF for a multilingual sentence transformer with knowledge of Danish, or using an LLM as the classifier if inference speed is not a constraint.

Happy to discuss all of this at the interview — have a good day!
