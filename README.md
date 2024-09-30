# Bio-Clinical Named Entity Recognition (NER) Model

This repository contains code for developing and evaluating a **bio-clinical Named Entity Recognition (NER)** model. The project uses a fine-tuned version of the **BERT-based transformer model**, pre-trained on a general corpus and adapted for bio-clinical text using a **synthetic dataset**. The NER model is optimized for recognizing and classifying medical entities in biomedical and clinical datasets.

## Project Overview

The project involves:

1. **Model Architecture**: 
   - The model is based on **BERT (Bidirectional Encoder Representations from Transformers)**, a transformer-based deep learning model that has been pre-trained on a large corpus. We fine-tune this model specifically for bio-clinical NER tasks.

2. **Synthetic Dataset**: 
   - The model is trained on a synthetic bio-clinical dataset that mimics real-world clinical scenarios, including entity types such as diseases, symptoms, and treatments.

3. **Dataset Preprocessing**: 
   - Mapping bio-clinical dataset labels into numeric values for model compatibility.
   - Applying data filtration to clean the dataset and ensure high-quality training data.

4. **Fine-Tuning**: 
   - Fine-tuning the pre-trained BERT model on bio-clinical NER tasks allows the model to specialize in recognizing medical entities. This significantly enhances its accuracy on biomedical data.

5. **Evaluation**: 
   - The model is evaluated using a bio-clinical test set, measuring its effectiveness in recognizing clinical entities.

6. **Metric Computation**: 
   - Performance is measured using accuracy, F1 score, precision, and recall to assess the NER model's clinical entity recognition capabilities.
