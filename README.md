# Monologue: Movie-Dialogue Conversational AI

## Project Brief
**Monologue** is a specialized conversational agent designed to simulate cinematic dialogue. While general-purpose chatbots often provide generic responses, this project focuses on fine-tuning the **DialoGPT-medium** architecture specifically on the **Cornell Movie-Dialogue Corpus**. The goal was to bridge the gap between standard AI messaging and the nuanced, subtext-heavy exchanges found in professional film scripts.

## My Contributions
To push the baseline implementation further, I implemented several technical enhancements to improve both model quality and training efficiency:

* **Data Quality Engineering**: Developed a custom filtering pipeline to remove "low-information" dialogue pairs (e.g., short affirmations like "Yes" or "No"). This forced the model to learn more descriptive and engaging conversational patterns.
* **Memory-Optimized Training**: Solved hardware constraints (15GB VRAM limits) by integrating **Gradient Accumulation** and **FP16 Mixed Precision**. This allowed for stable training of a 345M parameter model on consumer-grade hardware without runtime crashes.
* **Hyperparameter Tuning**: Optimized the learning rate and increased the training regimen to 3 epochs to ensure the model captured cinematic "rhythm" and subtext without overfitting.
* **Interactive Deployment**: Designed and deployed a stateful web interface using **Streamlit**, enabling users to interact with the fine-tuned model in real-time with session-based memory.

## Key Learnings
* **End-to-End NLP Pipeline**: Gained a deep understanding of the journey from raw, unstructured text data to a deployed transformer-based web application.
* **Large Language Model (LLM) Fine-Tuning**: Mastered the use of the Hugging Face `Trainer` API and understood the mathematical significance of **Causal Language Modeling (CLM)**.
* **Hardware Resource Management**: Learned practical strategies for training large models on restricted compute resources, specifically the trade-offs between batch size, sequence length, and gradient accumulation.
* **Decoding Strategies**: Experimented with **Top-p (Nucleus) Sampling** and **Temperature** settings to balance the bot’s creativity versus its coherence.

## Project Structure
* `app.py`: Streamlit deployment script.
* `monologue_final.ipynb`: Optimized training notebook.

---
**Developer:** Kanishk Singh  
**Architecture:** DialoGPT-Medium  
**Dataset:** Cornell Movie-Dialogue Corpus