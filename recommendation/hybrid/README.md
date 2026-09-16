# Hybrid Recommendation Engine

This project implements a **hybrid recommendation system** that combines collaborative filtering and content-based methods.  
By leveraging both user–item interaction data and item metadata (such as genres, categories, or descriptions), the hybrid engine achieves better personalization and mitigates cold-start issues.

---

## Features
- Integrates collaborative filtering (matrix factorization, embeddings) with content-based similarity
- Handles cold-start scenarios by using item features when user history is sparse
- Evaluates performance using recall@k, precision@k, and nDCG
- Built with PyTorch and scikit-learn for flexibility

---

## Requirements
All dependencies are listed in the root `requirements.txt` file of this repository.  
Install them with:

```bash
pip install -r ../../requirements.txt
