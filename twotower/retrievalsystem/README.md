# Two-Tower Recommendation Model

This project implements a **two-tower neural network architecture** for personalized recommendations.  
The model learns separate embeddings for users and items, then combines them to predict relevance.  
This approach is widely used in modern recommendation systems for scalability and efficiency.

---

## Features
- Learns user and item embeddings independently
- Supports binary relevance labels (liked vs. not liked)
- Evaluates performance using recall@k and other ranking metrics
- Built with PyTorch for flexibility and extensibility

---

## Requirements
All dependencies are listed in the root `requirements.txt` file of this repository.  
Install them with:

```bash
pip install -r ../../requirements.txt
