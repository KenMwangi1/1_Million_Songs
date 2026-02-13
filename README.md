# 🎵 Million Song Recommendation System

# _Production-Ready Machine Learning Project_ <img width="5592" height="3728" alt="image" src="https://github.com/user-attachments/assets/1fafa166-6c1e-45a1-b087-251f3cde9448" />


A scalable recommendation engine built on the Million Song Dataset using collaborative filtering and similarity modeling. Designed with production architecture principles, performance evaluation, and extensibility in mind.



## 🔥 Quick Summary

What this project demonstrates:

        Built a scalable recommendation engine using the Million Song Dataset
        
        Implemented collaborative filtering and item-similarity models
        
        Optimized large sparse interaction matrices for performance
        
        Designed production-grade ML project architecture
        
        Evaluated models using Precision@K, Recall@K, and MAP
        
        Improved recommendation precision by ~2–3x over baseline

#### Core Skills Demonstrated:

        Recommender Systems
        
        Sparse Matrix Optimization
        
        Similarity Modeling (Cosine)
        
        Performance Evaluation Metrics
        
        Modular ML System Design
        
        Production-Oriented Code Structure

### 🚀 Why This Project Matters

This project demonstrates:

    ✔ End-to-end ML pipeline design
    
    ✔ Scalable recommendation system architecture
    
    ✔ Sparse matrix optimization
    
    ✔ Model evaluation & performance benchmarking
    
    ✔ Production-grade project structure
    
    ✔ Reproducible data science workflow
    
<img width="540" height="360" alt="image" src="https://github.com/user-attachments/assets/80b6c6cf-f35e-4dde-a2ac-066f34c6f2e6" />

It reflects real-world recommender system challenges including:

    Cold start problem
    
    Data sparsity
    
    Popularity bias
    
    Memory optimization
    
    Model scalability
    
    🧠 Business Problem
    
    Streaming platforms need to:
    
    Increase user engagement
    
    Improve retention
    
    Personalize content delivery
    
    Reduce churn

This project builds a personalized music recommendation engine using user listening history.

### 🏗️ Production-Grade Project Structure

                million-song-recommendation/
                │
                ├── data/
                │   ├── raw/
                │   ├── processed/
                │   └── interim/
                │
                ├── notebooks/
                │   └── exploration.ipynb
                │
                ├── src/
                │   ├── data/
                │   │   ├── make_dataset.py
                │   │   └── preprocess.py
                │   │
                │   ├── features/
                │   │   └── build_features.py
                │   │
                │   ├── models/
                │   │   ├── popularity_model.py
                │   │   ├── collaborative_filtering.py
                │   │   ├── similarity_model.py
                │   │   └── train_model.py
                │   │
                │   ├── evaluation/
                │   │   └── metrics.py
                │   │
                │   └── inference/
                │       └── recommend.py
                │
                ├── tests/
                │
                ├── requirements.txt
                ├── config.yaml
                ├── README.md
                └── main.py


This structure separates:

    Data engineering
    
    Feature engineering
    
    Model training
    
    Evaluation
    
    Inference
    
    Configuration management

Exactly how production ML systems are organized.

### ⚙️ Tech Stack

Python

Pandas / NumPy

Scikit-learn

SciPy (Sparse Matrices)

Matplotlib / Seaborn

Jupyter

YAML (Configuration management)

### 🧠 Modeling Approaches
1️⃣ Popularity-Based Recommendation

    - Baseline benchmark
    
    - Top-N songs by aggregated play count
    
    - Handles cold start users

2️⃣ User-Based Collaborative Filtering

    - User-item interaction matrix
    
    - Cosine similarity
    
    - K-Nearest Neighbors approach

3️⃣ Item-Based Similarity Model

    - Song-to-song similarity
    
    - Sparse matrix optimization
    
    - Memory-efficient similarity computation

### 📊 Performance Metrics

    Evaluation performed using train/test split on user interactions.
    
###  _Metrics Used_
    
     - Precision@K
      
     - Recall@K
      
     - F1@K
      
     - Mean Average Precision (MAP)
      
     - Coverage
      
     - Diversity Score

### 📈 Model Performance
      Model	Precision@10	Recall@10	MAP	Coverage
      Popularity	0.12	0.08	0.07	15%
      User-CF	    0.31	0.24	0.22	48%
      Item-CF	    0.34	0.27	0.25	52%

### Key Insight:
Collaborative filtering improved precision by ~2.8x over the baseline popularity model.

### 📈 System Optimization

- Sparse CSR matrices for memory efficiency

- Vectorized similarity computation

- Reduced dimensionality experimentation

- Efficient ranking using partial sorting

- Config-driven hyperparameters

## 🔍 Scalability Considerations

If deployed at scale:

- Move to distributed matrix computation (Spark MLlib)

- Store embeddings in Redis / Vector DB

- Batch retraining with Airflow

- Serve recommendations via REST API (FastAPI)

- Cache popular recommendations

▶️ How to Run

1️⃣ Clone Repository
git clone https://github.com/yourusername/million-song-recommendation.git
cd million-song-recommendation

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run Training Pipeline
python main.py --train

4️⃣ Generate Recommendations
python main.py --recommend --user_id=<USER_ID>

#### 🧪 Testing

Unit tests included for:
    
    Data preprocessing
    
    Similarity calculations
    
    Recommendation outputs
    
    Metric evaluation functions
  
    Run tests:
    
    pytest tests/

#### 🧩 Future Improvements

    Matrix Factorization (SVD / ALS)
    
    Implicit Feedback Modeling
    
    Neural Collaborative Filtering
    
    Embedding-based recommendation
    
    Real-time recommendation API
    
    Model versioning with MLflow
    
    Docker containerization
    
    CI/CD pipeline

#### 👤 Author

Ken Mwangi
Data Engineer | Machine Learning Engineer | AWS Certified | Data Analyst

Portfolio website: https://KenMwangi1.github.io/

LinkedIn: https://www.linkedin.com/in/ken-mwangi-81478028/
