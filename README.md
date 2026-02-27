# M.Tech AI/ML Academic Portfolio
**BITS Pilani M.Tech Program - Comprehensive AI/ML Journey**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red.svg)](https://pytorch.org)
[![Docker](https://img.shields.io/badge/Docker-Enabled-blue.svg)](https://docker.com)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-Ready-blue.svg)](https://kubernetes.io)

> *A comprehensive collection of AI/ML projects, assignments, and research implementations from my M.Tech program at BITS Pilani. This repository demonstrates practical applications across Deep Learning, Reinforcement Learning, Natural Language Processing, MLOps, and Generative AI.*

---

## Table of Contents
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Projects Showcase](#projects-showcase)
    - [Semester 3: Advanced Systems & MLOps](#semester-3-advanced-systems--mlops)
    - [Semester 2: Deep Learning & NLP](#semester-2-deep-learning--nlp)
    - [Semester 1: Foundations & Game AI](#semester-1-foundations--game-ai)
- [Technologies Used](#technologies-used)
- [Academic Progress](#academic-progress)
- [Setup & Usage](#setup--usage)
- [Contact](#contact)

---

## Overview

This repository showcases my academic journey through advanced AI/ML concepts, transitioning from foundational algorithms to production-ready MLOps pipelines and Generative AI systems.

**Key Highlights:**
- **End-to-End MLOps**: Full lifecycle management using Docker, Kubernetes, and CI/CD pipelines.
- **Generative AI**: Implementation of RAG systems and LLM integration.
- **Advanced NLP**: From statistical translation to modern transformer-based architectures.
- **Reinforcement Learning**: Solving complex environments with Actor-Critic and DQN agents.

---

## Repository Structure

```text
academic-ai-ml-portfolio/
├── semester-3/                       # Advanced Systems, GenAI & MLOps
│   ├── conversational-ai/           # Hybrid RAG & LLM Applications
│   ├── mlops/                       # Production ML Pipelines (Docker/K8s)
│   └── nlp-applications/            # SMT & Web-based NLP Tools
├── semester-2/                       # Deep Learning & Core NLP
│   ├── deep-neural-networks/        # CNN/DNN Architectures
│   ├── deep-reinforcement-learning/ # RL Algorithms (DQN, PPO)
│   └── natural-language-processing/ # Financial Sentiment Analysis
├── semester-1/                       # Foundations
│   └── game-ai-minimax/             # Strategic Game AI
└── environments/                     # Configuration & Setup
```

---

## Projects Showcase

### Semester 3: Advanced Systems & MLOps

#### 1. **Binary Image Classification – MLOps Pipeline (Cats vs Dogs)**
**Domain**: MLOps | **Location**: [semester-3/mlops/assignment2](semester-3/mlops/assignment2/README.md)
- **Description**: An end-to-end MLOps pipeline covering data versioning (DVC), experiment tracking (MLflow), FastAPI inference, containerization (Docker), and CI/CD via GitHub Actions.
- **Tech Stack**: TensorFlow/Keras, MLflow, DVC, FastAPI, Docker, GitHub Actions, Prometheus.

#### 2. **Hybrid RAG Question Answering System**
**Domain**: Generative AI | **Location**: `semester-3/conversational-ai/assignment/assignment-2`
- **Description**: A sophisticated RAG (Retrieval-Augmented Generation) system that combines dense (FAISS) and sparse (BM25) retrieval to answer questions from a dynamic Wikipedia corpus. Features Reciprocal Rank Fusion (RRF) for optimal context retrieval.
- **Tech Stack**:
    - **LLM**: Flan-T5-base (Instruction Tuned)
    - **Vector DB**: FAISS (Dense Retrieval)
    - **Search**: Rank-BM25 (Sparse Retrieval)
    - **Backend**: Flask

#### 3. **Heart Disease Prediction MLOps Pipeline**
**Domain**: MLOps & Healthcare | **Location**: [semester-3/mlops/assignment1/mlops-heart-disease](semester-3/mlops/assignment1/mlops-heart-disease/README.md)
- **Description**: A production-ready Machine Learning pipeline for heart disease prediction, demonstrating end-to-end MLOps practices. Includes containerization, orchestration, and automated testing.
- **Tech Stack**:
    - **Infrastructure**: Docker, Kubernetes (K8s)
    - **ML Framework**: Scikit-learn
    - **Tools**: PyTest, CI/CD workflows

#### 4. **Statistical Machine Translation (SMT) System**
**Domain**: NLP Applications | **Location**: `semester-3/nlp-applications/assignment/assignment2`
- **Description**: A web-based translation workbench supporting multiple languages with integrated BLEU score evaluation. Allows comparison of translations against reference texts with detailed n-gram precision analysis.
- **Tech Stack**:
    - **Core**: Python, Flask
    - **Evaluation**: BLEU (SacreBLEU), N-gram analysis
    - **API**: Google Translate API

#### 5. **Intelligent Spell Checker**
**Domain**: NLP Applications | **Location**: `semester-3/nlp-applications/spell-checker-web-app-flask`
- **Description**: A Flask-based web application for real-time spell checking. Features a history tracking system to learn from past errors and provides context-aware suggestions.
- **Tech Stack**: Flask, PySpellChecker, JSON Storage

---

### Semester 2: Deep Learning & NLP

#### 6. **Financial Sentiment Analysis Engine**
**Domain**: NLP | **Location**: [semester-2/natural-language-processing/assignment1](semester-2/natural-language-processing/assignment1/README.md)
- **Description**: An NLP system designed to analyze financial text data. It compares Skip-gram and CBOW word embedding models to classify market sentiment, aiding in automated trading decisions.
- **Tech Stack**: NLTK, Gensim (Word2Vec), Scikit-learn, Pandas

#### 7. **Sepsis Treatment Optimization & Drone Battery Management**
**Domain**: Deep Reinforcement Learning | **Location**: [semester-2/deep-reinforcement-learning/assignment1](semester-2/deep-reinforcement-learning/assignment1/README.md)
- **Description**: Application of Actor-Critic RL algorithms to optimize sepsis treatment strategies in ICU settings, alongside a Drone Battery Management system using DQN/DDQN for autonomous surveillance.
- **Tech Stack**: PyTorch, OpenAI Gym, Stable-Baselines3

#### 8. **Deep Neural Networks: Architecture Design & Optimization**
**Domain**: Deep Learning | **Location**: [semester-2/deep-neural-networks/assignment1](semester-2/deep-neural-networks/assignment1/README.md)
- **Description**: A comprehensive study of Neural Network architectures on the MNIST dataset. Implementation and comparative analysis of regularization techniques (Dropout, L2) and depth variations to optimize performance.
- **Tech Stack**: TensorFlow/Keras, NumPy, Matplotlib

---

### Semester 1: Foundations & Game AI

#### 9. **Strategic Game AI: Crossword Puzzle with Minimax Algorithm**
**Domain**: Game Theory | **Location**: [semester-1/game-ai-minimax](semester-1/game-ai-minimax/README.md)
- **Description**: An AI agent capable of playing a strategic two-player crossword game. Implements the Minimax algorithm with depth-limited search to make optimal moves against human or AI opponents.
- **Tech Stack**: Python, Search Algorithms, Game Theory

---

## Technologies Used

| Domain | Technologies |
|--------|-------------|
| **Generative AI** | Transformers (Hugging Face), RAG, FAISS, LangChain concepts |
| **MLOps** | MLflow, DVC, Docker, Kubernetes, CI/CD, PyTest, Prometheus |
| **Deep Learning** | TensorFlow, Keras, PyTorch |
| **NLP** | NLTK, Spacy, Gensim, BLEU, Word2Vec |
| **Reinforcement Learning** | OpenAI Gym, Stable-Baselines3, Ray RLlib |
| **Backend/Web** | FastAPI, Flask, Python, JSON |
| **Data Science** | Pandas, NumPy, Scikit-learn, Matplotlib |

---

## Academic Progress

- **Semester 1**: Foundations (Algorithms, Python, Game AI)
- **Semester 2**: Core AI (Deep Learning, RL, NLP)
- **Semester 3**: Advanced Systems (Generative AI, MLOps, Applied NLP)
- **Semester 4**: Dissertation & Research (Upcoming)

---

## Setup & Usage

### Prerequisites
```bash
# Python 3.8+
python --version

# Docker (for MLOps projects)
docker --version
```

### Installation
```bash
# Clone repository
git clone https://github.com/anandsuraj/mtech-ai-ml-journey.git
cd mtech-ai-ml-journey

# Create virtual environment
python -m venv venv
source venv/bin/activate
```

### Running a Project (Example: Hybrid RAG)
```bash
cd semester-3/conversational-ai/assignment/assignment-2
pip install -r requirements.txt
./run.sh
```

---

## Contact

**Suraj Anand**  
*M.Tech AI/ML, BITS Pilani*  

[surya13493@gmail.com](mailto:surya13493@gmail.com)  
[LinkedIn](https://linkedin.com/in/anandsuraj) | [GitHub](https://github.com/anandsuraj)

---
*Created as part of the M.Tech AI/ML academic curriculum.*