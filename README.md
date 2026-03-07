# AI Becomes DB

Optimizing data-centric AI algorithms with database techniques: **ANN indexing**, **intermediate result reuse**, and **TopK pruning** across In-Memory (FAISS), Milvus, and Spark backends.

## Algorithms

| #   | Task              | Algorithm         | Optimizations               | Directory                                                  |
| --- | ----------------- | ----------------- | --------------------------- | ---------------------------------------------------------- |
| 1   | Coreset Selection | **CRAIG**         | ANN + Residual Reuse        | [`coreset_selection/craig/`](coreset_selection/craig/)     |
| 2   | Coreset Selection | **DEFT-UCS**      | ANN + Distance Reuse + TopK | [`coreset_selection/deftucs/`](coreset_selection/deftucs/) |
| 3   | Active Learning   | **CAL**           | ANN + Probability Reuse     | [`active_learning/`](active_learning/)                     |
| 4   | Semantic Dedup    | **SemDeDup**      | ANN + Search Results Reuse  | [`semantic_dedup/`](semantic_dedup/)                       |
| 5   | OOD Detection     | **KNN-OOD**       | ANN (IVF) + GPU             | [`ood_detection/`](ood_detection/)                         |
| 6   | KNN Prompting     | **KNN Prompting** | Intermediate Results Reuse  | [`knn_prompting/`](knn_prompting/)                         |
| 7   | Code Pruning      | **SCIP**          | ANN + Distance Reuse + TopK | [`code_pruning/`](code_pruning/)                           |

## Repository Structure

```
ai-becomes-db/
├── README.md                      # This file
├── coreset_selection/
│   ├── craig/                     # CRAIG with ANN + Residual Reuse
│   └── deftucs/                   # DEFT-UCS with ANN + Distance Reuse + TopK
├── active_learning/               # CAL with ANN + Probability Reuse
├── semantic_dedup/                # SemDeDup with ANN + Search Results Reuse
├── ood_detection/                 # KNN-OOD with ANN (IVF) + GPU
├── knn_prompting/                 # KNN Prompting with Intermediate Results Reuse
└── code_pruning/                  # SCIP with ANN + Distance Reuse + TopK
```

## Backends

Each algorithm supports up to three deployment backends:

| Backend       | Technology             | Use Case                                      |
| ------------- | ---------------------- | --------------------------------------------- |
| **In-Memory** | FAISS / scikit-learn   | Single-machine, fastest for small-medium data |
| **Milvus**    | Milvus vector database | Persistent storage, warm-start queries        |
| **Spark**     | Apache Spark (PySpark) | Distributed, large-scale data                 |

## Getting Started

1. Navigate to any algorithm directory (e.g., `cd coreset_selection/craig/`)
2. Install dependencies: `pip install -r requirements.txt`
3. Follow the algorithm-specific `README.md`

## Datasets

| Algorithm     | Dataset              | Acquisition                                 |
| ------------- | -------------------- | ------------------------------------------- |
| CRAIG         | MNIST, Fashion MNIST | Auto-download (`tensorflow.keras.datasets`) |
| DEFT-UCS      | WikiLarge            | See README for download                     |
| CAL           | CIFAR-10             | Auto-download (`torchvision`)               |
| SemDeDup      | CIFAR-10             | Auto-download                               |
| KNN-OOD       | ImageNet, CIFAR-10   | ImageNet: manual download; CIFAR-10: auto   |
| KNN Prompting | SST-2, AGNews        | Included in `data/` or HuggingFace          |
| SCIP          | The Stack v1.1       | Auto-download (`datasets` library)          |
