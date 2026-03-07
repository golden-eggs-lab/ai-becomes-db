# AI Becomes DB

**When AI Pipelines Become Database Workloads: An Experimental Study of Similarity, Reuse, and Ranking**

This repository contains the source code and experiment scripts for reproducing results in the paper. We study eight embedding-centric ML algorithms across six DCAI tasks and show that their execution bottlenecks correspond to violations of three **database execution invariants**. Aligning execution with these invariants consistently reduces runtime while preserving task performance.

## Execution Invariants

| Invariant              | Bottleneck Addressed            | Implementation                                         |
| ---------------------- | ------------------------------- | ------------------------------------------------------ |
| **IV1: Approximation** | B1: Exact similarity evaluation | ANN indexing (FAISS IVF, NearPy LSH, sklearn BallTree) |
| **IV2: Reuse**         | B2: Overlapping computation     | Intermediate result materialization & reuse            |
| **IV3: Partial Order** | B3: Total ordering              | Top-k selection via `argpartition`                     |

## Algorithms

| Task                | Algorithm              | Invariants    | Directory                                                  |
| ------------------- | ---------------------- | ------------- | ---------------------------------------------------------- |
| Coreset Selection   | **CRAIG** [43]         | IV1, IV2      | [`coreset_selection/craig/`](coreset_selection/craig/)     |
| Coreset Selection   | **DEFT-UCS** [13]      | IV1, IV2, IV3 | [`coreset_selection/deftucs/`](coreset_selection/deftucs/) |
| Active Learning     | **CAL** [40]           | IV1, IV2      | [`active_learning/`](active_learning/)                     |
| Semantic Dedup      | **SemDeDup** [1]       | IV1, IV2      | [`semantic_dedup/`](semantic_dedup/)                       |
| OOD Detection       | **KNN-OOD** [58]       | IV1           | [`ood_detection/`](ood_detection/)                         |
| In-Context Learning | **KNN Prompting** [66] | IV2           | [`knn_prompting/`](knn_prompting/)                         |
| Code Data Pruning   | **SCIP** [68]          | IV1, IV2, IV3 | [`code_pruning/`](code_pruning/)                           |

## System Setups

Each algorithm is evaluated under up to three execution environments:

| Setup               | Technology                         | Description                                          |
| ------------------- | ---------------------------------- | ---------------------------------------------------- |
| **In-Memory**       | FAISS / scikit-learn / NearPy      | Single-machine, all data in CPU memory               |
| **Vector Database** | Milvus-Lite (PyMilvus)             | Database-backed embedding storage and indexed search |
| **Distributed**     | Apache Spark (PySpark + MLlib LSH) | Cluster-style distributed execution                  |

## Experiments

| #   | Experiment                | Paper Reference         | Script Location                          |
| --- | ------------------------- | ----------------------- | ---------------------------------------- |
| 1   | End-to-End Evaluation     | Table 3                 | Each algorithm's main script             |
| 2   | Ablation of IV1/IV2/IV3   | Table 4                 | `run_ablation.*` scripts                 |
| 3   | Component-Level Breakdown | Table 5                 | Breakdown fields in E2E/ablation scripts |
| 4   | Execution Overhead        | Table 6                 | Build/query time fields in scripts       |
| 5   | Cross-Setup Evaluation    | Table 7                 | In-memory + Milvus + Spark scripts       |
| 6   | Memory Constraint         | Figure 3                | CRAIG memory-limited scripts             |
| 7   | Dataset Size Ratio        | Figure 4 (top)          | Ratio sampling scripts                   |
| 8   | Cache Capacity            | Figure 4 (bottom-right) | Cache size sweep scripts                 |
| 9   | ANN Hyper-Parameter Sweep | Figure 5                | ANN sweep scripts                        |

## Repository Structure

```
ai-becomes-db/
├── README.md
├── coreset_selection/
│   ├── craig/         # CRAIG: IV1 (ANN) + IV2 (Residual Reuse)
│   └── deftucs/       # DEFT-UCS: IV1 + IV2 (Distance Reuse) + IV3 (Top-k)
├── active_learning/   # CAL: IV1 (ANN) + IV2 (Probability Reuse)
├── semantic_dedup/    # SemDeDup: IV1 (ANN) + IV2 (Search Results Reuse)
├── ood_detection/     # KNN-OOD: IV1 (ANN with GPU)
├── knn_prompting/     # KNN Prompting: IV2 (Intermediate Results Reuse)
└── code_pruning/      # SCIP: IV1 + IV2 (Distance Reuse) + IV3 (Top-k)
```

## Datasets

| Algorithm            | Dataset                      | Acquisition                                 |
| -------------------- | ---------------------------- | ------------------------------------------- |
| CRAIG                | MNIST, Fashion-MNIST         | Auto-download (`tensorflow.keras.datasets`) |
| DEFT-UCS             | CoEDIT, WikiLarge            | See README for download instructions        |
| CAL                  | SST-2, IMDB                  | Auto-download (`torchvision` / HuggingFace) |
| SemDeDup / FairDeDup | CIFAR-10, CIFAR-100          | Auto-download (`torchvision`)               |
| KNN-OOD              | ImageNet-1K, CIFAR-10        | ImageNet: manual; CIFAR-10: auto            |
| KNN Prompting        | SST-2, AGNews                | Included in `data/` or HuggingFace          |
| SCIP                 | CodeSearchNet (Python, Java) | Auto-download (`datasets` library)          |

## Hardware

All experiments are conducted on a machine with 64 vCPUs, 500 GB memory, and 8× RTX 6000 GPUs. Results are reported as mean ± std over 3 independent runs.
