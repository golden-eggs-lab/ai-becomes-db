# AI Becomes DB

**When AI Pipelines Become Database Workloads: An Experimental Study of Similarity, Reuse, and Ranking**

This repository contains the source code and experiment scripts for reproducing all results in the paper. We study eight embedding-centric ML algorithms across six DCAI tasks and show that their execution bottlenecks correspond to violations of three **database execution invariants**. Aligning execution with these invariants consistently reduces runtime while preserving task performance.

## Execution Invariants

| Invariant              | Bottleneck Addressed            | Implementation                                         |
| ---------------------- | ------------------------------- | ------------------------------------------------------ |
| **IV1: Approximation** | B1: Exact similarity evaluation | ANN indexing (FAISS IVF, NearPy LSH, sklearn BallTree) |
| **IV2: Reuse**         | B2: Overlapping computation     | Intermediate result materialization & reuse            |
| **IV3: Partial Order** | B3: Total ordering              | Top-k selection via `argpartition`                     |

## Algorithms

| Task                | Algorithm                | Invariants    | Directory                                                  |
| ------------------- | ------------------------ | ------------- | ---------------------------------------------------------- |
| Coreset Selection   | **CRAIG** [43]           | IV1, IV2      | [`coreset_selection/craig/`](coreset_selection/craig/)     |
| Coreset Selection   | **DEFT-UCS** [13]        | IV1, IV2, IV3 | [`coreset_selection/deftucs/`](coreset_selection/deftucs/) |
| Active Learning     | **CAL** [40]             | IV1, IV2      | [`active_learning/`](active_learning/)                     |
| Semantic Dedup      | **SemDeDup & FairDeDup** | IV1, IV2      | [`semantic_dedup/`](semantic_dedup/)                       |
| OOD Detection       | **KNN-OOD** [58]         | IV1           | [`ood_detection/`](ood_detection/)                         |
| In-Context Learning | **KNN Prompting** [66]   | IV2           | [`knn_prompting/`](knn_prompting/)                         |
| Code Data Pruning   | **SCIP** [68]            | IV1, IV2, IV3 | [`code_pruning/`](code_pruning/)                           |

## System Setups

| Setup               | Technology                         | Description                                          |
| ------------------- | ---------------------------------- | ---------------------------------------------------- |
| **In-Memory**       | FAISS / scikit-learn / NearPy      | Single-machine, all data in CPU memory               |
| **Vector Database** | Milvus-Lite (PyMilvus)             | Database-backed embedding storage and indexed search |
| **Distributed**     | Apache Spark (PySpark + MLlib LSH) | Cluster-style distributed execution                  |

## Experiments

The paper presents 9 experiments. The table below shows which algorithms are covered in each experiment and where the corresponding scripts are located.

### Exp 1: End-to-End Evaluation (Table 3)

Compares original vs IV-aligned runtime and task performance.

| Algorithm            | Script                                                                 |
| -------------------- | ---------------------------------------------------------------------- |
| CAL                  | `active_learning/run_e2e_comparison.sh`                                |
| DEFT-UCS             | `coreset_selection/deftucs/benchmark_e2e_all.sh`                       |
| CRAIG                | `coreset_selection/craig/run_craig_benchmark.py`                       |
| SemDeDup & FairDeDup | `semantic_dedup/experiments/run_cifar10_experiment.py`                 |
| KNN-OOD              | `ood_detection/run_imagenet.py`, `ood_detection/run_cifar10_knnood.py` |
| KNN Prompting        | `knn_prompting/run_comparison.sh`                                      |
| SCIP                 | `code_pruning/run_e2e_ablation.sh`                                     |

### Exp 2: Ablation of IV1/IV2/IV3 (Table 4)

Individual and combined effects of each invariant.

| Algorithm            | Script                                      |
| -------------------- | ------------------------------------------- |
| CAL                  | `active_learning/run_ablation.sh`           |
| DEFT-UCS             | `coreset_selection/deftucs/run_ablation.py` |
| SemDeDup & FairDeDup | `semantic_dedup/scripts/run_ablation.sh`    |
| SCIP                 | `code_pruning/run_e2e_ablation.sh`          |

### Exp 3: Component-Level Breakdown (Table 5)

Breakdown timing fields are included in E2E / ablation script outputs.

### Exp 4: Execution Overhead (Table 6)

Build vs query time and peak cache size.

| Algorithm            | Script                                                                           |
| -------------------- | -------------------------------------------------------------------------------- |
| CAL                  | Natively in `active_learning/run_al.py` logs; cache in `measure_cache_memory.py` |
| CRAIG                | `coreset_selection/craig/measure_build_query_cache.py`                           |
| SemDeDup & FairDeDup | `semantic_dedup/measure_semdedup_memory.py`                                      |
| KNN-OOD              | Build/query time logged in `ood_detection/run_imagenet.py`                       |
| SCIP                 | `code_pruning/measure_build_query_cache.py`                                      |

### Exp 5: Cross-Setup Evaluation (Table 7)

In-memory, vector-database, and distributed setups.

| Algorithm            | Milvus                                          | Spark                     |
| -------------------- | ----------------------------------------------- | ------------------------- |
| CAL                  | `run_milvus_comparison.sh`                      | `run_spark_comparison.sh` |
| DEFT-UCS             | —                                               | —                         |
| CRAIG                | `run_craig_benchmark.py --backend milvus/spark` | same                      |
| SemDeDup & FairDeDup | `experiments/run_cifar10_milvus_experiment.py`  | `run_semdedup_spark.py`   |
| KNN-OOD              | `run_milvus_benchmark.py`                       | `run_spark_comparison.sh` |
| KNN Prompting        | `run_milvus_comparison.sh`                      | `run_spark_comparison.sh` |
| SCIP                 | `run_milvus_simple.py`                          | `run_spark_fixed.py`      |

### Exp 6: Memory Constraint (Figure 3)

Runtime under constrained memory budgets (CRAIG).

| Script                                                    | Description               |
| --------------------------------------------------------- | ------------------------- |
| `coreset_selection/craig/run_memory_experiment.py`        | NearPy-based memory sweep |
| `coreset_selection/craig/run_faiss_memory_experiment.py`  | FAISS-based memory sweep  |
| `coreset_selection/craig/run_memory_experiment_v2.py`     | Extended version          |
| `coreset_selection/craig/run_memory_experiment_1class.py` | Per-class runtime         |

### Exp 7: Dataset Size Ratio (Figure 4 top)

Selection recall, task performance, and runtime ratio under varying dataset sizes.

| Algorithm            | Script                                                |
| -------------------- | ----------------------------------------------------- |
| CAL                  | `active_learning/run_imdb_scaling.sh`                 |
| DEFT-UCS             | `coreset_selection/deftucs/run_scaling_experiment.py` |
| SemDeDup & FairDeDup | `semantic_dedup/run_scaling_experiment.py`            |
| KNN-OOD              | `ood_detection/run_scaling.sh`                        |
| SCIP                 | `code_pruning/run_scip_scaling.py`                    |

### Exp 8: Cache Capacity Sensitivity (Figure 4 bottom-right)

Runtime ratio under varying LRU cache sizes for 5 algorithms with IV2 (reuse).

| Script                                         | Description                                     |
| ---------------------------------------------- | ----------------------------------------------- |
| `active_learning/run_cache_benchmark_ratio.py` | Ratio-based cache sweep (covers ALL algorithms) |

### Exp 9: ANN Hyper-Parameter Sweep (Figure 5)

Selection recall, task performance, and search time under nlist/nprobe sweeps.

| Algorithm            | Script                                                    |
| -------------------- | --------------------------------------------------------- |
| CAL                  | `active_learning/run_ann_hyperparam_cal.py`               |
| CRAIG                | `coreset_selection/craig/run_ann_hyperparam_craig.py`     |
| DEFT-UCS             | `coreset_selection/deftucs/run_ann_hyperparam_coreset.py` |
| SemDeDup & FairDeDup | `semantic_dedup/run_ann_hyperparameter_study.py`          |
| KNN-OOD              | `ood_detection/run_ann_hyperparam_ood.py`                 |
| SCIP                 | `code_pruning/run_ann_hyperparam_codepruning.py`          |

### Plotting Scripts

Cross-algorithm plotting scripts are in [`draw/`](draw/):

| Script                             | Experiment                       |
| ---------------------------------- | -------------------------------- |
| `plot_ann_hyperparam_multitask.py` | Exp 9: ANN sweep (Figure 5)      |
| `plot_ann_sweep_separate_v3.py`    | Exp 9: ANN sweep (per-algorithm) |
| `plot_cache_benchmark_multi.py`    | Exp 8: Cache capacity (Figure 4) |
| `plot_cache_benchmark_ratio.py`    | Exp 8: Cache ratio               |
| `plot_scaling_results.py`          | Exp 7: Dataset size (Figure 4)   |

## Repository Structure

```
ai-becomes-db/
├── README.md
├── coreset_selection/
│   ├── craig/         # CRAIG: IV1 (ANN) + IV2 (Residual Reuse)
│   └── deftucs/       # DEFT-UCS: IV1 + IV2 (Distance Reuse) + IV3 (Top-k)
├── active_learning/   # CAL: IV1 (ANN) + IV2 (Probability Reuse)
├── semantic_dedup/    # SemDeDup & FairDeDup: IV1 (ANN) + IV2 (Search Results Reuse)
├── ood_detection/     # KNN-OOD: IV1 (ANN with GPU)
├── knn_prompting/     # KNN Prompting: IV2 (Intermediate Results Reuse)
├── code_pruning/      # SCIP: IV1 + IV2 (Distance Reuse) + IV3 (Top-k)
└── draw/              # Cross-algorithm plotting scripts
```

## Datasets

| Algorithm            | Dataset                      | Acquisition                                 |
| -------------------- | ---------------------------- | ------------------------------------------- |
| CRAIG                | MNIST, Fashion-MNIST         | Auto-download (`tensorflow.keras.datasets`) |
| DEFT-UCS             | CoEDIT, WikiLarge            | `python prepare_data.py` (auto-download)    |
| CAL                  | SST-2, IMDB                  | `bash get_data.sh`                          |
| SemDeDup / FairDeDup | CIFAR-10, CIFAR-100          | Auto-download (`torchvision`)               |
| KNN-OOD              | ImageNet-1K, CIFAR-10        | ImageNet: manual; CIFAR-10: auto            |
| KNN Prompting        | SST-2, AGNews                | Included in `data/` or HuggingFace          |
| SCIP                 | CodeSearchNet (Python, Java) | Auto-download (`datasets` library)          |

## Hardware

All experiments are conducted on a machine with 64 vCPUs, 500 GB memory, and 8× RTX 6000 GPUs. Results are reported as mean ± std over 3 independent runs.
