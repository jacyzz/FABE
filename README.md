# FABE Project Documentation

## 1. Project Overview

**FABE (Framework for Adversarial Backdoor Extraction)** is a comprehensive MLOps pipeline designed for the research and development of robust AI models, specifically in the domain of code intelligence. The project's primary goal is to create models that can understand, refactor, and secure source code while being resilient to potential backdoors or stylistic attacks.

The project is organized into four main components, each serving a distinct purpose in the pipeline:

-   **`IST` (Implicit Style Transformer)**: The data generation engine. It takes clean source code and programmatically applies a variety of "good" and "bad" transformations to create complex, ranked preference datasets.
-   **`PRO` (Preference Ranking Optimization)**: A training framework that uses a DPO-like (Direct Preference Optimization) method. It trains a model to understand and rank different versions of code, learning to prefer "clean" and "correct" code over "dirty" or "buggy" versions.
-   **`Tuna`**: An alternative training framework that uses a pairwise margin ranking loss. It provides a different methodology for training preference models and serves as a valuable point of comparison with `PRO`.
-   **`inference`**: A unified, production-ready module for running batch inference using the models trained by either `PRO` or `Tuna`.

This document details the architecture and workflow of the FABE project as of our latest refactoring efforts.

---

## 2. The Data Pipeline: `IST`

The foundation of the FABE project is its sophisticated data generation pipeline, powered by the `IST` module.

### 2.1. Core Logic and Purpose

The `IST` module is responsible for creating the rich, ranked datasets required for preference tuning. Its core purpose is to transform a simple piece of source code into a complex learning signal for the model.

The process, orchestrated by `universal_data_transformer.py`, is as follows:

1.  **Input**: The script takes a "clean" piece of code (e.g., `func1` from the `clone-detect` dataset).
2.  **Dirty Prefix Generation**: It first creates a "dirty" version of the code by applying a random combination of negative transformations, such as inserting dead code or obfuscating variable names. This dirty version becomes the `input` for the language model, simulating a real-world scenario where the model must clean up messy code.
3.  **Ranked Output Generation**: The script then generates a ranked list of `output` versions, from best to worst:
    *   **Rank 1 (Best)**: The original, clean source code.
    *   **Rank 2**: A semantically equivalent version with standardized variable names.
    *   **Rank 3**: An alternative but correct implementation (e.g., from `func2` in the dataset or a version with altered control flow).
    *   **Rank 4 (Worst)**: A partially dirty version, containing a single "bad" transformation.
4.  **Reward Assignment**: A corresponding list of `score` values is assigned to the outputs, with higher scores for better ranks.

### 2.2. Dataset Format Transformation

A key outcome of our work was the standardization of the data format.

-   **Before**: The data generation process was tightly coupled with the training projects, creating model-specific formats (e.g., with hardcoded `<|prompter|>` tokens).
-   **After**: The `IST` pipeline now produces a single, **universal dataset format**. This format is model-agnostic and highly flexible.

**Universal `.jsonl` Format Example:**

```json
{
  "id": "some_unique_id",
  "instruction": "Please refactor the following code to improve its structure and style...",
  "input": "(Dirty Code Snippet)",
  "output": [
    "(Clean Code - Rank 1)",
    "(Code with Standardized Names - Rank 2)",
    "(Alternative Correct Code - Rank 3)",
    "(Partially Dirty Code - Rank 4)"
  ],
  "score": [3.0, 1.5, 0.5, -1.0]
}
```

### 2.3. How to Generate Data

A dedicated script, `FABE/IST/sh/generate_clone_data.sh`, automates this process for the `clone-detect` dataset. It correctly configures all paths and parameters to process the entire dataset.

---

## 3. Training Frameworks: `PRO` and `Tuna`

With a universal data format, we can now feed the same dataset into two different training frameworks.

### 3.1. `PRO` Training Process

The `PRO` project was heavily refactored for modularity and flexibility.

-   **Data Handling**: The `Coding_DataManager` in `utils/data_manager.py` is designed to read the universal format. It iterates through the `output` and `score` lists to create preference pairs for training.
-   **Template System**: A critical improvement was externalizing the prompt templates into `utils/templates.py`. The training script now uses the `--model_template` argument (e.g., `--model_template deepseek`) to dynamically select and apply the correct prompt format at runtime. This decouples the data from the model architecture.
-   **Efficient Fine-Tuning**: The framework fully supports LoRA and 4-bit quantization, configured via command-line arguments in the training script (`train_clone_detect.sh`).

### 3.2. `Tuna` Training Process

The `Tuna` project was adapted to be compatible with the new pipeline.

-   **Data Handling**: The `SupervisedDataset` in `train_tuna.py` was confirmed to be compatible with the universal data format. We modified its `DataArguments` to accept multiple input files, allowing it to consume the entire sharded dataset generated by `IST`.
-   **Template System**: `Tuna` uses its own internal, robust template system, selected via the `--chat_template` argument. This system was already compatible with our goals and required no changes.
-   **Efficient Fine-Tuning**: `Tuna` also has built-in support for LoRA and QLoRA, configured via the `--peft` argument.

---

## 4. Unified Inference

To complete the pipeline, we created a single, powerful script for inference.

### 4.1. `batch_inference.py`

Located in `FABE/inference/`, this script is designed to be the unified endpoint for any model trained within the FABE ecosystem.

-   **Model Loading**: It automatically loads a base model and merges the trained LoRA adapter weights, creating a production-ready model for maximum inference speed.
-   **Dynamic Templates**: It re-uses the same `templates.py` module from the `PRO` project, ensuring that the prompt format used during inference is identical to the one used during training.
-   **Batch Processing**: The script is optimized for high-throughput inference, processing data in configurable batches.
-   **Usage**: It is controlled via command-line arguments, requiring paths to the base model, the LoRA adapter, and the input/output files.

---

## 5. Summary of Work Completed

This project involved a significant architectural refactoring to create a robust and scalable MLOps workflow. Key achievements include:

1.  **Decoupled Data Pipeline**: We successfully separated data generation from training by creating a universal, model-agnostic dataset format.
2.  **Refactored `PRO` Project**: Overhauled the `PRO` framework to use a dynamic template system, making it adaptable to any model architecture.
3.  **Adapted `Tuna` Project**: Ensured the `Tuna` framework was fully compatible with the new multi-file, universal dataset.
4.  **Unified Inference Script**: Built a single, efficient batch inference script that can serve models from either training pipeline.
5.  **End-to-End Workflow**: Created a complete set of shell scripts (`generate_clone_data.sh`, `train_clone_detect.sh` for both `PRO` and `Tuna`) that automate the entire process from raw data to a trained model.
6.  **Bug Fixes & Verification**: Conducted a thorough review of all components, identified and resolved several issues (such as argument parsing and dataset limitations), and verified the correctness of the entire pipeline.

The FABE project is now a mature, flexible, and powerful framework for state-of-the-art research in code intelligence.

