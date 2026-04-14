# Noisy Wordle Solver Framework

A comprehensive Python framework for benchmarking and visualizing autonomous stochastic search algorithms against a noisy variant of Wordle. Instead of standard deterministic feedback, the solver must intelligently navigate symmetrical noise models using Sequential Probability Ratio Tests (SPRT), POMCP, and hardware-accelerated Branch & Bound logic.

## 🚀 Key Features

*   **Algorithmic Benchmarking**: Dynamically test advanced search heuristics against huge dictionaries:
    *   **POMCP**: Partially Observable Monte-Carlo Planning.
    *   **Greedy Parallel Trie**: Map-Reduce Likelihood (LLR) accumulation traversing a dynamically built prefix-tree.
    *   **Thompson Sampling**: Multi-armed bandit strategies scaling belief states.
    *   **GPU-Unique Bins**: `numba` CUDA-accelerated hardware entropy calculators.
*   **Interactive Web Visualizer**: Includes a fully interactive, lightweight local Web Server (`animate.py`) built directly on the Python Standard Library to visualize the physics tracking and contender logs in real-time.
*   **Dataset Tooling**: Scripts to dynamically build hyper-common frequency-mapped vocabularies or totally random `n`-length constraints against standard dictionary corpuses.

## 📁 Repository Structure

```text
├── animate.py                  # Local Web Server for Interactive Visualizations
├── game.py                     # Main terminal-based Benchmarking Suite
├── create_small_datasets.py    # Auto-generates frequency-based testing vocabularies
├── datasets/                   # Generated vocabulary data (english_small_5.json, etc.)
└── strategies/                 # Core Solver Architecture Modules
    ├── pomcp.py
    ├── sprt_greedyLL_parallel_trie.py
    ├── sprt_thompson.py
    └── sprt_unique_gpu.py
```

## 🎮 How to Run 

### 1. The Interactive Visualizer (WebApp)
To visually trace how the AI converges on the target bounds under heavy symmetrical noise, spin up the animation server:
```bash
python animate.py
```
> The server will host a clean Light-Mode UI at `http://127.0.0.1:8000`. You can dynamically input any 5 or 6-letter target word and watch the solver mathematically crunch the top contenders!

### 2. The Headless Benchmarker
To cleanly track the average turns taken by the parallelized strategies across large baseline datasets:
```bash
python game.py
```

## ⚙️ Environment Setup
The tracking scripts rely physically on NumPy and Numba strictly for parallel processing and JIT compilation.
```bash
pip install -r requirements.txt
```