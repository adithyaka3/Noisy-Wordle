# Noisy Wordle

Most of us are familiar with the mechanics of Wordle, but what happens if we introduce uncertainty into its core feedback loop? Rather than receiving deterministic, absolute feedback for a particular guess, imagine receiving probabilistic, completely noisy feedback instead. This raises a fascinating mathematical challenge: how can a solver navigate this variant, and more fundamentally, is it even provably solvable? 

This project explores this problem by applying mathematical models and computationally efficient search algorithms to evaluate how well different strategies can solve this noisy variant of Wordle.

### The Noise Model

The type of noise we model is defined as follows:

Let the true, deterministic feedback for a given letter be denoted by $f^*$ (where $f^* \in \{\text{Gray}, \text{Yellow}, \text{Green}\}$). The observed noisy feedback $f$ presented to the algorithm is strictly sampled from the following probability distribution:
- $P(f = f^*) = 0.6$ *(60% chance of showing the correct, true Wordle output)*
- $P(f = f_1) = 0.2$ *(20% chance of showing one of the other two incorrect colors)*
- $P(f = f_2) = 0.2$ *(20% chance of showing the remaining incorrect color)*

### The Dilemma of Convergence

In a conventional game of Wordle, the session conclusively ends the moment a player receives an all-green sequence. However, in this noisy variant, receiving five green tiles does not guarantee that the correct target word has been guessed—it could merely be a deceptive output born from symmetrical noise. This introduces a significant dilemma: the solver must mathematically deduce when its confidence threshold is high enough to officially stop guessing.

While framed as a game, the underlying statistical mechanics directly mirror critical real-world problems. For instance, in fault detection systems or medical diagnostics, physical probes often yield non-deterministic results. The root fault must still be isolated with high statistical confidence using as few sequential diagnostic tests as possible.

### Evaluated Strategies

To tackle this challenge, this framework benchmarks several sophisticated, non-trivial search methods:
1. **Wald's SPRT (Sequential Probability Ratio Test):** Iteratively tracks bounds tracking hypotheses based on sequential likelihood updates.
2. **Thompson Sampling:** A probabilistic multi-armed bandit approach that samples dynamically from the belief state.
3. **Greedy 1-Step Lookahead Likelihood:** Greedily selects the guess that promises the highest immediate mathematical information gain.
4. **POMCP (Partially Observable Monte-Carlo Planning):** Evaluates a deep search tree using Monte-Carlo simulations to navigate the partially observable state space.

## Key Features

*   **Algorithmic Benchmarking**: Dynamically test advanced search heuristics against huge dictionaries:
    *   **POMCP**: Partially Observable Monte-Carlo Planning.
    *   **Greedy Parallel Trie**: Map-Reduce Likelihood (LLR) accumulation traversing a dynamically built prefix-tree.
    *   **Thompson Sampling**: Multi-armed bandit strategies scaling belief states.
    *   **GPU-Unique Bins**: `numba` CUDA-accelerated hardware entropy calculators.
*   **Interactive Web Visualizer**: Includes a fully interactive, lightweight local Web Server (`animate.py`) built directly on the Python Standard Library to visualize the physics tracking and contender logs in real-time.
*   **Dataset Tooling**: Scripts to dynamically build hyper-common frequency-mapped vocabularies or totally random `n`-length constraints against standard dictionary corpuses.

## Outputs, Results, Insights

**TBA**

## Repository Structure

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

## How to Run

You can explore the solvers through either granular graphical analysis or high-level statistical benchmarking:

**Query-by-Query Analysis (Web App)**  
Use the local web server to visually analyze the *Greedy Parallel* strategy step-by-step. This mode lets you dynamically input any 5 or 6-letter target word and watch the algorithm mathematically crunch the log-likelihoods for top contenders in real-time.
```bash
python animate.py
```
> The server hosts the interactive UI at `http://127.0.0.1:8000`.

**Statistical Overview (Benchmarker)**  
Use the main benchmarking file to see a complete structural overview of all the methods' results across a dataset. This mode runs multiple iterations silently and returns aggregated statistics, success rates, average time taken, and total turns.
```bash
python game.py
```

## Environment Setup
The tracking scripts rely physically on NumPy and Numba strictly for parallel processing and JIT compilation.
```bash
pip install -r requirements.txt
```