# Optimisation-Experiment 
CMA-ES vs Genetic Algorithm vs Gradient Descent on the 2-D Rastrigin Function

This guide shows you **exactly** how to install, configure, and run the project.

---

## 1  Prerequisites

| Tool | Minimum version | Test command |
|------|-----------------|--------------|
| **Python** | 3.9 – 3.12 | `python --version` |
| **git** (optional) | any recent | `git --version` |

> **Linux / macOS**: ensure `python3` is the interpreter you’ll use below.

---
## 2 Clone repo from github

```bash

git clone https://github.com/Tarequzzaman/es-vs-ga-gradient.git

cd es-vs-ga-gradient  

```

## 3  Create & activate a virtual environment

```bash

# (B) Make a venv called .venv/
python -m venv .venv

# (C) Activate it
# ─ macOS / Linux
source .venv/bin/activate

```

## 4 Install dependencies
```bash
pip install -r requirements.txt

```

## 5  Run the project

The entry point is **`main.py`**.  
All functionality is controlled via command-line flags:

```bash
python main.py [flags]
```
### Common examples of flags

| Purpose                             | Command                                                                                                         |
| ----------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| **Quick stats** – 20 runs, no plots | `python main.py`                                                                                                |
| 50 independent runs                 | `python main.py --runs 50`                                                                                      |
| Visual demo of first 8 generations  | `python main.py --plots 8`                                                                                      |
| Tune CMA-ES: σ = 0.8, λ = 100       | `python main.py --cma_sigma 0.8 --cma_lambda 100`                                                               |
| Larger GA population (100)          | `python main.py --ga_pop 100`                                                                                   |
| Faster GD (η = 0.02)                | `python main.py --gd_lr 0.02`                                                                                   |
| Double iteration budget             | `python main.py --max_iter 2000`                                                                                |
| **Full custom run**                 | `python main.py --runs 40 --plots 12 --cma_sigma 0.6 --cma_lambda 80 --ga_pop 80 --gd_lr 0.015 --max_iter 1500` |

### Default values 

| Flag                  | Default  | Meaning                                                       |
| --------------------- | -------- | ------------------------------------------------------------- |
| `--runs` *INT*        | **20**   | Number of random seeds (independent runs)                     |
| `--plots` *INT*       | **0**    | Show 3-D wire-frame of *N* generations (0 = no plots)         |
| `--cma_sigma` *FLOAT* | **0.5**  | Initial step-size σ for CMA-ES                                |
| `--cma_lambda` *INT*  | **50**   | Population size λ for CMA-ES                                  |
| `--ga_pop` *INT*      | **50**   | Population size for GA                                        |
| `--gd_lr` *FLOAT*     | **0.01** | Learning rate for Gradient Descent                            |
| `--max_iter` *INT*    | **1000** | Max generations/iterations for CMA-ES & GA (GD uses 2 × this) |


## 5 Deactivate the venv

```bash
deactivate
```
