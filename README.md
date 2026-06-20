# Paradigm Swarm — Reproducible Experiments

## Quick Start
```bash
cd experiments

# Core experiments (<10s each, CPU-only, NumPy):
python3 paradigm_swarm_experiment.py       # §4.2  Weight Isolation (0% forgetting)
python3 paradigm_swarm_shift.py            # §4.3  Boundary Shift (0% vs 10.1%)
python3 paradigm_swarm_gap_strong.py       # §4.4  Gap Detection (100%)
python3 paradigm_swarm_distortion.py       # §4.5  Distortion Accumulation
python3 paradigm_swarm_benchmark.py        # §4.6  7-method Benchmark (PS=Oracle)
python3 paradigm_swarm_clean.py            # §4.12 Adversarial Sweep + multi-seed
python3 paradigm_swarm_ecosystem.py        # §4.13 Situational Responsibility
python3 paradigm_swarm_qstrust.py          # §4.14 Query-Specific Trust (+29pp)
python3 paradigm_swarm_100tasks.py         # §4.15 100-Task Forgetting + multi-seed
python3 paradigm_swarm_final2.py           # §4.16 Strong+Weak vs SGD (+44pp)
python3 paradigm_swarm_50tasks.py          # §4.10 50-Task Scaling + multi-seed
python3 paradigm_swarm_cross.py            # §4.11 Cross-Paradigm Dependency

# Routing experiments:
python3 paradigm_swarm_router_free.py      # §4.17 Router-Free (75%, no API)
python3 paradigm_swarm_trained_router.py   # §4.17 Trained Router (negative: ~22%)
python3 paradigm_swarm_router_llm.py       # §4.7  LLM Router (needs API key or dry-run)
python3 paradigm_swarm_routing.py          # §4.18 End-to-End Routing (density=100%)

# Requires external datasets (MNIST/CIFAR .npz in /tmp/):
# python3 paradigm_swarm_fair.py           # §4.8  Split-MNIST
# python3 paradigm_swarm_cifar.py          # §4.9  Split-CIFAR

# Figures:
python3 generate_figures.py                # Figures 1-6
python3 generate_figures_extra.py          # Figures 7-9
```

## Key Results (all reproducible)

| Section | Experiment | Key Metric |
|---------|-----------|------------|
| §4.2 | paradigm_swarm_experiment.py | PS 0% forgetting, SGD +45pp |
| §4.3 | paradigm_swarm_shift.py | PS 0.0% shift, SGD 10.1% |
| §4.4 | paradigm_swarm_gap_strong.py | PS 100% gap, SGD 90% overconfident |
| §4.5 | paradigm_swarm_distortion.py | P3: SGD 0.000, PS 0.960 |
| §4.6 | paradigm_swarm_benchmark.py | PS=0.861, Oracle=0.863 |
| §4.10 | paradigm_swarm_50tasks.py | PS=0.909±0.003, SGD=0.522±0.006 |
| §4.11 | paradigm_swarm_cross.py | Transfer +1.5pp over pure isolation |
| §4.12 | paradigm_swarm_clean.py | SGD degrades -5.1pp, PS stable, 20/20 feat |
| §4.13 | paradigm_swarm_ecosystem.py | Adversarial tasks self-segregate |
| §4.14 | paradigm_swarm_qstrust.py | +29pp on uncertain queries |
| §4.15 | paradigm_swarm_100tasks.py | PS=0.771±0.008, SGD=0.540±0.015 |
| §4.16 | paradigm_swarm_final2.py | PS=0.826, SGD=0.386, gap +44pp |
| §4.17 | paradigm_swarm_router_free.py | 10/10 class, 2/2 gap, 12/16 overall (75%) |
| §4.17 | paradigm_swarm_trained_router.py | ~22% accuracy (negative result) |
| §4.18 | paradigm_swarm_routing.py | Density router 100%, PS+SGD gap +0.190 |

## LLM Router (§4.7)

```bash
python3 paradigm_swarm_router_llm.py
```

Three modes (auto-detected):
1. **Dry-run** (default, no API key) — shows expected results from paper (96%)
2. **OpenRouter** (free tier) — `echo 'sk-or-v1-...' > ~/.ps_router_key`
3. **DeepSeek** — `echo 'sk-...' > ~/.ps_router_key`
4. **Ollama** (local) — `ollama pull llama3.2:3b && ollama serve`

Get a free OpenRouter key: https://openrouter.ai/keys

## Paper
- `paper/paradigm_swarm_paper.md` — Full paper (§1-§6, 19 refs)
- `paper/paradigm_swarm_findings.md` — Research document

## Figures
- `figures/fig1-10.png` — All paper figures

## Requirements
- Python 3.11+, NumPy
- matplotlib (figures only)
- No GPU — all synthetic experiments run on CPU in <10s
- MNIST/CIFAR experiments require pre-downloaded .npz files in /tmp/
