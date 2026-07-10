# Multi-Signal Compliance Audit Pipeline for RAG Systems

Companion code for *A Multi-Signal Compliance Audit Pipeline for RAG Systems*
(EMNLP 2026 System Demonstrations).

RAG and compliance-answering systems are commonly audited with a single
metric — cosine similarity or NLI argmax — both of which have demonstrable
blind spots. This project combines NLI entailment/contradiction scoring,
topical embedding similarity, and a supervised classifier targeting
precedent/exception framing (e.g., "historically, we've allowed
exceptions...") — a failure mode neither signal alone can resolve.

**Live demo:** [https://multi-signal-compliance-guard-qq3xu59gdprwtww69gcrsh.streamlit.app/]
**Paper:** [TODO: link once available, e.g. OpenReview / ACL Anthology]
**Video:** [TODO: Loom/YouTube link]

## What's in this repo

| File | Purpose |
|---|---|
| `streamlit_app.py` | Interactive demo (matches the paper's final v10 architecture) |
| `output.csv` | Hand-verified evaluation dataset (fit / held-out / adversarial splits) |
| `solutions_final.ipynb` | Full development notebook — every design iteration (v0–v10), including two discarded approaches, kept intact rather than trimmed to only the working version |
| `emnlp_paper.tex`, `custom.bib` | Paper source |
| `requirements.txt` | Python dependencies |

## Quickstart

```bash
git clone <this-repo-url>
cd <repo-name>
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The precedent classifier trains automatically at startup from `output.csv`
(the fit split only — held-out and adversarial rows are never used for
training or threshold selection).

## Reproducibility Notes

Synthetic dataset generation (notebook Appendix) is optional to run.
This section calls the Openai API to generate candidate evaluation
examples and requires your own OPENAI_API_KEY. You do not need to run
it: the dataset it produces, output.csv, is already included in this
repo, hand-verified and ready to use. Every downstream cell (v0–v10,
bootstrap significance testing, results tables) reads from output.csv
directly and does not depend on the generation cell having been executed.
If you don't have an API key, skip that section entirely — the notebook
still runs top-to-bottom without it.

Getting output.csv into your environment.


Cloning this repo (git clone <repo-url>) or opening the notebook
via Colab's File → Open notebook → GitHub option: output.csv is
already present alongside the notebook — no extra steps needed.
Running the notebook standalone (e.g., you downloaded only the
.ipynb file, not the full repo): upload output.csv manually before
running the evaluation cells:

```
  python  from google.colab import files
  files.upload()  # select output.csv when prompted
```



## Architecture

Each (policy, response) pair is evaluated through five priority-ordered
gates:

1. **Direct contradiction** (NLI contradiction > 0.35)
2. **Background-leak** (high-neutral + moderate contradiction)
3. **Precedent classifier** — a small logistic regression trained on
   response embeddings, catching exception/discretion framing that neither
   NLI nor similarity can separate from genuinely compliant text
4. **Topical similarity** (catches topic drift / deflection)
5. **Direct NLI margin** (fallback for non-neutral cases)

Full rationale, including two failed signal designs (a repurposed NLI probe
and zero-shot classification) that motivated the final supervised-classifier
approach, is documented in `multi-signal compliance guard.ipynb`

## Results

Evaluated on held-out and adversarial data never used for threshold
selection or classifier training:

| Category | n | Naive Baseline | Multi-Signal Guard |
|---|---|---|---|
| Safe | 14 | 100.0% | 100.0% |
| Direct Failure | 16 | 100.0% | 100.0% |
| Topic Drift | 16 | 6.2% | 93.8% |
| Precedent Framing | 16 | 0.0% | 75.0% |
| Adversarial | 15 | 100.0% | 100.0% |
| **Aggregate** | **77** | **59.8%** | **93.6%** ($p < 0.001$) |

See the paper's Limitations section for a full account of the four
remaining boundary-adjacent false negatives.

## Evaluation methodology

All thresholds and the classifier are fit **only** on the `fit` split.
Reported numbers come **only** from `held_out` and `adversarial` splits,
which were never used to select thresholds or train anything. This
separation exists because an earlier design in this project's history
(documented in the notebook) achieved a spurious 100% accuracy by evaluating
on the exact data used to build it — the fit/held-out/adversarial split
was introduced specifically to prevent that from happening again.

## Dataset generation

Evaluation examples were generated with LLM assistance and individually
hand-verified before use. See `solutions_final.ipynb`, Appendix, for the
full generation and verification protocol.

## Licensing

Code: [TODO: state license, e.g. MIT]
Models used (via `sentence-transformers`): `cross-encoder/nli-deberta-v3-base`,
`all-MiniLM-L6-v2` — both openly licensed on Hugging Face.

## Citation

```bibtex
[TODO: add once the paper has a formal citation / anthology entry]
```
