# Theoretical Foundations of Neural Networks and PINNs

This repository contains the LaTeX source code of the thesis
on the theoretical foundations of neural networks and
Physics-Informed Neural Networks (PINNs) for differential equations.

## Structure

- `main.tex` — Main LaTeX entry point
- `Sections/` — Thesis chapters
- `Images/` — Figures
- `references.bib` — Bibliography
- `preamble.tex` — LaTeX configuration
- `notation.tex` — Mathematical notation
- `titlepage.tex` — Title page

## AI transparency

Built with Claude (Anthropic) as an assistant: 12 of the 24 commits carry a Claude
co-author trailer. Scope:

- **The thesis is mine** — the theory, the proofs, and the argument it makes.
- **The AI-assisted commits are editing and reruns.** Compile errors, missing captions
  and bibliography entries, clearing overfull boxes, tightening several proofs and
  unifying notation in Chapters 2–3, and re-running experiments 3, 4 and 5 with the
  configurations the text specifies.
- **What is checked rather than asserted.** Each of the six experiments writes its
  results to a committed JSON, and `code/pinns/run_all.py` regenerates all of them from
  scratch, so any claim in the text can be re-derived rather than taken on trust.
