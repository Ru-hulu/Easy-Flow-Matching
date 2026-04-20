# Flow Matching toy reproduction

This is a minimal NumPy-only reproduction of the central idea in Lipman et al.,
**Flow Matching for Generative Modeling**. It trains a continuous vector field
that transports a simple distribution, `N(0, I)`, into a complex 2D mixture.

The example intentionally avoids image generation so the math stays visible.

## Paper-to-code mapping
"Flow Matching for Generative Modeling"
section 4.1 example2
## Run

```bash
python3 toy_flow_matching.py --steps 2500
```

For a faster smoke test:

```bash
python3 toy_flow_matching.py --steps 300 --samples 1000
```

Or use the bundled runner:

```bash
./run.sh smoke
./run.sh default
./run.sh long
```

If your default `python3` does not have NumPy installed, either install the
dependency or point the runner at another Python:

```bash
python3 -m pip install -r requirements.txt
PYTHON_BIN=/path/to/python3 ./run.sh smoke
```

Outputs are written under `outputs/`:

- `loss.csv`: training loss over time.
- `samples.npz`: source noise, target samples, generated samples, and path snapshots.
- `flow_matching_toy.svg`: visual comparison of source, target, generated samples, and learned trajectory snapshots.

## What success looks like

The generated points should move from a standard Gaussian cloud into the same
multi-modal checkerboard/ring-like structure as the target distribution. With
more training steps, the generated sample gets sharper and mode coverage
improves.
