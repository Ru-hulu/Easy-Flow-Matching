# Flow Matching toy reproduction

This is a minimal NumPy-only reproduction of the central idea in Lipman et al.,
**Flow Matching for Generative Modeling**. It trains a continuous vector field
that transports a simple distribution, `N(0, I)`, into a complex 2D mixture.

The example intentionally avoids image generation so the math stays visible.

## Paper-to-code mapping

The paper defines a probability path `p_t(x)` and a vector field `u_t(x)` that
generates that path through an ODE:

```text
dx / dt = v_t(x)
```

Directly fitting the marginal vector field is hard, so the paper uses
Conditional Flow Matching:

```text
L_CFM = E_{t, x_1, x_t | x_1} || v_theta(t, x_t) - u_t(x_t | x_1) ||^2
```

This reproduction uses the paper's Optimal Transport conditional path:

```text
sigma_t = 1 - (1 - sigma_min) t
x_t     = sigma_t x_0 + t x_1
u_t     = x_1 - (1 - sigma_min) x_0
```

where:

- `x_0 ~ N(0, I)` is simple noise.
- `x_1 ~ q(x)` is a data sample from a 2D mixture distribution.
- `x_t` is the conditional path sample at time `t`.
- `u_t` is the target conditional velocity.
- `v_theta` is a small MLP trained to match `u_t`.

After training, generation starts from fresh Gaussian noise and integrates:

```text
dx / dt = v_theta(t, x)
```

from `t = 0` to `t = 1`.

## Run

```bash
python3 toy_flow_matching.py --steps 2500
```

For a faster smoke test:

```bash
python3 toy_flow_matching.py --steps 300 --samples 1000
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
