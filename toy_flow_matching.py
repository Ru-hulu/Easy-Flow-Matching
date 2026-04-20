#!/usr/bin/env python3
"""Minimal Flow Matching reproduction on a 2D toy distribution.

The implementation deliberately uses only NumPy. It trains a small MLP vector
field with the Conditional Flow Matching objective from Lipman et al. using the
Optimal Transport conditional path.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--sigma-min", type=float, default=0.05)
    parser.add_argument("--samples", type=int, default=2500)
    parser.add_argument("--ode-steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def sample_target(rng: np.random.Generator, n: int) -> np.ndarray:
    """A multi-modal 2D target distribution."""
    ring_angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    ring = np.stack([2.8 * np.cos(ring_angles), 2.8 * np.sin(ring_angles)], axis=1)
    grid_values = np.array([-1.6, 0.0, 1.6])
    grid = np.array([(x, y) for x in grid_values for y in grid_values])
    centers = np.concatenate([ring, grid], axis=0)

    idx = rng.integers(0, len(centers), size=n)
    noise_scale = np.where(idx < len(ring), 0.18, 0.12)[:, None]
    samples = centers[idx] + noise_scale * rng.standard_normal((n, 2))
    return samples.astype(np.float64)


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def silu_grad(x: np.ndarray) -> np.ndarray:
    sig = 1.0 / (1.0 + np.exp(-x))
    return sig * (1.0 + x * (1.0 - sig))


@dataclass
class MLP:
    params: dict[str, np.ndarray]

    @classmethod
    def create(cls, rng: np.random.Generator, hidden: int) -> "MLP":
        def init(in_dim: int, out_dim: int) -> np.ndarray:
            return rng.standard_normal((in_dim, out_dim)) * np.sqrt(2.0 / in_dim)

        params = {
            "w1": init(3, hidden),
            "b1": np.zeros(hidden),
            "w2": init(hidden, hidden),
            "b2": np.zeros(hidden),
            "w3": init(hidden, 2),
            "b3": np.zeros(2),
        }
        return cls(params)

    def forward(self, x: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        inp = np.concatenate([x, t], axis=1)
        z1 = inp @ self.params["w1"] + self.params["b1"]
        h1 = silu(z1)
        z2 = h1 @ self.params["w2"] + self.params["b2"]
        h2 = silu(z2)
        out = h2 @ self.params["w3"] + self.params["b3"]
        cache = {"inp": inp, "z1": z1, "h1": h1, "z2": z2, "h2": h2}
        return out, cache

    def backward(self, cache: dict[str, np.ndarray], grad_out: np.ndarray) -> dict[str, np.ndarray]:
        grads: dict[str, np.ndarray] = {}

        grads["w3"] = cache["h2"].T @ grad_out
        grads["b3"] = grad_out.sum(axis=0)
        grad_h2 = grad_out @ self.params["w3"].T

        grad_z2 = grad_h2 * silu_grad(cache["z2"])
        grads["w2"] = cache["h1"].T @ grad_z2
        grads["b2"] = grad_z2.sum(axis=0)
        grad_h1 = grad_z2 @ self.params["w2"].T

        grad_z1 = grad_h1 * silu_grad(cache["z1"])
        grads["w1"] = cache["inp"].T @ grad_z1
        grads["b1"] = grad_z1.sum(axis=0)

        return grads

    def predict(self, x: np.ndarray, t_scalar: float) -> np.ndarray:
        t = np.full((len(x), 1), t_scalar, dtype=np.float64)
        pred, _ = self.forward(x, t)
        return pred


class Adam:
    def __init__(self, params: dict[str, np.ndarray], lr: float):
        self.lr = lr
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.step_num = 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        self.step_num += 1
        for key in params:
            self.m[key] = self.beta1 * self.m[key] + (1.0 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1.0 - self.beta2) * (grads[key] ** 2)
            m_hat = self.m[key] / (1.0 - self.beta1**self.step_num)
            v_hat = self.v[key] / (1.0 - self.beta2**self.step_num)
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def flow_matching_batch(
    rng: np.random.Generator,
    batch_size: int,
    sigma_min: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x0 = rng.standard_normal((batch_size, 2))
    x1 = sample_target(rng, batch_size)
    t = rng.random((batch_size, 1))

    # Paper eq. (20): OT conditional path with linear mean and std.
    sigma_t = 1.0 - (1.0 - sigma_min) * t

    # Paper eq. (22): psi_t(x0) = sigma_t * x0 + t * x1.
    xt = sigma_t * x0 + t * x1

    # Paper eq. (23): u_t(psi_t(x0) | x1), obtained by substituting
    # eq. (22) into the conditional vector field in eq. (21).
    ut = x1 - (1.0 - sigma_min) * x0
    return xt, t, ut


def train(args: argparse.Namespace) -> tuple[MLP, list[tuple[int, float]]]:
    rng = np.random.default_rng(args.seed)
    model = MLP.create(rng, args.hidden)
    optim = Adam(model.params, args.lr)
    losses: list[tuple[int, float]] = []

    for step in range(1, args.steps + 1):
        xt, t, target_velocity = flow_matching_batch(rng, args.batch_size, args.sigma_min)
        pred, cache = model.forward(xt, t)
        diff = pred - target_velocity

        # Paper eq. (9): Conditional Flow Matching objective.
        loss = float(np.mean(np.sum(diff * diff, axis=1)))
        grad_out = (2.0 / args.batch_size) * diff
        grads = model.backward(cache, grad_out)
        optim.step(model.params, grads)

        if step == 1 or step % max(1, args.steps // 50) == 0:
            losses.append((step, loss))
            print(f"step={step:5d} loss={loss:.6f}")

    return model, losses


def integrate(model: MLP, x0: np.ndarray, ode_steps: int) -> tuple[np.ndarray, np.ndarray]:
    x = x0.copy()
    snapshots = [x.copy()]
    snapshot_steps = {ode_steps // 4, ode_steps // 2, (3 * ode_steps) // 4, ode_steps}
    dt = 1.0 / ode_steps

    for i in range(ode_steps):
        t = i * dt
        # Paper eqs. (1)-(2): integrate the learned CNF vector field.
        # Heun's method is still tiny, but much cleaner than Euler for learned flows.
        v0 = model.predict(x, t)
        proposal = x + dt * v0
        v1 = model.predict(proposal, min(1.0, t + dt))
        x = x + 0.5 * dt * (v0 + v1)
        if i + 1 in snapshot_steps:
            snapshots.append(x.copy())

    return x, np.stack(snapshots, axis=0)


def write_loss(path: Path, losses: list[tuple[int, float]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "loss"])
        writer.writerows(losses)


def svg_panel(points: np.ndarray, width: int, height: int, color: str, title: str) -> str:
    points = np.clip(points, -4.2, 4.2)
    sx = (points[:, 0] + 4.2) / 8.4 * width
    sy = height - (points[:, 1] + 4.2) / 8.4 * height
    circles = "\n".join(
        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="1.35" fill="{color}" fill-opacity="0.48" />'
        for x, y in zip(sx, sy)
    )
    return (
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#fbfaf7" />\n'
        f'<text x="12" y="24" font-size="17" font-family="Arial" fill="#222">{title}</text>\n'
        f"{circles}"
    )


def write_svg(path: Path, source: np.ndarray, target: np.ndarray, generated: np.ndarray, snapshots: np.ndarray) -> None:
    panel_w, panel_h = 310, 310
    gap = 18
    width = 3 * panel_w + 2 * gap
    height = 2 * panel_h + gap

    panels = [
        (0, 0, svg_panel(source, panel_w, panel_h, "#4f6fae", "source: N(0, I)")),
        (panel_w + gap, 0, svg_panel(target, panel_w, panel_h, "#c65f47", "target: q(x)")),
        (2 * (panel_w + gap), 0, svg_panel(generated, panel_w, panel_h, "#248b67", "generated: ODE result")),
    ]

    path_colors = ["#4f6fae", "#5f7f9d", "#718d8c", "#9d895f", "#248b67"]
    path_panel = ['<rect x="0" y="0" width="966" height="310" fill="#fbfaf7" />']
    path_panel.append('<text x="12" y="24" font-size="17" font-family="Arial" fill="#222">learned flow snapshots: t = 0, 0.25, 0.5, 0.75, 1</text>')
    for i, snap in enumerate(snapshots):
        pts = np.clip(snap[:700], -4.2, 4.2)
        sx = (pts[:, 0] + 4.2) / 8.4 * 966
        sy = 310 - (pts[:, 1] + 4.2) / 8.4 * 310
        for x, y in zip(sx, sy):
            path_panel.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{1.0 + 0.15 * i:.2f}" '
                f'fill="{path_colors[i]}" fill-opacity="0.24" />'
            )
    panels.append((0, panel_h + gap, "\n".join(path_panel)))

    body = []
    for x, y, panel in panels:
        body.append(f'<g transform="translate({x},{y})">{panel}</g>')
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        f'<rect width="100%" height="100%" fill="#eee9df" />\n'
        + "\n".join(body)
        + "\n</svg>\n"
    )
    path.write_text(svg)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model, losses = train(args)
    rng = np.random.default_rng(args.seed + 100)
    source = rng.standard_normal((args.samples, 2))
    target = sample_target(rng, args.samples)
    generated, snapshots = integrate(model, source, args.ode_steps)

    np.savez(
        args.out_dir / "samples.npz",
        source=source,
        target=target,
        generated=generated,
        snapshots=snapshots,
    )
    np.savez(args.out_dir / "model_params.npz", **model.params)
    write_loss(args.out_dir / "loss.csv", losses)
    write_svg(args.out_dir / "flow_matching_toy.svg", source, target, generated, snapshots)

    print(f"saved {args.out_dir / 'samples.npz'}")
    print(f"saved {args.out_dir / 'flow_matching_toy.svg'}")


if __name__ == "__main__":
    main()
