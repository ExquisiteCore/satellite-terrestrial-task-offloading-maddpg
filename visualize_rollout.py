from __future__ import annotations

import argparse
from pathlib import Path

from config import MODELS_DIR, VISUALIZATION_DIR
from visualization.trace import generate_rollout_trace, write_rollout_trace


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate multi-policy rollout trace data for visualization.")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cpu")
    parser.add_argument("--no-load-models", action="store_true")
    parser.add_argument("--dqn-checkpoint", type=Path, default=MODELS_DIR / "dqn.pt")
    parser.add_argument("--maddpg-checkpoint", type=Path, default=MODELS_DIR / "maddpg_best.pt")
    parser.add_argument("--output", type=Path, default=VISUALIZATION_DIR / "rollout_trace.json")
    args = parser.parse_args()

    trace = generate_rollout_trace(
        steps=args.steps,
        seed=args.seed,
        load_models=not args.no_load_models,
        device=args.device,
        dqn_checkpoint=args.dqn_checkpoint,
        maddpg_checkpoint=args.maddpg_checkpoint,
    )
    output_path = write_rollout_trace(trace, args.output)
    print(f"Wrote rollout trace to {output_path}")


if __name__ == "__main__":
    main()
