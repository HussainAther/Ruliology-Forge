"""Generate a Version 1 Houdini bundle for Rule 110."""

from pathlib import Path

from ruliology_forge import (
    export_houdini_bundle,
    run_perturbation_experiment,
)


def main() -> None:
    result = run_perturbation_experiment(
        rule=110,
        width=201,
        steps=200,
        perturb_time=80,
        perturb_center=100,
        perturb_radius=5,
        perturbation="bit_flip",
        perturbation_timing="after_update",
        initial_condition="random",
        initial_density=0.5,
        boundary="periodic",
        seed=42,
        recovery_threshold=0.01,
        recovery_persistence=5,
        max_shift=10,
    )

    bundle_dir = export_houdini_bundle(
        result,
        Path("results")
        / "houdini"
        / "rule110_bundle",
        overwrite=True,
    )

    print(f"Houdini bundle written to: {bundle_dir}")


if __name__ == "__main__":
    main()
