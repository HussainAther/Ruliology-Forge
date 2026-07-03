from ruliology_forge import run_perturbation_experiment, plot_divergence, plot_trajectory


def main():
    result = run_perturbation_experiment(
        rule=110,
        width=201,
        steps=200,
        perturb_time=80,
        perturb_radius=5,
        perturbation="bit_flip",
    )

    print(f"Rule: {result.rule}")
    print(f"Restoration coefficient: {result.restoration_coefficient:.4f}")
    print(f"Final restoration: {result.final_restoration:.4f}")

    plot_trajectory(result.control, title="Rule 110 control")
    plot_trajectory(result.difference, title="Rule 110 XOR difference")
    plot_divergence(result.divergence)


if __name__ == "__main__":
    main()
