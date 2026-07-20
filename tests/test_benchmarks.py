from ruliology_forge.benchmarks import (
    BenchmarkScenario,
    BenchmarkSuite,
    benchmark_markdown,
    benchmark_rules,
    rank_benchmark_rows,
)


def test_suite_fingerprint_is_stable_and_sensitive():
    a = BenchmarkSuite("tiny", (BenchmarkScenario("one", perturb_time=5),))
    b = BenchmarkSuite("tiny", (BenchmarkScenario("one", perturb_time=5),))
    c = BenchmarkSuite("tiny", (BenchmarkScenario("one", perturb_time=6),))
    assert a.fingerprint() == b.fingerprint()
    assert a.fingerprint() != c.fingerprint()


def test_rank_benchmark_rows_orders_stronger_rule_first():
    rows = []
    for rule, restoration, recovered in [(1, 0.9, True), (2, 0.3, False)]:
        for scenario in ("a", "b"):
            rows.append(
                {
                    "rule": rule,
                    "scenario": scenario,
                    "restoration_coefficient": restoration,
                    "recovery_time": 1 if recovered else None,
                    "recovered": recovered,
                    "shift_tolerant_restoration": restoration,
                }
            )
    ranking = rank_benchmark_rows(rows)
    assert [row["rule"] for row in ranking] == [1, 2]
    assert ranking[0]["rank"] == 1


def test_tiny_benchmark_runs_and_writes_report_shape():
    suite = BenchmarkSuite(
        "tiny",
        (
            BenchmarkScenario("flip", perturb_time=5, perturb_radius=1),
            BenchmarkScenario("void", perturbation="void", perturb_time=6, perturb_radius=1),
        ),
    )
    raw, ranking = benchmark_rules(
        [0, 90], suite=suite, repeats=2, width=21, steps=15, seed=7
    )
    assert len(raw) == 8
    assert len(ranking) == 2
    report = benchmark_markdown(ranking, suite=suite)
    assert "| Rank | Rule |" in report
    assert "Suite fingerprint" in report
