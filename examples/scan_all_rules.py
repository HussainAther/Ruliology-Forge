import csv
from pathlib import Path

from ruliology_forge import scan_rules


def main():
    rows = scan_rules(range(256), width=201, steps=200, perturb_time=80, perturb_radius=5)
    output = Path("results/eca_scan.csv")
    output.parent.mkdir(exist_ok=True)

    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    top = sorted(rows, key=lambda r: r["restoration_coefficient"], reverse=True)[:10]
    print("Top 10 rules by Restoration Coefficient:")
    for row in top:
        print(row)


if __name__ == "__main__":
    main()
