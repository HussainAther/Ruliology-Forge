# Ruliology-Forge: A Toolkit for Forensic Morphogenesis

**Ruliology-Forge** is an open-source framework designed to measure, simulate, and visualize the resilience of computational patterns. Based on the manuscript *"Ruliological Resilience: Pattern Restoration and Robustness in Wolfram Patterns"* (Ather & Gordon, 2026), this toolkit provides the mathematical and computational bridge between Wolfram’s Elementary Cellular Automata (ECA) and biological regeneration.

## 🌟 Key Features

* **Restoration Coefficient ($R$) Engine:** Quantify the "healing" capacity of any ECA rule using Boolean XOR difference mapping.
* **Perturbation Architect:** Systematically introduce "injuries" (localized entropy) at specific temporal horizons ($T$) to test pattern robustness.
* **The Forensic Library:** A digital catalog of 200 *Conus* specimens mapped to their ruliological analogs (Rules 110, 54, 232, and 30).
* **Blender Integration (Beta):** Procedural Geometry Node setups to "grow" 3D shells using ruliological logic.

## 🧪 Scientific Foundation

Traditional ruliology focuses on growth. **Ruliology-Forge** focuses on *recovery*.

By mapping the "Differentiation Code" (Gordon & Gordon, 2019) to computational resilience, this software allows researchers to visualize the **Ruliological Scar**—the moment where a system (like Rule 110) prioritizes structural logic over spatial coordinates.

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/HussainAther/Ruliology.git
cd Ruliology
pip install -r requirements.txt

```

### Measuring Resilience ($R$)

Calculate the restoration score for Rule 110 with a 5-cell injury at step 50:

```python
from ruliology_forge import ECAEngine

rule = 110
engine = ECAEngine(rule)
r_score = engine.calculate_resilience(injury_size=5, horizon=50)

print(f"Restoration Coefficient for Rule {rule}: {r_score}")

```

## 📂 Project Structure

* `/core`: The Python engine for ECA simulation and XOR difference mapping.
* `/forensics`: High-resolution data and indices for the *Conus* shell library.
* `/blender`: `.blend` templates and scripts for ruliological displacement mapping.
* `/notebooks`: Jupyter tutorials on calculating the **Perturbation Horizon**.

## 🤝 Contributing

We welcome contributions from computational physicists, theoretical biologists, and 3D artists.

1. Fork the Project.
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`).
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the Branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

## 📄 Citation

If you use this software in your research, please cite:

> Ather, S. H., & Gordon, R. (2026). *Ruliological Resilience: Pattern Restoration and Robustness in Wolfram Patterns*. Biosystems (In Review).

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---

**Maintained by [Hussain Ather**](https://github.com/HussainAther) *Part of the Janus Sphere Innovations Research Initiative*

