![conf_guard](https://github.com/user-attachments/assets/15dd0700-a917-4d1f-a7e8-a27199b7ec9c)

<div align="center">

<strong>[🌐Website](https://cleverhans.io/confidential_guardian/) • [📄Paper](https://cleverhans.io/confidential_guardian/) • [🖼️Poster](https://cleverhans.io/confidential_guardian/) • [📊 Slides](https://cleverhans.io/confidential_guardian/) • [🎬 Video](https://cleverhans.io/confidential_guardian/)</strong>

</div>

## 🧠 Abstract

Cautious predictions—where a machine learning model abstains when uncertain—are crucial for limiting harmful errors in safety-critical applications. In this work, we identify a novel threat: a dishonest institution can exploit these mechanisms to discriminate or unjustly deny services under the guise of uncertainty. We demonstrate the practicality of this threat by introducing an uncertainty-inducing attack called **Mirage**, which deliberately reduces confidence in targeted input regions, thereby covertly disadvantaging specific individuals. At the same time, Mirage maintains high predictive performance across all data points. To counter this threat, we propose **Confidential Guardian**, a framework that analyzes calibration metrics on a reference dataset to detect artificially suppressed confidence. Additionally, it employs zero-knowledge proofs of verified inference to ensure that reported confidence scores genuinely originate from the deployed model. This prevents the provider from fabricating arbitrary model confidence values while protecting the model’s proprietary details. Our results confirm that Confidential Guardian effectively prevents the misuse of cautious predictions, providing verifiable assurances that abstention reflects genuine model uncertainty rather than malicious intent.

## ⚙️ Installation with `uv`

We are using [`uv`](https://github.com/astral-sh/uv) as our package manager (and we think you should, too)! It is a fast Python dependency management tool and drop-in replacement for `pip`.

### Step 1: Install `uv` (if not already installed)

```bash
pip install uv
```

### Step 2: Install dependencies 

```bash
uv pip install -e .
```

### Step 3: Activate environment 

```bash
source .venv/bin/activate
```

### Step 4: Launch jupyter

```bash
jupyter notebook
```

## 🗂️ Codebase overview

- `mirage.py`: Contains code for the Mirage attack discussed in the paper.
- `conf_guard.py`: Contains code for computing calibration metrics and reliability diagrams.
- `gaussian_experiments.ipynb`: Notebook for the synthethic Gaussian experiments.
- `image_experiments.ipynb`: Notebook for the image experiments on CIFAR-100 and UTKFace.
- `tabular_experiments.ipynb`: Notebook for the tabular experiments on Adult and Credit.
- `regression_experiments.ipynb`: Notebook for the regression experiments.

## 🎓 BibTeX citation

```bibtex
@inproceedings{rabanser2025confidential,
  title = {Confidential Guardian: Cryptographically Prohibiting the Abuse of Model Abstention},
  author = {Stephan Rabanser and Ali Shahin Shamsabadi and Olive Franzese and Xiao Wang and Adrian Weller and Nicolas Papernot},
  year = {2025},
  booktitle = {},
}
```
