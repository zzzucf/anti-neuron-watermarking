# Anti-Neuron Watermarking — ECCV 2022

Official research code for **Anti-Neuron Watermarking: Protecting Personal Data Against Unauthorized Neural Networks**.

**[Zihang Zou](https://scholar.google.com/citations?user=GLuGAK0AAAAJ&hl=en), Boqing Gong, and Liqiang Wang** · ECCV 2022 · pp. 449–465

[Paper / arXiv](https://arxiv.org/abs/2109.09023) · [Open-access PDF](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136730449.pdf) · [Published version](https://doi.org/10.1007/978-3-031-19778-9_26) · [BibTeX](citation.bib)

## Research overview

How can an individual verify that their personal images were used to train a neural network without authorization, even when those images form only a small fraction of the training set?

Anti-Neuron Watermarking applies a specialized **linear color transformation** to a user's images. Training on those images can imprint the user's watermark signature on a neural classifier. An arbitrator can then infer the signature from the trained network to assess potential unauthorized data use. The paper studies watermark properties and signature spaces for this verification process.

This work is relevant to **dataset watermarking, data ownership verification, and detection of unauthorized neural network training**. The verification mechanism concerns use of training data; see the paper for its assumptions and experimental scope.

The earlier preprint was titled *Anti-Neuron Watermarking: Protecting Personal Data Against Unauthorized Neural Model Training* (arXiv:2109.09023, first submitted in 2021). The published conference version is ECCV **2022** and uses the title shown above.

## Code and experiments

| File | Purpose |
| --- | --- |
| [main.py](main.py) | Training entry point and experiment arguments |
| [datasets.py](datasets.py) | Dataset handling and image transformations |
| [models](models) | Neural network model definitions |
| [signature_inference_test.ipynb](signature_inference_test.ipynb) | Signature inference notebook |
| [mi_attacks.py](mi_attacks.py) | Membership inference attack code |
| [requirements.txt](requirements.txt) | Recorded PyTorch and torchvision versions |

The recorded dependencies include `torch==1.7.1` and `torchvision==0.8.2`; use an environment compatible with those versions. Review dataset paths and notebook configuration before running experiments. After dependencies are installed, inspect the training options with:

```bash
python main.py --help
```

The parser exposes dataset, watermark ratio, signature key, and training options. Match these settings to the paper's experiment you intend to reproduce. This documentation update does not establish a fully validated reproduction environment.

## Citation

Please cite the published ECCV 2022 version when building on this work:

```bibtex
@inproceedings{Zou_2022_ECCV,
  author = {Zou, Zihang and Gong, Boqing and Wang, Liqiang},
  title = {Anti-Neuron Watermarking: Protecting Personal Data Against Unauthorized Neural Networks},
  booktitle = {Computer Vision -- ECCV 2022},
  year = {2022},
  pages = {449--465},
  doi = {10.1007/978-3-031-19778-9_26},
  url = {https://arxiv.org/abs/2109.09023}
}
```

Machine-readable citation metadata is available in [CITATION.cff](CITATION.cff), with the paper as the preferred citation, and in [citation.bib](citation.bib).

## Related research

[Neural Plagiarism (ICCV 2025)](https://github.com/zzzucf/Neural-Plagiarism) investigates diffusion-model transformations that challenge visible and invisible image copyright markers.

[Zihang Zou on Google Scholar](https://scholar.google.com/citations?user=GLuGAK0AAAAJ&hl=en) · [Research profile](https://github.com/zzzucf)
