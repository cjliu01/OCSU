# OCSU: Optical Chemical Structure Understanding for Molecule-centric Scientific Discovery

[![arXiv](https://img.shields.io/badge/arXiv-Paper-red)](https://arxiv.org/pdf/2501.15415)
[![ckpts](https://img.shields.io/badge/ckpts-Mol--VL--7B-blue)](https://huggingface.co/PharMolix/Mol-VL-7B)
[![dataset](https://img.shields.io/badge/dataset-Vis--CheBI20-purple)](https://huggingface.co/datasets/PharMolix/Vis-CheBI20)

Molecules represent tokens of the language of chemistry, which underlies not only chemistry itself, but also scientific fields that use chemical information such as pharmacy, material science, and molecular biology. Existing molecular information is distributed across text books, publications, and patents. To describe structural information (spatial arrangement of atoms), molecules are commonly drawn as 2D images in such documents, which makes Optical Chemical Structure Understanding (OCSU) play an important role in molecule-centric scientific discovery.

Optical chemical structure recognition (OCSR) is a well-explored task in the molecular information extraction field, which takes the image as input and predicts the SMILES representation of the molecule. Nevertheless, SMILES representation is machine readable strings, while it is not chemist-friendly. This hampers overall understanding of the optical chemical structure and hinders further application of cutting-edge natural language processing approaches, e.g. LLM, for moleculecentric scientific discovery. Our work expands low-level recognition to multilevel understanding and aims to translate chemical structure diagrams into readable strings for both machine and chemist.

OCSU aims to automatically translate chemical structure diagrams into chemist-readable or machine-readable strings that describe the molecule from motif level to molecule level and abstract level. Typically, it includes four subtasks, that is, functional group caption, molecular description, chemist-readable IUPAC naming, and machine-readable SMILES naming (OCSR). On the basis of these, molecular structural information can be fully extracted to support downstream tasks, such as moleculecentric chat, property prediction, and molecule editing.

More information can be found at [Arxiv](https://arxiv.org/pdf/2501.15415). OCSU will be added to [OpenBioMed](https://github.com/PharMolix/OpenBioMed) toolkit. Stay tuned!

## News
* [2025/03/18] Released model Mol-VL-7B @ [HuggingFace🤗](https://huggingface.co/PharMolix/Mol-VL-7B) & [WiseModel](https://wisemodel.cn/models/PharMolix/Mol-VL-7B)
* [2025/03/14] Released dataset Vis-CheBI20 @ [HuggingFace🤗](https://huggingface.co/datasets/PharMolix/Vis-CheBI20) & [WiseModel](https://wisemodel.cn/datasets/PharMolix/Vis-CheBI20)

## Setup
* Setup environment

    ```bash
    # Virtual environment
    conda create -n ocsu python=3.10
    conda activate ocsu

    # llama_factory (0.9.1.dev0) setup
    cd OCSU
    pip install -r requirements.txt
    pip install -e .[metrics]
    llamafactory-cli version

    # Optional installation
    pip install deepspeed
    pip install flash-attn --no-build-isolation
    ```

* Setup dataset

    * Download [Vis-CheBI20](https://huggingface.co/datasets/PharMolix/Vis-CheBI20) Dataset
    * Create the symlinks

        ```bash
        # Training set
        ln -s ${Vis-CheBI20_ROOT}/train.json data/Vis-CheBI20
        # Test set
        ln -s ${Vis-CheBI20_ROOT}/test.json data/Vis-CheBI20
        ```
    
## Experiments
* Training

    ```bash
    # Here we provide a script for training with Vis-CheBI20.
    CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train config/mol-vl_7b.yaml
    ```

* Inference
    > Mol-VL-7B is released @ [HuggingFace](https://huggingface.co/PharMolix/Mol-VL-7B)

    * Inference service

        ```bash
        # Setup service @ port 8000
        CUDA_VISIBLE_DEVICES=0 API_PORT=8000 llamafactory-cli api config/inference/mol-vl_7b.yaml
        ```
    
    * Predict

        ```bash
        # Here we provide a script for evaluation on Vis-CheBI20.
        python scripts/inference.py
        ```

* Evaluation

    ```bash
    # Here we provide a script for evaluation on Vis-CheBI20.
    python scripts/evaluate.py
    ```

    > [scibert_scivocab_uncased](https://huggingface.co/allenai/scibert_scivocab_uncased) is adopted for evaluation. Please download to `ckpts/scibert_scivocab_uncased`.

* Demo

    Checkout our [Jupytor notebooks](./scripts/demo.ipynb) for a quick start!

## Acknowledgment
This repo benefits from [LLaMA_Factory](https://github.com/hiyouga/LLaMA-Factory) group. Thanks for their wonderful works.

## Citation
If you find our work helpful to your research, please consider giving this repository a 🌟star and 📎citing the following article. Thank you for your support!

```
@article{fan2025ocsu,
  title={OCSU: Optical Chemical Structure Understanding for Molecule-centric Scientific Discovery},
  author={Fan, Siqi and Xie, Yuguang and Cai, Bowen and Xie, Ailin and Liu, Gaochao and Qiao, Mu and Xing, Jie and Nie, Zaiqing},
  journal={arXiv preprint arXiv:2501.15415},
  year={2025}
}
```


