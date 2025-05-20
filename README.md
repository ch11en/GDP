Code + Models for ACL 2024 Findings paper "Pinpointing Diffusion Grid Noise to Enhance Aspect Sentiment Quad Prediction"
Paper Link: https://aclanthology.org/2024.findings-acl.222/

Module Requirements:

You can recreate the full Conda environment used by running the following (may require some tweaking of the environment name/path to run on your machine):
```
conda env create -f environment.yml
```
```
Python >- 3.9+
torch >= 1.10
pytorch-lightning >= 1.8.6
sentencepiece >= 0.1.97
transformers >= 4.19.0
```

Module Usage:
1. `conda activate GDP`
2. Run `python main.py` for model training/inference. 

Please cite our paper as such:
```
@inproceedings{zhu-etal-2024-pinpointing,
    title = "Pinpointing Diffusion Grid Noise to Enhance Aspect Sentiment Quad Prediction",
    author = "Zhu, Linan  and
      Chen, Xiangfan  and
      Guo, Xiaolei  and
      Zhang, Chenwei  and
      Zhu, Zhechao  and
      Zhou, Zehai  and
      Kong, Xiangjie",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Findings of the Association for Computational Linguistics ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand and virtual meeting",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.222",
    pages = "3717--3726",
    abstract = "Aspect sentiment quad prediction (ASQP) has garnered significant attention in aspect-based sentiment analysis (ABSA). Current ASQP research primarily relies on pre-trained generative language models to produce templated sequences, often complemented by grid-based auxiliary methods. Despite these efforts, the persistent challenge of generation instability remains unresolved and the effectiveness of grid methods remains underexplored in current studies. To this end, we introduce \textbf{G}rid Noise \textbf{D}iffusion \textbf{P}inpoint Network (\textbf{GDP}), a T5-based generative model aiming to tackle the issue of generation instability. The model consists of three novel modules, including Diffusion Vague Learning (DVL) to facilitate effective model learning and enhance overall robustness; Consistency Likelihood Learning (CLL) to discern the characteristics and commonalities of sentiment elements and thus reduce the impact of distributed noise; and GDP-FOR, a novel generation template, to enable models to generate outputs in a more natural way. Extensive experiments on four datasets demonstrate the remarkable effectiveness of our approach in addressing ASQP tasks.",
}
```
