# Atomic Feature Mimicking

**Unified Low-rank Compression Framework for Large-scale Recommendation Systems**

**Authors**: Hao Yu, Minghao Fu, Jiandong Ding, Yusheng Zhou, Jianxin Wu


<!-- [[`arXiv`](https://arxiv.org/pdf/2401.16811.pdf)] [[`bibtex`](#Citation)] -->

<!-- **Introduction**: This repository provides an implementation for the paper: "[Unified Low-rank Compression Framework for Large-scale Recommendation Systems](https://arxiv.org/pdf/2401.16811.pdf)" based on [DeepCTR-Torch](https://github.com/shenweichen/DeepCTR-Torch). We propose a unified and efficient low-rank decomposition framework compresses the embedding tables and MLP layers of CTR prediction models. -->

**Introduction**: This repository provides an implementation for the KDD2024 ADS track paper: "Unified Low-rank Compression Framework for Large-scale Recommendation Systems" based on [DeepCTR-Torch](https://github.com/shenweichen/DeepCTR-Torch). We propose a unified and efficient low-rank decomposition framework compresses the embedding tables and MLP layers of CTR prediction models.

**Note**: Due to copyright restrictions, we can only provide evaluation code, including [eval_criteo.py](https://github.com/yuhao318/Atomic_Feature_Mimicking/blob/main/eval_criteo.py) and [eval_avazu.py](https://github.com/yuhao318/Atomic_Feature_Mimicking/blob/main/eval_avazu.py), and the core compression code demo ([afm_emb.py](https://github.com/yuhao318/Atomic_Feature_Mimicking/blob/main/afm_emb.py) and [afm_mlp.py](https://github.com/yuhao318/Atomic_Feature_Mimicking/blob/main/afm_mlp.py)). We cannot provide the detailed engineering code. 

**Compress MLP Layers**:

First given a pre-trained CTR prediction model, calculate the MLP output of each layer of the model and calculate $U$, $V$ and $AVG$ ([step2()](https://github.com/yuhao318/Atomic_Feature_Mimicking/blob/main/afm_mlp.py#L86)). Then merge weight ([step3()](https://github.com/yuhao318/Atomic_Feature_Mimicking/blob/main/afm_mlp.py#L127)) and save checkpoint.


**Compress Embedding Tables**:

First given a pre-trained model, count the output of each sparse embedding feature of the model and calculate $U$, $V$ and $AVG$ ([step2()](https://github.com/yuhao318/Atomic_Feature_Mimicking/blob/main/afm_emb.py#L83)). Then merge the embedding table $D$ and compresion weight $U$ ([step3()](https://github.com/yuhao318/Atomic_Feature_Mimicking/blob/main/afm_emb.py#L113)) and retain the checkpoint. Note that fusion will fail when the interaction layer exists between the compressed weight $U^\top$ and the first FC layer weight $W$.

Please view our paper for more informations.

## <a name="Citation"></a>Citation

```bib
@article{yu2024Unified,
  title={Unified Low-rank Compression Framework for Large-scale Recommendation Systems},
  author={Hao Yu,Minghao Fu,Jiandong Ding,Yusheng Zhou,Jianxin Wu},
  year={2024}
}
```

## Contact

If you have any questions about our work, feel free to contact us through email (Hao Yu: yuh@lamda.nju.edu.cn) or Github issues.
