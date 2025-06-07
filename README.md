# Directed-Link-Prediction-using-GNN-with-Local-and-Global-Feature-Fusion

Link prediction is a classical problem in graph analysis with many practical applications. For directed graphs, recently developed deep learning approaches typically analyze node similarities through contrastive learning and aggregate neighborhood information through graph convolutions. In this work, we propose a novel graph neural network (GNN) framework to fuse feature embedding with community information. We theoretically demonstrate that such hybrid features can improve the performance of directed link prediction. To utilize such features efficiently, we also propose an approach to transform input graphs into directed line graphs so that nodes in the transformed graph can aggregate more information during graph convolutions. Experiments on benchmark datasets show that our approach outperforms the state-of-the-art in most cases when 30%, 40%, 50%, and 60% of the connected links are used as training data, respectively.



## Requirements
- Python 3.x
- PyTorch
- [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) (`torch-geometric`)
- ......

## run

train.py

## Cite
If you use this repository in your work, please cite the corresponding DOI: 10.1109/TNSE.2024.3498434


@ARTICLE{10763430,
  author={Zhang, Yuyang and Shen, Xu and Xie, Yu and Wong, Ka-Chun and Xie, Weidun and Peng, Chengbin},
  journal={IEEE Transactions on Network Science and Engineering}, 
  title={Directed Link Prediction Using GNN With Local and Global Feature Fusion}, 
  year={2025},
  volume={12},
  number={1},
  pages={409-422},
  keywords={Directed graphs;Labeling;Graph neural networks;Deep learning;Transforms;Predictive models;Contrastive learning;Aggregates;Perturbation methods;Accuracy;Community detection;directed graphs;line graphs;link prediction;node embedding},
  doi={10.1109/TNSE.2024.3498434}}
