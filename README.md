# Fuseformer:A Manifold Metric Fusing Attention for Pedestrian Trajectory Prediction

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Build](https://img.shields.io/badge/build-passing-brightgreen.svg)
![Contributors](https://img.shields.io/github/contributors/your-repo)

## 📌 Introduction

Abstract--Pedestrian trajectory prediction is challenging due to ambiguous walking intents and complex interactions in non-Euclidean spaces among multiple agents. Recently, Transformers have made significant progress in pedestrian trajectory prediction. Their success is primarily attributed to the cross-stacking of Multi-Head Self-attention (MHA) mechanisms. However, self-attention and most of its variants heavily rely on dot product operations in Euclidean space, which are insufficient for capturing the non-Euclidean structures of pedestrian interactions.To address this issue, we propose a hybrid Transformer structure with a gating network, called FuseFormer. FuseFormer introduces Geodesic Self-Attention (GSA) alongside MHA to capture the non-Euclidean features in pedestrian interactions and adaptively fuse these features with learnable weights 
through the gating network. Additionally, to enhance the temporal dependency of the neural network, we apply Neural Ordinary Differential Equations in the decoder to model the continuous-time dynamics oftrajectories. We demonstrate that FuseFormer achieves state-of-the-art performance on pedestrian trajectory prediction datasets, including ETH/UCY and SDD. Moreover, we conduct extensive ablation studies to validate the effectiveness of our proposed method. 

## 🚀 Features
- MANIFOLD
- GSA(Geodesic Self-Attention)
- NEURAL ODE

## 🛠 Installation

```sh
# Clone the repository
git clone https://github.com/joyecnu/Fuseformer.git
cd Fuseformer

# Install dependencies
pip install -r requirements.txt
```

## 📖 Usage

```sh
# Example command to run the project
python train.py 
```


## 📝 Contributing

We welcome contributions! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

If you have any questions, feel free to contact:
- **Yi Zou** - yizdudub@outlook.com
Outlook.com
- GitHub: [joyecnu](https://github.com/joyecnu)

---

Made with ❤️ by [Yi Zou]






