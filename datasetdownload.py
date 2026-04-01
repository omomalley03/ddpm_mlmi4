import kagglehub
from torchvision import datasets

# Download CelebA-HQ
celeba_path = kagglehub.dataset_download("badasstechie/celebahq-resized-256x256")
print("Path to CelebA-HQ dataset files:", celeba_path)

# Download CIFAR-10
cifar_dataset = datasets.CIFAR10(root="./data", train=True, download=True)
print("CIFAR-10 downloaded to: ./data/cifar-10-batches-py")