import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler

def get_class_distribution(dataset_obj):
    count_dict = {k: 0 for k, v in dataset_obj.class_to_idx.items()}
    for element in dataset_obj:
        y_lbl = element[1]
        y_lbl = idx2class[y_lbl]
        count_dict[y_lbl] += 1
    return count_dict

# Define the transformations to apply to the dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the CIFAR10 dataset
cifar10_dataset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)

# Define the desired subset size
subset_size = 1000

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

idx2class = {v: k for k, v in cifar10_dataset.class_to_idx.items()}
print("Distribution of classes: \n", get_class_distribution(cifar10_dataset))

# Create a subset of the CIFAR10 dataset
subset_indices = torch.randperm(len(cifar10_dataset))[:subset_size]
subset_cifar10 = Subset(cifar10_dataset, subset_indices)


# Create the subset DataLoader
batch_size = 32
subset_dataloader = DataLoader(subset_cifar10, batch_size=batch_size, shuffle=True)

# Now you can use the subset_dataloader for training or analysis
for batch in subset_dataloader:
    # Perform your training or analysis operations here
    pass