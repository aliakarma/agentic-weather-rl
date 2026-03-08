from datasets import load_dataset

print("Downloading SEVIR dataset from HuggingFace...")

dataset = load_dataset("aliakarma/SEVIR")

print("Dataset downloaded successfully!")
print(dataset)
