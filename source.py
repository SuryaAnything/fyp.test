import torch

def main():
    # 1. Check for CUDA (GPU) availability
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU is available and will be used: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("GPU not found, using CPU instead.")

    print(f"PyTorch Version: {torch.__version__}")
    print(f"Using device: {device}")

    # 3. Demonstrate creating a tensor on the selected device
    try:
        print("\nCreating a sample tensor...")
        # Create a 2x3 tensor of random numbers
        example_tensor = torch.rand(2, 3, device=device)
        print("Successfully created tensor on the device:")
        print(example_tensor)
        print(f"Tensor is on device: {example_tensor.device}")
    except Exception as e:
        print(f"\nAn error occurred during tensor creation: {e}")


if __name__ == "__main__":
    main()