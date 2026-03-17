import torch
import tensorflow as tf

def main():
    print("--- PyTorch ---")
    print(f"MPS (GPU) built: {torch.backends.mps.is_built()}")
    print(f"MPS (GPU) available: {torch.backends.mps.is_available()}")

    print("\n--- TensorFlow ---")
    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPUs available: {len(gpus)}")


    if gpus:
        for gpu in gpus:
            print(f"Device: {gpu.name}")
    else:
        print("TensorFlow is falling back to CPU.")


if __name__ == "__main__":
    main()
