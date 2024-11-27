import subprocess
import os

def main():
    print("Select an option:")
    print("1. Initialize Dataset (CIFAR-10)")
    print("2. Test Dataset (CIFAR-10)")
    print("3. Train MobileNetV3S")
    print("4. Train DETR")
    print("5. Exit")

    choice = input("Enter the number of your choice: ")

    if choice == "1":
        subprocess.run(["python", "src/dataset/initialize_dataset.py"])
    elif choice == "2":
        subprocess.run(["python", "src/dataset/test_dataset.py"])
    elif choice == "3":
        # Prompt for training parameters
        try:
            epochs = int(input("Enter the number of epochs: "))
            learning_rate = float(input("Enter the learning rate: "))
            batch_size = int(input("Enter the batch size: "))
            
            # Pass parameters as environment variables
            env = {
                "EPOCHS": str(epochs),
                "LEARNING_RATE": str(learning_rate),
                "BATCH_SIZE": str(batch_size),
            }
            
            subprocess.run(
                ["python", "src/models/train_mobilenet.py"], 
                env={**env, **os.environ}
            )
        except ValueError:
            print("Invalid input. Please enter valid numbers for epochs, learning rate, and batch size.")
    elif choice == "4":
        print("DETR training script not implemented yet.")
    elif choice == "5":
        print("Exiting...")
    else:
        print("Invalid choice. Please select a valid option.")

if __name__ == "__main__":
    main()
