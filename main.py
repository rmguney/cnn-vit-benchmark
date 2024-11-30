import os
import subprocess
import sys


def main():
    print("Select an option:")
    print("1. Initialize Dataset (CIFAR-10)")
    print("2. Test Dataset (CIFAR-10)")
    print("3. Train DeiT-T")
    print("4. Train MobileNetV3S")
    print("5. Train MobileNetV3L")
    print("6. Benchmark Models")
    print("7. Exit")

    choice = input("Enter the number of your choice: ")
    
    python_executable = sys.executable
    
    if choice == "1":
        subprocess.run([python_executable, "src/dataset/initialize_dataset.py"], env=os.environ)
    elif choice == "2":
        subprocess.run([python_executable, "src/dataset/test_dataset.py"], env=os.environ)
    elif choice == "3":
        epochs = input("Enter the number of epochs: ")
        learning_rate = input("Enter the learning rate: ")
        batch_size = input("Enter the batch size: ")
        os.environ["EPOCHS"] = epochs
        os.environ["LEARNING_RATE"] = learning_rate
        os.environ["BATCH_SIZE"] = batch_size
        subprocess.run([python_executable, "src/models/train_deit.py"], env=os.environ)
    elif choice == "4":
        epochs = input("Enter the number of epochs: ")
        learning_rate = input("Enter the learning rate: ")
        batch_size = input("Enter the batch size: ")
        os.environ["EPOCHS"] = epochs
        os.environ["LEARNING_RATE"] = learning_rate
        os.environ["BATCH_SIZE"] = batch_size
        subprocess.run([python_executable, "src/models/train_mobilenet_s.py"], env=os.environ)
    elif choice == "5":
        epochs = input("Enter the number of epochs: ")
        learning_rate = input("Enter the learning rate: ")
        batch_size = input("Enter the batch size: ")
        os.environ["EPOCHS"] = epochs
        os.environ["LEARNING_RATE"] = learning_rate
        os.environ["BATCH_SIZE"] = batch_size
        subprocess.run([python_executable, "src/models/train_mobilenet.py"], env=os.environ)
    elif choice == "6":
        print("Benchmarking Models...")
        subprocess.run([python_executable, "src/benchmark_models.py"], env=os.environ)
    elif choice == "7":
        print("Exiting...")
        exit()
    else:
        print("Invalid choice. Please select a valid option.")


if __name__ == "__main__":
    main()
