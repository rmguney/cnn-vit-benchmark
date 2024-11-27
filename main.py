import subprocess
import sys
import os

def main():
    print("Select an option:")
    print("1. Initialize Dataset (CIFAR-10)")
    print("2. Test Dataset (CIFAR-10)")
    print("3. Train MobileNet")
    print("4. Train DETR")
    print("5. Exit")

    choice = input("Enter the number of your choice: ")

    # Get the Python interpreter being used
    python_interpreter = sys.executable

    if choice == "1":
        # Ensure the correct script path is used
        subprocess.run([python_interpreter, os.path.join("src", "dataset", "initialize_dataset.py")])
    elif choice == "2":
        subprocess.run([python_interpreter, os.path.join("src", "dataset", "test_dataset.py")])
    elif choice == "3":
        print("MobileNet training script will be implemented next.")
    elif choice == "4":
        print("DETR training script will be implemented next.")
    elif choice == "5":
        print("Exiting...")
        return
    else:
        print("Invalid choice. Please select a valid option.")

if __name__ == "__main__":
    main()
