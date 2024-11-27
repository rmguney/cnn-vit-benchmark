import subprocess

def main():
    print("Select an option:")
    print("1. Initialize Dataset (CIFAR-10)")
    print("2. Test Dataset (CIFAR-10)")
    print("3. Train MobileNet")
    print("4. Train DETR")
    print("5. Exit")

    choice = input("Enter the number of your choice: ")
    if choice == "1":
    elif choice == "2":
    elif choice == "3":
    else:
        print("Invalid choice. Please select a valid option.")

if __name__ == "__main__":
    main()
