import sys

def main():
    while True:
        # Prompt user for input
        user_input = input(">> ").strip().lower()
        
        if user_input == 'install':
            print("Installing...")
            # Placeholder for installation logic
        elif user_input == 'exit':
            print("Exiting...")
            sys.exit()
        else:
            print("Invalid command. Please enter 'install' or 'exit'.")

if __name__ == "__main__":
    print("Hello from Selm on GitHub")
    main()
