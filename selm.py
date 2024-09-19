def install_selms():
    """
    Function to handle SELM installation locally.
    """
    # Placeholder for actual installation commands
    print("Installing SELM locally...")
    # Example installation command (replace with actual installation commands)
    #os.system("echo 'Running SELM installation commands here...'")
    print("SELM installation completed.")

def show_menu():
    """
    Display the menu and handle user input.
    """
    while True:
        print("\nSELM Menu")
        print("1. Install SELM (locally)")
        print("2. Exit")
        
        choice = input("Enter your choice (1 or 2): ")
        
        if choice == '1':
            install_selms()
        elif choice == '2':
            print("Exiting...")
            sys.exit()
        else:
            print("Invalid choice, please enter 1 or 2.")

if __name__ == "__main__":
    # Print "Hello from Selm on GitHub" to the console
    print("Hello from Selm on GitHub")
    show_menu()
