import sys

def fetch_readme(url):
    """Fetch the README.md file from the provided URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text.rstrip()  # Remove trailing newline characters
    except requests.RequestException as e:
        print(f"Failed to fetch README: {e}")
        return None

def main():
    readme_url = "https://raw.githubusercontent.com/UsernamesAre4Slaves/selm/refs/heads/main/README.md"
    
    while True:
        # Prompt user for input
        user_input = input(">> ").strip().lower()
        
        if user_input == 'install':
            print("Installing...")
            # Placeholder for installation logic
        elif user_input == 'about':
            readme_content = fetch_readme(readme_url)
            if readme_content:
                print("\n" + readme_content)
        elif user_input == 'exit':
            print("Exiting...")
            sys.exit()
        else:
            print("Invalid command. Please enter 'install', 'about', or 'exit'.")

if __name__ == "__main__":
    main()
