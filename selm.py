import sys
import requests

def install():
    print("Installing...")

def upload(file_name):
    try:
        with open(file_name, 'r') as file:
            content = file.read()
            print(f"File '{file_name}' contents:\n{content}")
            
            # Send POST request to the specified URL
            url = f"http://selm.atwebpages.com/?param1={file_name}&param2={content}"
            response = requests.post(url)
            
            print(f"POST request sent. Status code: {response.status_code}")
            print(f"Response: {response.text}")

    except FileNotFoundError:
        print(f"File '{file_name}' not found.")
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")

def main():
    if len(sys.argv) < 2:
        print("No command provided. Use 'install' or 'upload'.")
        return

    command = sys.argv[1].lower()

    if command == 'install':
        install()
    elif command == 'upload':
        if len(sys.argv) < 3:
            print("Please provide a file name after 'upload'.")
        else:
            file_name = sys.argv[2]
            upload(file_name)
    else:
        print(f"Unknown command: {command}. Use 'install' or 'upload'.")

if __name__ == "__main__":
    main()
