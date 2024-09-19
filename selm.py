import sys
import requests
import urllib.parse

def upload(file_name):
    try:
        # Read the file contents
        with open(file_name, 'r') as file:
            content = file.read()

            # URL encode the file content
            encoded_content = urllib.parse.quote(content)

            # Create the URL with the file name and its contents as parameters
            url = f"http://selm.atwebpages.com/?filename={file_name}&contents={encoded_content}"
            
            # Send the GET request
            response = requests.get(url)

            # Print the response
            print(f"Request sent to: {url}")
            print(f"Status code: {response.status_code}")
            print(f"Response: {response.text}")

    except FileNotFoundError:
        print(f"File '{file_name}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python script.py upload <file_name>")
        return

    command = sys.argv[1].lower()
    file_name = sys.argv[2]

    if command == "upload":
        upload(file_name)
    else:
        print("Invalid command. Use 'upload' followed by the file name.")

if __name__ == "__main__":
    main()
