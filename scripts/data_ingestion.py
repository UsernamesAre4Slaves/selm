import requests
from bs4 import BeautifulSoup
import pandas as pd
import asyncio
import aiohttp
from typing import List, Dict

async def fetch_url(session: aiohttp.ClientSession, url: str) -> str:
    """
    Fetch the content of a URL asynchronously.

    Parameters:
    - session (aiohttp.ClientSession): The session object to use for the request.
    - url (str): The URL to fetch.

    Returns:
    - str: The HTML content of the page.
    """
    async with session.get(url) as response:
        response.raise_for_status()  # Raise an exception for HTTP errors
        return await response.text()

def parse_html(html: str, element_tags: List[str]) -> List[str]:
    """
    Parse HTML content to extract text from specified elements.

    Parameters:
    - html (str): The HTML content to parse.
    - element_tags (List[str]): List of HTML tags to extract text from.

    Returns:
    - List[str]: Extracted text from the specified HTML elements.
    """
    soup = BeautifulSoup(html, 'html.parser')
    texts = []
    for tag in element_tags:
        texts.extend([element.get_text(strip=True) for element in soup.find_all(tag)])
    return texts

def extract_metadata(soup: BeautifulSoup) -> Dict[str, str]:
    """
    Extract metadata from the HTML content.

    Parameters:
    - soup (BeautifulSoup): The BeautifulSoup object of the HTML content.

    Returns:
    - Dict[str, str]: Extracted metadata including title and potentially other fields.
    """
    metadata = {}
    title = soup.find('title')
    metadata['title'] = title.get_text(strip=True) if title else 'No title found'
    
    # Example: Extract publication date if available
    date = soup.find('meta', {'name': 'date'})
    metadata['date'] = date['content'] if date else 'No date found'
    
    return metadata

async def scrape_data(urls: List[str], output_file: str, element_tags: List[str] = None):
    """
    Scrape data from a list of URLs and save to a CSV file.

    Parameters:
    - urls (List[str]): List of URLs to scrape data from.
    - output_file (str): Path to save the scraped data.
    - element_tags (List[str], optional): List of HTML tags to extract text from. Defaults to ['p'].
    """
    if element_tags is None:
        element_tags = ['p']

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(session, url) for url in urls]
        html_pages = await asyncio.gather(*tasks)
        
        all_texts = []
        all_metadata = []

        for html in html_pages:
            soup = BeautifulSoup(html, 'html.parser')
            texts = parse_html(html, element_tags)
            metadata = extract_metadata(soup)

            all_texts.extend(texts)
            all_metadata.append(metadata)
        
        # Create DataFrame with metadata and text
        df_texts = pd.DataFrame(all_texts, columns=['text'])
        df_metadata = pd.DataFrame(all_metadata)

        # Combine text and metadata into a single DataFrame
        df_combined = pd.concat([df_metadata, df_texts], axis=1)
        df_combined.to_csv(output_file, index=False)

# Example usage
if __name__ == '__main__':
    urls = ['https://example.com', 'https://anotherexample.com']
    asyncio.run(scrape_data(urls, 'data/raw/enhanced_data.csv', element_tags=['p', 'h1', 'h2']))
