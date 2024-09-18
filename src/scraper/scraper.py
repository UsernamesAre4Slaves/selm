import os
import logging
import csv
import json
import argparse
import aiohttp
import asyncio
from lxml import html
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WebScraper:
    def __init__(self, base_url, output_file, item_class='item-class', title_tag='h2', content_tag='p',
                 user_agents=None, delay=1, retries=3, pagination_param='page'):
        self.base_url = base_url
        self.output_file = output_file
        self.item_class = item_class
        self.title_tag = title_tag
        self.content_tag = content_tag
        self.user_agents = user_agents or [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0'
        ]
        self.delay = delay
        self.retries = retries
        self.pagination_param = pagination_param

    async def fetch_page(self, session, url):
        """Fetch the content of the URL asynchronously."""
        headers = {
            'User-Agent': random.choice(self.user_agents)
        }
        for attempt in range(self.retries):
            try:
                async with session.get(url, headers=headers) as response:
                    response.raise_for_status()
                    return await response.text()
            except (aiohttp.ClientError, aiohttp.http_exceptions.HttpBadRequest) as e:
                logging.error(f"Attempt {attempt + 1} failed to fetch page: {url} with error: {e}")
                if attempt < self.retries - 1:
                    await asyncio.sleep(self.delay)
                else:
                    raise

    async def run(self):
        """Run the scraper asynchronously with pagination support."""
        async with aiohttp.ClientSession() as session:
            logging.info(f"Starting scraper for {self.base_url}")
            page_number = 1
            all_data = []
            while True:
                url = self.base_url.format(page_number=page_number)
                logging.info(f"Fetching page {page_number}: {url}")
                page_content = await self.fetch_page(session, url)
                data = self.parse_page(page_content)
                if not data:
                    break  # Exit loop if no more data is found
                all_data.extend(data)
                page_number += 1
                await asyncio.sleep(self.delay)
            self.save_to_csv(all_data)
            logging.info(f"Scraping completed.")

    def parse_page(self, page_content):
        """Parse the content of the page using BeautifulSoup and lxml."""
        tree = html.fromstring(page_content)
        data = []
        items = tree.xpath(f'//div[@class="{self.item_class}"]')
        
        for item in items:
            try:
                title = item.xpath(f'.//{self.title_tag}/text()')[0].strip()
                content = item.xpath(f'.//{self.content_tag}/text()')[0].strip()
                data.append({'title': title, 'content': content})
            except IndexError as e:
                logging.warning(f"Error parsing item: {e}")
        return data

    def validate_data(self, data):
        """Ensure data quality by validating entries."""
        return [entry for entry in data if entry['title'] and entry['content']]

    def deduplicate_data(self, data):
        """Remove duplicate entries based on title."""
        seen = set()
        unique_data = []
        for entry in data:
            if entry['title'] not in seen:
                seen.add(entry['title'])
                unique_data.append(entry)
        return unique_data

    def save_to_csv(self, data):
        """Save the data to a CSV file."""
        data = self.validate_data(data)
        data = self.deduplicate_data(data)
        try:
            with open(self.output_file, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=['title', 'content'])
                writer.writeheader()
                writer.writerows(data)
            logging.info(f"Data successfully saved to {self.output_file}")
        except IOError as e:
            logging.error(f"Error saving data to file: {self.output_file} with error: {e}")
            raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Web Scraper for scraping data.')
    parser.add_argument('--url', required=True, help='Base URL to scrape with pagination support (e.g., "https://example.com?page={page_number}")')
    parser.add_argument('--output', required=True, help='Output file path (CSV)')
    parser.add_argument('--item_class', default='item-class', help='CSS class of the items to scrape')
    parser.add_argument('--title_tag', default='h2', help='HTML tag for the title')
    parser.add_argument('--content_tag', default='p', help='HTML tag for the content')
    parser.add_argument('--delay', type=int, default=1, help='Delay between requests in seconds')
    parser.add_argument('--retries', type=int, default=3, help='Number of retries for failed requests')
    
    args = parser.parse_args()

    scraper = WebScraper(
        base_url=args.url,
        output_file=args.output,
        item_class=args.item_class,
        title_tag=args.title_tag,
        content_tag=args.content_tag,
        delay=args.delay,
        retries=args.retries
    )

    asyncio.run(scraper.run())
