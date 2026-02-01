# Wikipedia data collection script
# Collects 200 fixed URLs + 300 random URLs = 500 total

import json
import random
import requests
from bs4 import BeautifulSoup
from typing import List, Dict
import time
import sys
import os
import pickle
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# CRITICAL FIX: Add User-Agent header - Wikipedia requires this
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# Session for connection pooling and better performance
session = requests.Session()
session.headers.update(HEADERS)

def get_random_articles_from_category(count: int = 50):
    """Get random Wikipedia articles using the API with proper error handling"""
    articles = []
    
    url = "https://en.wikipedia.org/w/api.php"
    
    # API limits to 10 per request, so we need multiple batches
    batches = (count // 10) + 1
    
    for batch_num in range(batches):
        if len(articles) >= count:
            break
            
        params = {
            'action': 'query',
            'format': 'json',
            'list': 'random',
            'rnnamespace': 0,  # Main namespace only
            'rnlimit': 10
        }
        
        max_retries = 5
        for attempt in range(max_retries):
            try:
                # Use session with headers for better reliability
                response = session.get(url, params=params, timeout=15)
                response.raise_for_status()
                
                # Try to parse JSON
                try:
                    data = response.json()
                except json.JSONDecodeError as je:
                    print(f"JSON decode error: {je}")
                    print(f"Response text: {response.text[:200]}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    else:
                        break
                
                # Check for valid response structure
                if 'query' not in data or 'random' not in data['query']:
                    print(f"Warning: Unexpected API response format")
                    print(f"Response keys: {data.keys()}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    else:
                        break
                
                # Extract article URLs
                for page in data['query']['random']:
                    article_url = f"https://en.wikipedia.org/wiki/{page['title'].replace(' ', '_')}"
                    articles.append(article_url)
                
                # Success - break retry loop
                time.sleep(0.5)  # Rate limiting
                break
                
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    print(f"Timeout, retrying... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(2 ** attempt)
                else:
                    print(f"Failed after {max_retries} timeout attempts")
                    
            except requests.exceptions.ConnectionError as e:
                if attempt < max_retries - 1:
                    print(f"Connection error, retrying... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(2 ** attempt)
                else:
                    print(f"Failed after {max_retries} connection attempts: {e}")
                    
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    print(f"Request error, retrying... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(2 ** attempt)
                else:
                    print(f"Failed after {max_retries} attempts: {e}")
            
            except Exception as e:
                print(f"Unexpected error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    break
    
    return articles[:count]

def validate_article(url: str, min_words: int = config.MIN_WORDS_PER_PAGE) -> bool:
    """Check if article has enough words"""
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            response = session.get(url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            content = soup.find('div', {'id': 'mw-content-text'})
            if not content:
                return False
            
            paragraphs = content.find_all('p')
            text = ' '.join([p.get_text() for p in paragraphs])
            word_count = len(text.split())
            
            return word_count >= min_words
            
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return False
            
        except requests.exceptions.RequestException:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return False
            
        except Exception as e:
            print(f"Error validating {url}: {e}")
            return False
    
    return False

def extract_article(url: str) -> Dict:
    """Extract article content with title and text"""
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            response = session.get(url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Get title
            title_elem = soup.find('h1', {'id': 'firstHeading'})
            title = title_elem.get_text() if title_elem else ""
            
            # Get main content
            content = soup.find('div', {'id': 'mw-content-text'})
            if not content:
                return None
            
            paragraphs = content.find_all('p')
            text = '\n\n'.join([p.get_text() for p in paragraphs])
            
            # Clean up text
            import re
            text = re.sub(r'\[[\d\s]+\]', '', text)  # Remove citation numbers like [1]
            text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
            text = text.strip()
            
            return {
                'url': url,
                'title': title,
                'content': text
            }
            
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            print(f"Timeout extracting {url}")
            return None
            
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            print(f"Request error extracting {url}: {e}")
            return None
            
        except Exception as e:
            print(f"Error extracting {url}: {e}")
            return None
    
    return None

def load_fixed_urls() -> List[str]:
    """Load the 200 fixed URLs from JSON file"""
    try:
        with open(config.FIXED_URLS_FILE, 'r') as f:
            data = json.load(f)
        print(f"Loaded {len(data['urls'])} fixed URLs")
        return data['urls']
    except FileNotFoundError:
        print("ERROR: fixed_urls.json not found!")
        print("Run: python generate_fixed_urls.py first")
        sys.exit(1)

def collect_random_urls(count: int) -> List[str]:
    """Collect random URLs that meet the criteria"""
    print(f"\nCollecting {count} random Wikipedia URLs...")
    print("These will be different each time you run the script")
    
    valid_urls = []
    attempts = 0
    max_attempts = count * 3  # Try up to 3x the needed amount
    
    with tqdm(total=count, desc="Random URLs") as pbar:
        while len(valid_urls) < count and attempts < max_attempts:
            # Get batch of articles
            batch = get_random_articles_from_category(20)
            
            if not batch:
                print("Warning: Failed to get batch, waiting before retry...")
                time.sleep(5)
                attempts += 10  # Penalize failed batches
                continue
            
            for url in batch:
                if len(valid_urls) >= count:
                    break
                
                # Skip duplicates
                if url in valid_urls:
                    continue
                
                # Check if article is valid
                if validate_article(url):
                    valid_urls.append(url)
                    pbar.update(1)
                
                attempts += 1
                time.sleep(0.5)  # Rate limiting - be nice to Wikipedia
    
    if len(valid_urls) < count:
        print(f"\nWarning: Only collected {len(valid_urls)} of {count} requested URLs")
    
    return valid_urls

def collect_corpus() -> List[Dict]:
    """Main function to collect all 500 articles (200 fixed + 300 random)"""
    print("="*70)
    print("WIKIPEDIA DATA COLLECTION")
    print("="*70)
    print("\nDataset:")
    print("  - 200 Fixed URLs (from fixed_urls.json)")
    print("  - 300 Random URLs (sampled fresh each run)")
    print("  - Minimum 200 words per article")
    print("  - Total: 500 articles")
    print()
    
    # Load the 200 fixed URLs
    print("Step 1: Loading fixed URLs...")
    fixed_urls = load_fixed_urls()
    print(f"  Loaded: {len(fixed_urls)} fixed URLs")
    
    # Get 300 new random URLs
    print("\nStep 2: Collecting random URLs...")
    random_urls = collect_random_urls(config.RANDOM_URLS_COUNT)
    print(f"  Collected: {len(random_urls)} random URLs")
    
    # Combine them
    all_urls = fixed_urls + random_urls
    print(f"\nTotal URLs: {len(all_urls)}")
    
    # Extract content from all articles
    print("\nStep 3: Extracting article content...")
    corpus = []
    
    for url in tqdm(all_urls, desc="Extracting"):
        article = extract_article(url)
        if article and article['content']:
            corpus.append(article)
        time.sleep(0.3)  # Rate limiting
    
    print(f"\nSuccessfully extracted: {len(corpus)} articles")
    
    return corpus

def save_corpus(corpus: List[Dict]):
    """Save corpus to file"""
    os.makedirs(config.DATA_DIR, exist_ok=True)
    
    with open(config.CORPUS_FILE, 'wb') as f:
        pickle.dump(corpus, f)
    
    print(f"\nCorpus saved to: {config.CORPUS_FILE}")
    print(f"Total articles: {len(corpus)}")
    print(f"File size: {os.path.getsize(config.CORPUS_FILE) / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    print("\nStarting data collection...")
    print("This will take 15-20 minutes depending on network speed.\n")
    
    try:
        # Collect all articles
        corpus = collect_corpus()
        
        # Save to file
        save_corpus(corpus)
        
        print("\n" + "="*70)
        print("DATA COLLECTION COMPLETE")
        print("="*70)
        print("\nNext step: python src/preprocessing.py")
        
    except KeyboardInterrupt:
        print("\n\nCollection interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Close session
        session.close()