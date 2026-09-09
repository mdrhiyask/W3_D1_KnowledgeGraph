import json
import re
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.tn.gov.in/"
SCHEME_LIST_URL = "https://www.tn.gov.in/scheme_list.php?dep_id=Mg=="

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/115.0.0.0 Safari/537.36"
    )
}


def fetch_scheme_links():
    """Fetch all scheme detail URLs from the main listing page."""
    print(f"Fetching scheme listing from: {SCHEME_LIST_URL}")
    response = requests.get(SCHEME_LIST_URL, headers=HEADERS, timeout=15)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, "html.parser")
    scheme_links = []

    # Parse anchor tags matching scheme detail pattern
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        if "scheme_data.php" in href or "scheme_detail" in href:
            full_url = href if href.startswith("http") else BASE_URL + href.lstrip("/")
            scheme_name = a_tag.get_text(strip=True)
            if full_url not in [item["url"] for item in scheme_links]:
                scheme_links.append({"title": scheme_name, "url": full_url})

    print(f"Found {len(scheme_links)} scheme links.")
    return scheme_links


def parse_scheme_page(url, fallback_title):
    """Scrape and structure detailed fields from an individual scheme page."""
    try:
        res = requests.get(url, headers=HEADERS, timeout=15)
        res.raise_for_status()
        soup = BeautifulSoup(res.content, "html.parser")

        # Extract title
        title_tag = soup.find("h2") or soup.find("h3") or soup.find("title")
        scheme_name = title_tag.get_text(strip=True) if title_tag else fallback_title

        # Key-Value extraction from tables or definition lists
        content_data = {}
        tables = soup.find_all("table")
        for table in tables:
            rows = table.find_all("tr")
            for row in rows:
                cols = row.find_all(["th", "td"])
                if len(cols) == 2:
                    key = cols[0].get_text(strip=True).replace(":", "")
                    value = cols[1].get_text(strip=True)
                    content_data[key] = value

        # Full page text for unstructured vector embedding
        full_text = " ".join([p.get_text(strip=True) for p in soup.find_all("p")])
        if not full_text:
            full_text = soup.get_text(separator=" ", strip=True)

        # Normalize extracted entities
        structured_entities = {
            "scheme_name": content_data.get("Scheme Name", scheme_name),
            "department": content_data.get("Department", "Agriculture and Farmers Welfare"),
            "beneficiary_type": content_data.get("Target Group", content_data.get("Beneficiary", "Farmers")),
            "eligibility": content_data.get("Eligibility Criteria", content_data.get("Eligibility", "N/A")),
            "subsidy_details": content_data.get("Benefits", content_data.get("Subsidy", "N/A")),
            "documents_required": content_data.get("Documents Required", "Aadhaar Card, Patta/Chitta"),
            "application_process": content_data.get("How to Apply", "Contact District Agriculture Office")
        }

        return {
            "source_url": url,
            "structured_entities": structured_entities,
            "unstructured_content": full_text
        }

    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None

#Fetching the data from the links and dump in to the json file and save it in local
def main():
    scheme_links = fetch_scheme_links()
    scraped_data = []

    # If no dynamic links found, fallback to direct parsing logic
    if not scheme_links:
        print("No links found on listing page. Using mock template for demo initialization...")
        scheme_links = [
            {
                "title": "Micro Irrigation Scheme for Agriculture",
                "url": "https://www.tn.gov.in/scheme_list.php?dep_id=Mg=="
            }
        ]

    for idx, item in enumerate(scheme_links):
        print(f"Processing ({idx+1}/{len(scheme_links)}): {item['title']}")
        data = parse_scheme_page(item["url"], item["title"])
        if data:
            scraped_data.append(data)

    # Save outputs to JSON file
    output_file = "tn_agriculture_schemes.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(scraped_data, f, indent=4, ensure_ascii=False)

    print(f"\nScraping complete! Saved {len(scraped_data)} records to '{output_file}'.")


if __name__ == "__main__":
    main()