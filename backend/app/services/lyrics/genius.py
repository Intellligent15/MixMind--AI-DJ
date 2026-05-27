import logging
import os
import re

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

def _clean_lyrics(text: str) -> str:
    """Clean up lyrics by removing headers like [Chorus] and extra spaces."""
    # Remove bracketed section headers
    text = re.sub(r"\[.*?\]", "", text)
    # Remove extra blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _build_search_query(title: str, artist: str | None) -> str:
    # Remove text in brackets/parentheses
    clean_title = re.sub(r'\(.*?\)|\[.*?\]', '', title).strip()
    
    # Remove trailing "ft. ..." or "feat. ..." that aren't in parentheses
    clean_title = re.sub(r'(?i)\b(?:ft\.?|feat\.?)\b.*$', '', clean_title).strip()
    
    query = clean_title
    if artist:
        # If there's a hyphen in the title, it likely already includes the true artist,
        # and the `artist` param might just be a random YouTube uploader.
        if "-" not in title:
            clean_artist = re.sub(r'(?i)vevo', '', artist).strip()
            query = f"{clean_title} {clean_artist}"
            
    # Clean up multiple spaces
    query = re.sub(r'\s+', ' ', query)
    return query.strip()


from app.core.config import settings

async def search_genius_song(title: str, artist: str | None) -> dict | None:
    """Search Genius API for a song and return its metadata dict."""
    token = settings.genius_access_token
    if not token:
        logger.warning("GENIUS_ACCESS_TOKEN not set")
        return None

    query = _build_search_query(title, artist)
    logger.info(f"Genius search query: {query}")
    
    headers = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient() as client:
        try:
            res = await client.get(
                "https://api.genius.com/search",
                params={"q": query},
                headers=headers,
                timeout=10.0,
            )
            res.raise_for_status()
        except httpx.HTTPError as e:
            logger.error(f"Genius API search failed: {e}")
            return None

        data = res.json()
        hits = data.get("response", {}).get("hits", [])
        if not hits:
            return None

        # Take the first song hit
        for hit in hits:
            if hit["type"] == "song":
                return hit["result"]
                
    return None


async def scrape_genius_lyrics(url: str) -> str | None:
    """Scrape the lyrics text from a Genius song URL."""
    async with httpx.AsyncClient() as client:
        try:
            res = await client.get(url, timeout=10.0)
            res.raise_for_status()
        except httpx.HTTPError as e:
            logger.error(f"Genius lyrics scrape failed: {e}")
            return None

    soup = BeautifulSoup(res.text, "html.parser")
    # Genius lyrics are stored in divs with data-lyrics-container="true"
    containers = soup.find_all("div", attrs={"data-lyrics-container": "true"})
    
    if not containers:
        return None

    lyrics_text_chunks = []
    for c in containers:
        # Remove unwanted structural/metadata elements that pollute the text
        for unwanted in c.find_all(['div', 'h2', 'button', 'svg', 'path']):
            unwanted.decompose()
            
        # Replace <br> with newlines so we don't lose line breaks
        for br in c.find_all("br"):
            br.replace_with("\n")
            
        # Extract the text and preserve spacing
        text = c.get_text(separator=" ").strip()
        if text:
            lyrics_text_chunks.append(text)

    lyrics_text = "\n\n".join(lyrics_text_chunks)
    return _clean_lyrics(lyrics_text)


async def fetch_lyrics(title: str, artist: str | None) -> tuple[int, str] | None:
    """Fetch lyrics from Genius. Returns (genius_id, text) or None."""
    song_metadata = await search_genius_song(title, artist)
    if not song_metadata:
        return None
        
    genius_id = song_metadata["id"]
    url = song_metadata["url"]
    
    text = await scrape_genius_lyrics(url)
    if not text:
        return None
        
    return genius_id, text
