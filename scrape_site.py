"""Scrape a WordPress site (via its sitemaps) into docs/<site>/ as .txt files, and download linked PDFs."""
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

BASE = sys.argv[1] if len(sys.argv) > 1 else "https://liet.in"
SITEMAPS = ["page-sitemap.xml", "post-sitemap.xml"]
OUT = Path("docs") / urlparse(BASE).netloc
HEADERS = {"User-Agent": "Mozilla/5.0 (college-chatbot scraper)"}
NOISE = ["script", "style", "noscript", "nav", "header", "footer", "form", "svg", "iframe"]


def get(url: str) -> requests.Response:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r


def page_text(soup: BeautifulSoup) -> str:
    for tag in soup(NOISE):
        tag.decompose()
    root = soup.find("main") or soup.find(class_=re.compile("elementor")) or soup.body or soup
    lines = [" ".join(l.split()) for l in root.get_text("\n").splitlines()]
    seen, out = set(), []
    for l in lines:  # drop blanks and repeated menu/widget lines
        if len(l) > 2 and l not in seen:
            seen.add(l)
            out.append(l)
    return "\n".join(out)


HOSTS = {urlparse(BASE).netloc, "admissions." + urlparse(BASE).netloc}
SKIP_FILES = re.compile(r"\.(jpe?g|png|gif|webp|svg|zip|docx?|xlsx?|pptx?|mp4|css|js)$", re.I)
SKIP_PDF = re.compile(r"(feedback|survey)", re.I)


def discover() -> tuple[list[str], set[str]]:
    """Find every internal page: sitemaps first, then follow links (BFS). Returns (pages, pdf links)."""
    queue = [BASE + "/", "https://admissions." + urlparse(BASE).netloc + "/"]
    for sm in SITEMAPS:
        queue += re.findall(r"<loc>([^<]+)</loc>", get(urljoin(BASE, sm)).text)
    seen, pages, pdfs = set(), [], set()
    while queue:
        url = queue.pop(0).split("#")[0].split("?")[0].rstrip("/") or BASE
        if url in seen:
            continue
        seen.add(url)
        try:
            r = get(url)
        except Exception as e:
            print("skip", url, e)
            continue
        if "html" not in r.headers.get("content-type", ""):
            continue
        pages.append((url, r.text))
        for a in BeautifulSoup(r.text, "html.parser").find_all("a", href=True):
            link = urljoin(url, a["href"]).split("#")[0].split("?")[0]
            host = urlparse(link).netloc
            if host not in HOSTS:
                continue
            if link.lower().endswith(".pdf"):
                if not SKIP_PDF.search(link):
                    pdfs.add(link)
            elif not SKIP_FILES.search(link) and link.rstrip("/") not in seen:
                queue.append(link)
        time.sleep(0.3)
    return pages, pdfs


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pages, pdfs = discover()
    saved = 0
    for url, html in pages:
        soup = BeautifulSoup(html, "html.parser")
        title = soup.title.string.strip() if soup.title and soup.title.string else url
        text = page_text(soup)
        if len(text) < 200:
            continue
        parsed = urlparse(url)
        slug = re.sub(r"[^a-z0-9]+", "-", parsed.path.lower()).strip("-") or "home"
        if parsed.netloc.startswith("admissions."):
            slug = "admissions-" + slug
        (OUT / f"{slug}.txt").write_text(f"{title}\nSource: {url}\n\n{text}\n", encoding="utf-8")
        saved += 1
    print(f"Saved {saved} pages")
    for link in sorted(pdfs):
        name = Path(urlparse(link).path).name
        if (OUT / name).exists():
            continue
        try:
            (OUT / name).write_bytes(get(link).content)
            print("pdf", name)
        except Exception as e:
            print("skip pdf", link, e)
        time.sleep(0.3)


if __name__ == "__main__":
    main()
