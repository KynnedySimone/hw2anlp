import argparse
import asyncio
import csv
import json
import os
import re
import urllib.request
from fnmatch import fnmatch
from urllib.parse import urljoin
import yaml
from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import CrawlerRunConfig
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from pypdf import PdfReader

# Crawls the URLs in urls.yaml, plus any PDFs dropped in data/pdfs
# (PDF urls in urls.yaml get downloaded into data/pdfs automatically)
# Two rounds:
#   1. every URL in urls.yaml
#   2. links found in round 1 -- the follow_links seeds in urls.yaml, plus
#      Carnegie Museum of Art collection records (see discover_cmoa_objects)
# follow_links entries: seed, pattern (one glob or a list), category, and optional
# menus: true to also follow links that only appear in the site's nav/header/footer menus
# needs: pip install crawl4ai pypdf pyyaml cssselect  (cssselect is required for excluded_selector)
# scanned pdf pages are OCR'd: pip install pypdfium2 pytesseract, brew install tesseract
# --only-new: only crawl urls.yaml entries that aren't in scraped_data.json yet (follow_links
#   seeds are always re-crawled, so new links on them get followed)
# --pdfs-only: no crawling, just re-parse data/pdfs into scraped_data.json
# to run new: python scraper.py --only-new, to check coverage: python check_coverage.py --data data/scraped_data.json --eval public_evaluation_data.json

DATA_DIR = "data"
PDF_DIR = os.path.join(DATA_DIR, "pdfs")
JSON_FILE = os.path.join(DATA_DIR, "scraped_data.json")
SAVE_EVERY = 20  # write the json after this many pages
CONCURRENCY = 10  # pages crawled at once
PAGE_HARD_TIMEOUT = 300  # seconds before giving up on a page (longest page_timeout below is 180s)
USER_AGENT = "hw2-rag-scraper/1.0 (student project)"

# sites that load their content with javascript -> scroll + wait before grabbing html
DYNAMIC_SITES = [
    "*steelers.com*",
    "*events.cmu.edu*",
    "*improv.com*",
    "*pittsburgh.events*",
    "*pitt.libcal.com*",
    "*sports/*/schedule*",  # sidearm athletics sites (pitt, duquesne, rmu)
    "*downtownpittsburgh.com/event-series*",  # every date of a recurring event
]
# sites where remove_overlay_elements deletes the real content (steelers.com rosters come back empty)
KEEP_OVERLAY_SITES = [
    "*steelers.com*",
]

# Carnegie Museum of Art collection: which classifications to pull object records for.
# The site has 100k+ objects (mostly Teenie Harris photos); paintings is ~1,270.
# Other options: "sculpture", "drawings-and-watercolors", "prints", ...
CMOA_HOME = "https://collection.carnegieart.org/"
CMOA_CLASSIFICATIONS = ["paintings"]
# The museum's CC0 open-data release (2017 snapshot of its collection database), mirrored at
# github.com/roll/collection after the original cmoa/collection repo went away. Used for the
# acquisition dates the website doesn't show (see add_cmoa_open_data).
CMOA_OPEN_DATA_URL = "https://raw.githubusercontent.com/roll/collection/master/cmoa.csv"
CMOA_OPEN_DATA = os.path.join(DATA_DIR, "cmoa_open_data.csv")
CMOA_OPEN_DATA_MARKER = "#### From the museum's open data (2017)"

# hosts whose links get saved under "social links" and appended to the page text,
# since handles (e.g. instagram.com/carnegiemuseumofart) only exist in the link url
SOCIAL_DOMAINS = ["instagram.com", "facebook.com", "twitter.com", "x.com", "tiktok.com",
                  "youtube.com", "linkedin.com", "threads.net", "bsky.app"]
# pghcitypaper.com is not here: it answers 403/410 to any scraper, so it's dropped from urls.yaml


# js that keeps clicking a "load more" button until it disappears (or max_clicks)
def click_load_more(selector, max_clicks):
    return f"""
    for (let i = 0; i < {max_clicks}; i++) {{
        const btns = [...document.querySelectorAll('{selector}')]
            .filter(b => b.offsetParent !== null && !b.disabled);
        if (!btns.length) break;
        btns[btns.length - 1].click();
        await new Promise(r => setTimeout(r, 1500));
    }}
    """


# sites with a "load more" button: (url pattern, button selector, max clicks)
LOAD_MORE_SITES = [
    ("*ppgpaintsarena.com/events*", "#loadMoreEvents", 15),       # 24 -> 55 events
    ("*downtownpittsburgh.com/events*", ".load-more button", 30),  # 39 -> 73 events
]

# text that shows up on bot-check / error pages instead of real content
BLOCKED_MARKERS = [
    "just a moment...",
    "attention required",
    "verify you are human",
    "enable javascript and cookies",
    "access denied",
    "checking your browser",
    "request unsuccessful",
]
MIN_CHARS = 300  # anything shorter than this is probably not a real page
MIN_PAGE_CHARS = 50  # a pdf page with less extractable text than this is probably scanned -> OCR

YEAR = re.compile(r"^(19|20)\d\d$")
# one table cell: $ 1,234 / (1,234) / 12.5% / (0.8)% / — ; "$ " is glued to its number first
TABLE_VALUE = re.compile(r"^\$?\(?-?[\d,]*\d(\.\d+)?\)?%?$|^[—–]$")
# words that label a column under (or after) its year: "2024 Estimate", "Increase/ (Decrease)", "% Change"
COLUMN_WORD = re.compile(r"^(Actual|Budget|Estimate|Forecast|Projected|Proposed|Adopted|Amended|Approved|"
                         r"Revised|Requested|Increase/?|\(Decrease\)|Change|Variance|CAGR\*?|%)$")
# change columns whose second header line is missing
SHORT_COLUMN = {"Increase/": "Increase/(Decrease)", "%": "% Change"}
MULTIWORD_COLUMN = re.compile(r"\b(Amended|Adopted|Proposed|Approved|Revised|Requested) Budget\b")


def ocr_page(pdf_path, index):
    # render the page to an image and read it with tesseract (brew install tesseract)
    try:
        import pypdfium2
        import pytesseract
    except ImportError:
        print("OCR SKIPPED (pip install pypdfium2 pytesseract):", pdf_path)
        return ""
    image = pypdfium2.PdfDocument(pdf_path)[index].render(scale=300 / 72).to_pil()
    return pytesseract.image_to_string(image)


def table_header(line):
    """'Account 2025 2026 CAGR*' -> (['2025', '2026'], ['CAGR*']); None if the line isn't a year header."""
    tokens = line.split()
    first = next((i for i, t in enumerate(tokens) if YEAR.match(t)), None)
    # allow a short label for the row-name column ("Account", "Subclass") before the years
    if first is None or first > 2 or any(re.search(r"\d", t) for t in tokens[:first]):
        return None
    years = []
    for t in tokens[first:]:
        if not YEAR.match(t):
            break
        years.append(t)
    extra = tokens[first + len(years):]
    if len(years) < 2 or not all(COLUMN_WORD.match(t) for t in extra):
        return None
    return years, extra


def column_words(line):
    """'Actual Actual Amended Budget CAGR*' -> ['Actual', 'Actual', 'Amended Budget', 'CAGR*'], else None"""
    tokens = MULTIWORD_COLUMN.sub(lambda m: m.group(0).replace(" ", "_"), line).split()
    if tokens and not COLUMN_WORD.match(tokens[0].split("_")[0]):
        tokens = tokens[1:]  # label of the row-name column: "Subclass Budget Budget ..."
    if tokens and all(COLUMN_WORD.match(t.split("_")[0]) for t in tokens):
        return [t.replace("_", " ") for t in tokens]
    return None


def header_columns(years, extra, words):
    """Column names from a year header plus the optional line of column words under it."""
    if not words or len(words) < len(years):
        return years + [SHORT_COLUMN.get(t, t) for t in extra]
    columns = [f"{y} {w}" for y, w in zip(years, words)]
    below = words[len(years):]
    if extra and len(extra) == len(below):  # "Increase/" over "(Decrease)", "%" over "Change"
        return columns + [a + ("" if a.endswith("/") else " ") + b for a, b in zip(extra, below)]
    return columns + (below or extra)


def split_row(line):
    """'Real Estate Tax $ 145,141,754 $ 143,853,454' -> ('Real Estate Tax', ['$145,141,754', '$143,853,454'])"""
    tokens = re.sub(r"\$\s+", "$", line).split()
    values = []
    while tokens and TABLE_VALUE.match(tokens[-1]):
        values.insert(0, tokens.pop())
    label = " ".join(tokens)
    # a bare year among the "values" means this is a header/date line, not a row of numbers
    if not re.search(r"[A-Za-z]", label) or len(values) < 2 or any(YEAR.match(v) for v in values):
        return None
    return label, values


def label_table_rows(page):
    """Budget tables come out of pypdf as a year header plus bare rows of numbers, so a chunk
    with 'Real Estate Tax $ 145,141,754 $ 143,853,454 ...' doesn't say which number is 2025.
    Rewrite each row under a year header to name its columns:
    'Real Estate Tax — 2024 Estimate: $145,141,754; 2025 Budget: $143,853,454; ...'
    pypdf often puts a page's header *below* its rows, so rows above the first header use it."""
    lines = page.split("\n")
    headers = {}  # line index -> column names
    for i, line in enumerate(lines):
        found = table_header(line.strip())
        if found:
            words = column_words(lines[i + 1].strip()) if i + 1 < len(lines) else None
            headers[i] = header_columns(*found, words)
    if not headers:
        return page
    columns = headers[min(headers)]
    out = []
    for i, line in enumerate(lines):
        columns = headers.get(i, columns)
        row = split_row(line.strip())
        # trailing change columns are often blank, so a row may stop short of the last ones
        n_years = sum(1 for c in columns if YEAR.match(c[:4]))
        if row and n_years <= len(row[1]) <= len(columns):
            label, values = row
            line = f"{label} — " + "; ".join(f"{c}: {v}" for c, v in zip(columns, values))
        out.append(line)
    return "\n".join(out)


def get_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        # scanned pages are just an image, so pypdf finds (almost) no text -> OCR them
        if len(text.strip()) < MIN_PAGE_CHARS:
            print(f"OCR {os.path.basename(pdf_path)} page {i + 1}")
            text = ocr_page(pdf_path, i) or text
        if text.strip():
            pages.append(label_table_rows(text))
    return "\n\n".join(pages)


def dedupe(url_list):
    seen = {}
    for url, category in url_list:
        # treat #fragment / trailing-slash variants as the same page
        key = url.split("#")[0].rstrip("/")
        seen.setdefault(key, (url, category))
    return list(seen.values())


# URLs live in urls.yaml, grouped by category.
# The special follow_links section lists seed pages whose matching links get crawled too.
def load_urls(path="urls.yaml"):
    with open(path) as f:
        groups = yaml.safe_load(f)
    follow = groups.pop("follow_links", None) or []
    # `if url` skips blank "-" placeholder entries
    url_list = [(url.strip(), category) for category, urls in groups.items() for url in urls or [] if url]
    # seeds get crawled in round 1 like any other page
    url_list += [(spec["seed"], spec["category"]) for spec in follow]
    return dedupe(url_list), follow


def save(scraped_data):
    # write to a temp file first so a crash mid-write can't corrupt the json
    tmp = JSON_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(scraped_data, f, indent=4, ensure_ascii=False)
    os.replace(tmp, JSON_FILE)


def looks_blocked(text):
    if len(text) < MIN_CHARS:
        return True
    head = text[:2000].lower()
    return any(marker in head for marker in BLOCKED_MARKERS)


def download_pdf(url):
    path = os.path.join(PDF_DIR, url.rstrip("/").split("/")[-1])
    if os.path.exists(path):
        return
    print("downloading", url)
    # pittsburghpa.gov 403s a bare "Mozilla/5.0" UA but accepts an honest one
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp, open(path, "wb") as f:
            f.write(resp.read())
    except Exception as e:
        print("FAILED TO DOWNLOAD PDF", url, e)


def scrape_pdfs(scraped_data):
    for file in os.listdir(PDF_DIR):
        if not file.lower().endswith(".pdf"):
            continue
        path = os.path.join(PDF_DIR, file)
        try:
            text = get_pdf_text(path)
        except Exception as e:
            print("FAILED TO READ PDF", path, e)
            continue
        scraped_data[path] = {
            "page title": file,
            "source": path,
            "category": "pdf",
            "data": text,
        }


def social_links(result):
    # read from the raw html, not result.links: social icons usually live in the
    # <footer>, which excluded_tags strips before crawl4ai collects links
    links = []
    for href in re.findall(r'href=["\'](https?://[^"\'\s]+)', result.html or ""):
        host = href.split("/")[2].lower()
        if "share" in href or "/intent/" in href:  # "share this page" buttons, not the site's account
            continue
        if any(host == d or host.endswith("." + d) for d in SOCIAL_DOMAINS) and href not in links:
            links.append(href)
    return links


def html_links(result):
    # every href in the raw html, menus included: excluded_tags strips nav/header/footer
    # before crawl4ai collects result.links, and some sites only link subpages from the menu
    return [urljoin(result.url, h) for h in re.findall(r'href=["\']([^"\'\s]+)', result.html or "")]


def clean_page(link, data):
    # collection records end with a "More Artworks" list of 12 unrelated works,
    # which would make retrieval match the wrong painting
    if "collection.carnegieart.org/objects/" in link:
        data = data.split("More Artworks")[0]
    return data


# The collection site keeps its filters in page state (not the url), so the only way
# to list e.g. all paintings is the way a visitor would: click the filter, then page
# through the results. The js collects the object links into a <pre> we read back.
def cmoa_listing_js(classification):
    return f"""
    const sleep = ms => new Promise(r => setTimeout(r, ms));
    const objectLinks = () => [...document.querySelectorAll('a[href*="/objects/"]')].map(a => a.href);
    const label = 'label[for="classification-{classification}"]';
    for (let t = 0; t < 60 && !document.querySelector(label); t++) await sleep(500);
    document.querySelector(label).click();
    await sleep(4000);
    const seen = new Set();
    for (let page = 0; page < 500; page++) {{
        const before = objectLinks();
        before.forEach(h => seen.add(h));
        const next = [...document.querySelectorAll('a, button')].find(e => e.textContent.trim() === '→');
        if (!next) break;
        next.click();
        let changed = false;
        for (let t = 0; t < 20 && !changed; t++) {{
            await sleep(500);
            changed = objectLinks()[0] !== before[0];
        }}
        if (!changed) break;  // last page
    }}
    const pre = document.createElement('pre');
    pre.id = 'collected-links';
    pre.textContent = [...seen].join('\\n');
    document.body.appendChild(pre);
    """


async def discover_cmoa_objects(crawler):
    links = []
    for classification in CMOA_CLASSIFICATIONS:
        config = CrawlerRunConfig(js_code=cmoa_listing_js(classification),
                                  delay_before_return_html=1.0, page_timeout=600000)
        result = await crawler.arun(CMOA_HOME, config=config)
        found = re.search(r'<pre id="collected-links">(.*?)</pre>', result.html or "", re.S)
        found = found.group(1).split() if found else []
        print(f"carnegie museum of art: found {len(found)} {classification}")
        links += [(url, f"cmoa_{classification}") for url in found]
    return links


def cmoa_open_data_text(row):
    year = lambda d: (d or "")[:4]
    life = "–".join(y for y in (year(row["birth_date"]), year(row["death_date"])) if y)
    return "\n".join(line for line in [
        f"Creator: {row['full_name']} {row['nationality']} {life}".strip() if row["full_name"] else "",
        f"Date: {row['creation_date']}" if row["creation_date"] else "",
        f"Classification: {row['classification'].capitalize()}",
        f"Medium: {row['medium']}" if row["medium"] else "",
        f"Accession No.: {row['accession_number']}",
        f"Credit: {row['credit_line']}" if row["credit_line"] else "",
    ] if line)


def add_cmoa_open_data(scraped_data):
    """The collection website doesn't show acquisition dates, so add them (and the department)
    from the museum's CC0 open-data release, matched by accession number. Paintings the
    website listing missed are added as their own records.
    Gallery locations are left out: the release is a 2017 snapshot and locations have changed."""
    if not os.path.exists(CMOA_OPEN_DATA):
        print("downloading", CMOA_OPEN_DATA_URL)
        req = urllib.request.Request(CMOA_OPEN_DATA_URL, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=300) as resp, open(CMOA_OPEN_DATA, "wb") as f:
            f.write(resp.read())
    with open(CMOA_OPEN_DATA, encoding="utf-8") as f:
        rows = {r["accession_number"]: r for r in csv.DictReader(f)
                if r["classification"].lower() in CMOA_CLASSIFICATIONS}

    def extra(row):
        lines = [f"Acquisition date: {row['date_acquired']}" if row["date_acquired"] else "",
                 f"Department (2017): {row['department']}" if row["department"] else ""]
        return f"{CMOA_OPEN_DATA_MARKER}\n" + "\n".join(line for line in lines if line)

    enriched = set()
    for link, doc in scraped_data.items():
        if "collection.carnegieart.org/objects/" not in link:
            continue
        found = re.search(r"Accession No\.\s*(\S+)", doc["data"])
        row = found and rows.get(found.group(1))
        if row:
            # split off a previous run's block so re-running doesn't stack them
            doc["data"] = doc["data"].split(CMOA_OPEN_DATA_MARKER)[0].rstrip() + "\n\n" + extra(row)
            enriched.add(row["accession_number"])
    added = 0
    for accession, row in rows.items():
        if accession in enriched:
            continue
        scraped_data[f"cmoa-open-data:{accession}"] = {
            "page title": row["title"],
            "source": CMOA_OPEN_DATA_URL,
            "category": "cmoa_paintings_open_data",
            "data": f"## {row['full_name']}\n### {row['title']}\n{cmoa_open_data_text(row)}\n\n{extra(row)}",
        }
        added += 1
    print(f"carnegie museum of art open data: {len(enriched)} records enriched, {added} added")


async def crawl(crawler, url_list, configs, scraped_data, failed):
    """Crawl url_list into scraped_data; returns {url: {"content": hrefs, "all": hrefs}} for the
    pages that worked ("content" skips nav/header/footer menus, "all" includes them)."""
    categories = {url: category for url, category in url_list}
    page_links = {}
    if not categories:
        return page_links
    done = 0
    # not arun_many: some pages (steelers.com rosters, a few trustarts.org pages) hang forever
    # despite page_timeout, and arun_many has no way to give up on them, so the run never ends.
    # Each page here gets a hard time limit instead.
    sem = asyncio.Semaphore(CONCURRENCY)

    async def fetch(url):
        config = next(c for c in configs if c.is_match(url))  # first matching config, like arun_many
        async with sem:
            try:
                return url, await asyncio.wait_for(crawler.arun(url, config=config), PAGE_HARD_TIMEOUT)
            except asyncio.TimeoutError:
                return url, None

    for next_done in asyncio.as_completed([fetch(url) for url in categories]):
        url, result = await next_done
        done += 1
        if result is None:
            print(f"TIMED OUT after {PAGE_HARD_TIMEOUT}s", url)
            failed.append(url)
            continue
        link = result.url
        try:
            if not result.success:
                print("FAILED TO GET ANYTHING FROM", link, "-", result.error_message)
                failed.append(link)
                continue
            if not result.markdown:
                print("UNABLE TO GET MARKDOWN FROM", link)
                failed.append(link)
                continue

            data = result.markdown.raw_markdown or ""
            if looks_blocked(data) or result.status_code in (403, 404, 429, 503):
                print("LOOKS BLOCKED / EMPTY", link, "status", result.status_code, "chars", len(data))
                failed.append(link)
                continue
            data = clean_page(link, data)

            socials = social_links(result)
            if socials:
                data += "\n\nSocial media: " + " ".join(socials)

            title = (result.metadata or {}).get("title") or link.rstrip("/").split("/")[-1]
            scraped_data[link] = {
                "page title": title.strip(),
                "source": link,
                "category": categories.get(link, "unknown"),
                "data": data,
                "social links": socials,
            }
            content = result.links.get("internal", []) + result.links.get("external", [])
            page_links[link] = {"content": [l.get("href", "") for l in content], "all": html_links(result)}
        except Exception as e:
            print("ERROR PROCESSING", link, e)
            failed.append(link)

        if done % SAVE_EVERY == 0:
            save(scraped_data)
            print(f"saved progress: {done}/{len(categories)} pages")
    return page_links


def followed_links(follow, page_links):
    found = []
    for spec in follow:
        hrefs = page_links.get(spec["seed"], {}).get("all" if spec.get("menus") else "content", [])
        patterns = spec["pattern"] if isinstance(spec["pattern"], list) else [spec["pattern"]]
        matches = [h for h in hrefs if any(fnmatch(h, p) for p in patterns)]
        print(f"following {len(set(matches))} links from {spec['seed']}")
        found += [(h, spec["category"]) for h in matches]
    return found


async def main(only_new=False):
    os.makedirs(PDF_DIR, exist_ok=True)

    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, "r") as f:
            scraped_data = json.load(f)
    else:
        scraped_data = {}

    url_list, follow = load_urls()
    if only_new:
        # same #fragment / trailing-slash matching as dedupe(). follow_links seeds are kept:
        # round 2 needs their links, and listing pages (e.g. events) gain new links over time
        have = {url.split("#")[0].rstrip("/") for url in scraped_data}
        seeds = {spec["seed"] for spec in follow}
        url_list = [(url, c) for url, c in url_list
                    if url.split("#")[0].rstrip("/") not in have or url in seeds]
        print(f"--only-new: {len(url_list)} urls from urls.yaml not scraped yet")
    # crawl4ai can't read pdfs, so download them and parse with pypdf instead
    for url, _ in url_list:
        if url.lower().endswith(".pdf"):
            download_pdf(url)
    url_list = [(url, category) for url, category in url_list if not url.lower().endswith(".pdf")]

    # NOTE: don't use exclude_external_links -- crawl4ai deletes the whole <a>
    # (and the text after it), so restaurant/event names that link offsite vanish.
    # ignore_links keeps the link text but drops the urls instead.
    # NOTE: no PruningContentFilter either -- it threw away wikipedia infoboxes and
    # tables (e.g. "6 Super Bowls"), which is exactly the kind of fact questions ask about.
    # NOTE: don't exclude "form" -- pittsburghpa.gov wraps its whole page in one <form>
    # NOTE: no exclude_social_media_links -- same problem as exclude_external_links,
    # it deletes the link text. Social links are saved separately instead (social_links).
    run_config = CrawlerRunConfig(
        excluded_tags=["nav", "footer", "header", "aside", "script", "style"],
        # wikipedia: citation markers, edit links, navboxes, category list at the bottom
        excluded_selector=".reference, .mw-editsection, .navbox, .reflist, #toc, #catlinks, .printfooter",
        remove_overlay_elements=True,  # cookie banners, popups
        markdown_generator=DefaultMarkdownGenerator(
            options={"ignore_links": True, "ignore_images": True},
        ),
    )
    # javascript-heavy pages: scroll to the bottom to trigger lazy loading, then wait
    dynamic_config = run_config.clone(
        url_matcher=DYNAMIC_SITES,
        scan_full_page=True,
        scroll_delay=0.5,
        delay_before_return_html=3.0,
        page_timeout=90000,
    )
    load_more_configs = [
        dynamic_config.clone(
            url_matcher=pattern,
            js_code=click_load_more(selector, max_clicks),
            page_timeout=180000,
        )
        for pattern, selector, max_clicks in LOAD_MORE_SITES
    ]
    # collection records are rendered by javascript; wait until the record fields show up
    cmoa_config = run_config.clone(
        url_matcher="*collection.carnegieart.org/objects/*",
        wait_for="js:() => document.body.innerText.includes('Classification')",
        delay_before_return_html=0.5,
    )
    keep_overlay_config = dynamic_config.clone(url_matcher=KEEP_OVERLAY_SITES, remove_overlay_elements=False)
    # first matching config wins, so the most specific ones go first
    configs = [cmoa_config] + load_more_configs + [keep_overlay_config, dynamic_config, run_config]

    failed = []

    # scraped data dict
    # key = url
    # value = markdown data dict (page title, source, category, data, social links)
    try:
        async with AsyncWebCrawler() as crawler:
            # round 1: everything in urls.yaml
            page_links = await crawl(crawler, url_list, configs, scraped_data, failed)

            # round 2: links from the follow_links seeds + museum collection records.
            # these pages don't change, so skip any we already have from a previous run
            # (--only-new skips the slow museum listing: those records are already in)
            round2 = followed_links(follow, page_links)
            if not only_new:
                round2 += await discover_cmoa_objects(crawler)
            round2 = [(url, c) for url, c in dedupe(round2) if url not in scraped_data]
            for url, _ in round2:
                if url.lower().endswith(".pdf"):
                    download_pdf(url)
            round2 = [(url, c) for url, c in round2 if not url.lower().endswith(".pdf")]
            print(f"round 2: crawling {len(round2)} new pages")
            await crawl(crawler, round2, configs, scraped_data, failed)
    finally:
        # PDF SCRAPING... (includes any pdfs downloaded in round 2)
        scrape_pdfs(scraped_data)
        add_cmoa_open_data(scraped_data)
        # runs on success, crash, or ctrl-c so nothing scraped so far is lost
        save(scraped_data)

    print("Finished simple scraping", len(scraped_data), "documents")
    if failed:
        print(len(failed), "failed / blocked:")
        for link in failed:
            print("  ", link)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape urls.yaml (+ PDFs) into data/scraped_data.json")
    parser.add_argument("--only-new", action="store_true",
                        help="only crawl urls.yaml entries not already in scraped_data.json")
    parser.add_argument("--open-data-only", action="store_true",
                        help="no crawling: just (re)apply the museum open data to scraped_data.json")
    parser.add_argument("--pdfs-only", action="store_true",
                        help="no crawling: just re-parse data/pdfs into scraped_data.json")
    args = parser.parse_args()
    if args.open_data_only or args.pdfs_only:
        with open(JSON_FILE) as f:
            scraped_data = json.load(f)
        if args.pdfs_only:
            scrape_pdfs(scraped_data)
        else:
            add_cmoa_open_data(scraped_data)
        save(scraped_data)
    else:
        asyncio.run(main(args.only_new))
