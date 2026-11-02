import re

# Three chunking strategies, all written by hand (no LangChain text splitters):
#   fixed    - sliding window of N words with overlap, ignores document structure
#   sentence - packs whole sentences into chunks of ~N words, overlapping by a few sentences
#   section  - splits on markdown headings first, then packs paragraphs/lines inside each
#              section; anything too long falls back to sentence packing, then to words.
# Every chunk gets "page title > heading" prepended so short chunks like
# "Tickets: $25" still say what they're about.

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
# split after . ! ? (optionally followed by a quote/bracket) when the next thing looks like a new sentence.
# not perfect (e.g. "St. Louis"), but good enough and needs no punkt download
SENT_RE = re.compile(r"(?<=[.!?])[\"')\]]?\s+(?=[\"'(\[]?[A-Z0-9])")
# lines that are just page furniture
JUNK_LINE_RE = re.compile(r"^(skip to (main )?content|menu|search|share|close|back to top|loading\.*|\W*)$", re.I)


def clean_text(text):
    text = text.replace(" ", " ").replace("\r", "")
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        line = line.lstrip(">").strip()  # markdown blockquote markers
        line = re.sub(r"[ \t]+", " ", line)
        if JUNK_LINE_RE.match(line):
            lines.append("")  # keep paragraph breaks
            continue
        lines.append(line)
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def split_sentences(text):
    sents = []
    for line in text.split("\n"):
        sents.extend(s.strip() for s in SENT_RE.split(line) if s.strip())
    return sents


def n_words(text):
    return len(text.split())


def split_words(text, size, overlap):
    words = text.split()
    if len(words) <= size:
        return [" ".join(words)] if words else []
    step = max(1, size - overlap)
    out = []
    for start in range(0, len(words), step):
        out.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return out


def pack(units, size, overlap_units=0, joiner=" "):
    """Greedily pack text units (sentences/paragraphs) into chunks of <= size words.
    The last `overlap_units` units of each chunk are repeated at the start of the next.
    A single unit longer than `size` is split by words."""
    chunks, cur, cur_len = [], [], 0
    for unit in units:
        ulen = n_words(unit)
        if ulen > size:
            if cur:
                chunks.append(joiner.join(cur))
                cur, cur_len = [], 0
            chunks.extend(split_words(unit, size, size // 5))
            continue
        if cur and cur_len + ulen > size:
            chunks.append(joiner.join(cur))
            cur = cur[-overlap_units:] if overlap_units else []
            cur_len = sum(n_words(u) for u in cur)
            # overlap alone can't push us over the limit forever
            if cur_len + ulen > size:
                cur, cur_len = [], 0
        cur.append(unit)
        cur_len += ulen
    if cur:
        chunks.append(joiner.join(cur))
    return chunks


def split_sections(text):
    """Yield (heading_path, body) for each markdown section."""
    path, body = [], []
    for line in text.split("\n"):
        m = HEADING_RE.match(line)
        if m:
            if any(b.strip() for b in body):
                yield list(path), "\n".join(body).strip()
            level, title = len(m.group(1)), m.group(2).strip(" #*_")
            path = path[:level - 1] + [title] if title else path[:level - 1]
            body = []
        else:
            body.append(line)
    if any(b.strip() for b in body):
        yield list(path), "\n".join(body).strip()


def chunk_fixed(text, size, overlap):
    return [([], c) for c in split_words(text, size, overlap)]


def chunk_sentence(text, size, overlap_sents=2):
    return [([], c) for c in pack(split_sentences(text), size, overlap_sents)]


def chunk_section(text, size, overlap_sents=1, min_words=25):
    out = []
    for path, body in split_sections(text):
        # paragraphs, but event listings / tables are one item per line, so treat
        # each line as a unit; long lines get sentence-split
        units = []
        for line in body.split("\n"):
            if not line.strip():
                continue
            units.extend(split_sentences(line) if n_words(line) > size else [line])
        for c in pack(units, size, overlap_sents, joiner="\n"):
            out.append((path, c))
    # merge tiny chunks into the previous one from the same document, so a heading
    # with one line under it doesn't become its own useless chunk
    merged = []
    for path, c in out:
        if merged and n_words(c) < min_words and n_words(merged[-1][1]) + n_words(c) <= size * 1.25:
            prev_path, prev = merged[-1]
            header = " > ".join(path)
            merged[-1] = (prev_path, prev + "\n" + (header + ": " if header and path != prev_path else "") + c)
        else:
            merged.append((path, c))
    return merged


def chunk_document(doc, strategy="section", size=200, overlap=40):
    text = clean_text(doc["data"])
    if not text:
        return []
    if strategy == "fixed":
        pieces = chunk_fixed(text, size, overlap)
    elif strategy == "sentence":
        pieces = chunk_sentence(text, size)
    elif strategy == "section":
        pieces = chunk_section(text, size)
    else:
        raise ValueError(f"unknown chunking strategy {strategy}")

    title = doc.get("page title") or doc["source"]
    chunks = []
    for i, (path, body) in enumerate(pieces):
        header = " > ".join([title] + [p for p in path if p and p != title])
        chunks.append({
            "id": f"{doc['source']}#{i}",
            "source": doc["source"],
            "title": title,
            "text": f"{header}\n{body}",
        })
    return chunks
