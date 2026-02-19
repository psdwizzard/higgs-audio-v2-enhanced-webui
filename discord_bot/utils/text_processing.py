import re
from typing import List, Tuple


def detect_chapters(text: str) -> List[Tuple[str, str]]:
    """Split text into (chapter_title, chapter_body) tuples.

    Recognises patterns like:
      Chapter 1, Chapter One, CHAPTER I, Part 1, Part One,
      numbered headings like "1. Title", and markdown headings "# Title".
    Falls back to a single chapter if none are detected.
    """
    # Pattern matches common chapter/part headings on their own line
    pattern = re.compile(
        r"^(?:"
        r"(?:chapter|part)\s+(?:\d+|[IVXLCDM]+|one|two|three|four|five|six|seven|eight|nine|ten|"
        r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty"
        r"(?:[- ](?:one|two|three|four|five|six|seven|eight|nine))?)"
        r"(?:\s*[:.\-]\s*.*)?"
        r"|"
        r"\d+\.\s+\S.*"
        r"|"
        r"#{1,3}\s+\S.*"
        r")$",
        re.IGNORECASE | re.MULTILINE,
    )

    matches = list(pattern.finditer(text))

    if not matches:
        return [("Full Text", text.strip())]

    chapters = []
    for i, match in enumerate(matches):
        title = match.group(0).strip().lstrip("# ")
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            chapters.append((title, body))

    # If there's text before the first heading, prepend it as "Introduction"
    if matches and matches[0].start() > 0:
        intro = text[: matches[0].start()].strip()
        if intro:
            chapters.insert(0, ("Introduction", intro))

    # Merge short chapters (< 100 chars) into the next chapter
    merged = []
    carry = ""
    carry_title = ""
    for idx, (title, body) in enumerate(chapters):
        if carry:
            body = carry + "\n\n" + body
            title = carry_title + " / " + title if carry_title != "Introduction" else title
            carry = ""
            carry_title = ""
        if len(body) < 300 and idx < len(chapters) - 1:
            carry = body
            carry_title = title
        else:
            merged.append((title, body))
    if carry:
        if merged:
            prev_title, prev_body = merged[-1]
            merged[-1] = (prev_title, prev_body + "\n\n" + carry)
        else:
            merged.append((carry_title, carry))

    # Split oversized chapters into ~10k char parts, cutting at sentence boundaries
    final = []
    for title, body in (merged if merged else [("Full Text", text.strip())]):
        if len(body) <= 10000:
            final.append((title, body))
            continue
        parts = _split_at_sentence(body, target_size=10000)
        if len(parts) == 1:
            final.append((title, parts[0]))
        else:
            for pi, part in enumerate(parts, 1):
                final.append((f"{title} (Part {pi})", part))

    return final if final else [("Full Text", text.strip())]


def _split_at_sentence(text: str, target_size: int = 10000) -> List[str]:
    """Split text into parts of roughly *target_size* chars, always ending at a sentence boundary."""
    # Find all sentence-ending positions (after . ! or ?)
    sentence_ends = [m.end() for m in re.finditer(r"[.!?]+[\s\n]", text)]
    if not sentence_ends:
        # No sentence boundaries found — can't split cleanly, return as-is
        return [text.strip()]

    parts: List[str] = []
    start = 0
    for end_pos in sentence_ends:
        if end_pos - start >= target_size:
            parts.append(text[start:end_pos].strip())
            start = end_pos
    # Remainder
    if start < len(text):
        remaining = text[start:].strip()
        if remaining:
            # If remainder is short, merge it into the last part
            if parts and len(remaining) < 300:
                parts[-1] = parts[-1] + " " + remaining
            else:
                parts.append(remaining)

    return parts if parts else [text.strip()]


def smart_chunk_text(text: str, max_chunk_size: int = 200) -> List[str]:
    """Smart text chunking that respects sentence boundaries and paragraphs."""
    paragraphs = text.split("\n\n")
    chunks: List[str] = []

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        if len(paragraph) <= max_chunk_size:
            chunks.append(paragraph)
            continue

        sentences: List[str] = []
        sentence_parts = re.split(r"([.!?]+)", paragraph)

        current_sentence = ""
        for i in range(0, len(sentence_parts), 2):
            if i < len(sentence_parts):
                current_sentence = sentence_parts[i].strip()
                if i + 1 < len(sentence_parts):
                    current_sentence += sentence_parts[i + 1]
                if current_sentence.strip():
                    sentences.append(current_sentence.strip())

        current_chunk = ""
        for sentence in sentences:
            if current_chunk and len(current_chunk) + len(sentence) + 1 > max_chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

    return chunks


def clean_google_doc_text(text: str) -> str:
    """Remove Google Docs export artefacts from plain text."""
    # Remove BOM
    text = text.lstrip("\ufeff")
    # Normalise line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse 3+ blank lines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Normalize curly/smart quotes to straight equivalents
    text = text.translate(
        str.maketrans(
            {
                "\u201c": '"',   # left double curly quote -> straight
                "\u201d": '"',   # right double curly quote -> straight
                "\u201e": '"',   # double low-9 quote -> straight
                "\u201f": '"',   # double high reversed-9 quote -> straight
                "\u2018": "'",   # left single curly quote -> straight apostrophe
                "\u2019": "'",   # right single curly quote -> straight apostrophe
            }
        )
    )
    # Remove double quotes
    text = text.replace('"', " ")
    # Remove standalone single quotes but keep mid-word apostrophes (don't, can't, it's)
    text = re.sub(r"(?<!\w)'|'(?!\w)", " ", text)
    # Replace ellipses with a space (avoids TTS artifacts)
    text = text.replace("...", " ")
    text = text.replace("\u2026", " ")
    # Normalize whitespace but preserve paragraph breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
