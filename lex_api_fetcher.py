import requests
import json
import sys
import traceback
import csv
import os
import re
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import time
import sqlite3

# ===== SEARCH CONFIGURATION =====
# Start and end dates for document search
SEARCH_START_YEAR1 = 2010
SEARCH_START_MONTH1 = 1  # January
SEARCH_END_YEAR1 = 2024
SEARCH_END_MONTH1 = 12   # December

print("Starting Lex API fetcher script...")

# Helper functions for text processing

def find_sentence_start(text, position):
    """
    Find the start position of the sentence containing the given position.
    Looks backward for typical sentence-ending punctuation followed by space and capital letter.
    """
    # Look backward for standard sentence terminators (period, exclamation, question mark)
    # followed by space and capital letter
    sentence_terminators = ['. ', '! ', '? ', '.\n', '!\n', '?\n']
    
    # Maximum chars to look back (to avoid excessive processing)
    max_lookback = 1000
    search_start = max(0, position - max_lookback)
    
    # Text segment to search in
    text_segment = text[search_start:position]
    
    # Find the last sentence terminator
    last_term_pos = -1
    for term in sentence_terminators:
        pos = text_segment.rfind(term)
        if pos > last_term_pos:
            last_term_pos = pos
    
    if last_term_pos != -1:
        # Add the length of the terminator to get to the start of next sentence
        return search_start + last_term_pos + min(2, len(text_segment) - last_term_pos)
    
    # If no sentence terminator found, look for paragraph markers
    paragraph_markers = ['\n\n', '\r\n\r\n']
    for marker in paragraph_markers:
        pos = text_segment.rfind(marker)
        if pos != -1:
            return search_start + pos + len(marker)
    
    # If nothing found, return beginning of search segment
    return search_start

def find_sentence_end(text, position):
    """
    Find the end position of the sentence containing the given position.
    Looks forward for typical sentence-ending punctuation followed by space and capital letter.
    """
    # Maximum chars to look ahead (to avoid excessive processing)
    max_lookahead = 1000
    search_end = min(len(text), position + max_lookahead)
    
    # Text segment to search in
    text_segment = text[position:search_end]
    
    # Look for sentence terminators
    sentence_terminators = ['. ', '! ', '? ', '.\n', '!\n', '?\n']
    first_term_pos = len(text_segment)
    
    for term in sentence_terminators:
        pos = text_segment.find(term)
        if pos != -1 and pos < first_term_pos:
            # Check if this is not part of a common abbreviation
            if not is_abbreviation(text_segment[:pos+1]):
                first_term_pos = pos
    
    if first_term_pos < len(text_segment):
        # Add the terminator to the end position (including period and space)
        return position + first_term_pos + min(2, len(text_segment) - first_term_pos)
    
    # If no sentence terminator found, look for paragraph markers
    paragraph_markers = ['\n\n', '\r\n\r\n']
    for marker in paragraph_markers:
        pos = text_segment.find(marker)
        if pos != -1:
            return position + pos
    
    # If nothing found, return end of search segment
    return search_end

def is_abbreviation(text):
    """
    Check if the given text ends with a common abbreviation
    to avoid splitting sentences incorrectly.
    """
    common_abbr = [
        "art.", "ust.", "pkt.", "lit.", "tj.", "tzn.", "m.in.", "np.",
        "r.", "nr.", "str.", "zob.", "proc.", "ok.", "Dr.", "prof.", "al."
    ]
    
    for abbr in common_abbr:
        if text.strip().endswith(abbr):
            return True
    return False

# OAuth configuration with provided credentials
client_id = "6fae0f29-b734-45e9-9e8e-68e261a6a0ce"
client_secret = "q.6.G-PEPSMnat~3gvK6I.2L4SK0J.BrzoskaFound.wkapi"
token_url = "https://borg.wolterskluwer.pl/core/v2.0/connect/token"

# User credentials required for ROPC flow
username = "michal.wyrebkowski@sprawdzamy.com"
password = "cay-wvt9UNA*hqr7cnc"

# Prepare data for token request using the ROPC flow
token_data = {
    "client_id": client_id,
    "client_secret": client_secret,
    "grant_type": "password",
    "username": username,
    "password": password,
    "scope": "api://7f0641a3-1b27-45a5-b704-5d812c4c47a8/access_as_user offline_access"
}

# Request an access token
print("Requesting access token...")
try:
    token_response = requests.post(token_url, data=token_data, timeout=30)
    token_response.raise_for_status()
    token_json = token_response.json()
    access_token = token_json.get("access_token")
    print("Access token obtained:", access_token[:30] + "..." if access_token else "None")
except Exception as e:
    print(f"Error obtaining token: {str(e)}")
    sys.exit(1)

documents_api_url = "https://api.lex.pl/v4/documents"

# Set the authorization header with the access token
headers = {
    "Authorization": f"Bearer {access_token}",
    "Content-Type": "application/json"
}

# CSV file for results
csv_filename = "legal_passages_time_terms.csv"
# Fields for CSV output, including the passage and context
csv_fields = ["nro", "title", "signature", "documentType", "publicationDate", 
             "enactmentDate", "effectiveDate", "publisher", "passage", "article", "full_context", "article_context", "search_term", "source", "debug"]

# JSON output for Perplexity processing
json_output_file = "passages_for_perplexity.json"

# Search phrases - list of all phrases to search for
SEARCH_PHRASES = [
    "w terminie",
    "należy NEAR w ciągu",
    "powinien NEAR terminie",
    "w okresie NEAR od dnia",
    "nie później niż",
    "niezwłocznie",
    "obowiązek NEAR terminu",
    "przed upływem terminu",
    "z zachowaniem terminu",
    "w terminie określonym w",
    "w terminie wskazanym",
    "bez zbędnej zwłoki",
    "bezzwłocznie",
    "niezwłocznie po",
    "w terminie \\d+ dni od dnia"  # Using regex syntax for the API
]

# Add these constants near other file path definitions
FETCHER_DB_FILE = "lex_passages.db"

# Add this constant at the top with other configuration
API_REQUEST_DELAY = 0.05  # Delay between API requests in seconds

# Add this function to initialize the database
def initialize_fetcher_database():
    """Create SQLite database and necessary tables for raw Lex data"""
    conn = sqlite3.connect(FETCHER_DB_FILE)
    cursor = conn.cursor()
    
    # Create table for legal passages without analysis fields
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS raw_passages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nro TEXT,
        title TEXT,
        signature TEXT,
        document_type TEXT,
        publication_date TEXT,
        enactment_date TEXT,
        effective_date TEXT,
        publisher TEXT,
        passage TEXT,
        article TEXT,
        full_context TEXT,
        article_context TEXT,
        search_term TEXT,
        source TEXT,
        debug TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Now let's check if the table exists but is missing the debug column
    # This handles existing database files that were created before this update
    try:
        # Check if debug column already exists
        cursor.execute("PRAGMA table_info(raw_passages)")
        columns = [info[1] for info in cursor.fetchall()]
        
        if "debug" not in columns:
            print("Adding 'debug' column to existing database table...")
            cursor.execute("ALTER TABLE raw_passages ADD COLUMN debug TEXT")
            conn.commit()
    except Exception as e:
        print(f"Error checking/updating database schema: {str(e)}")
    
    conn.commit()
    return conn

# Add this function to store a passage in the database
def store_passage_in_database(conn, passage):
    """Store a passage in the SQLite database"""
    cursor = conn.cursor()
    
    try:
        # Extract data from the passage
        if "document_metadata" in passage and "match_data" in passage:
            meta = passage["document_metadata"]
            match = passage["match_data"]
            
            nro = meta.get("nro", "")
            title = meta.get("title", "")
            signature = meta.get("signature", "")
            document_type = meta.get("documentType", "")
            publication_date = meta.get("publicationDate", "")
            enactment_date = meta.get("enactmentDate", "")
            effective_date = meta.get("effectiveDate", "")
            
            # Ensure publisher is a string, even if it's a dictionary
            publisher_data = meta.get("publisher", "")
            if isinstance(publisher_data, dict):
                publisher = json.dumps(publisher_data, ensure_ascii=False)
            else:
                publisher = str(publisher_data)
            
            passage_text = match.get("passage", "")
            article = match.get("article", "")
            full_context = match.get("full_context", "")
            article_context = match.get("article_context", "")
            search_term = match.get("search_term", "")
            source = match.get("source", "")
            debug = match.get("debug", "")
        else:
            nro = passage.get("nro", "")
            title = passage.get("title", "")
            signature = passage.get("signature", "")
            document_type = passage.get("documentType", "")
            publication_date = passage.get("publicationDate", "")
            enactment_date = passage.get("enactmentDate", "")
            effective_date = passage.get("effectiveDate", "")
            
            # Ensure publisher is a string here too
            publisher_data = passage.get("publisher", "")
            if isinstance(publisher_data, dict):
                publisher = json.dumps(publisher_data, ensure_ascii=False)
            else:
                publisher = str(publisher_data)
            
            passage_text = passage.get("passage", "")
            article = passage.get("article", "")
            full_context = passage.get("full_context", "")
            article_context = passage.get("article_context", "")
            search_term = passage.get("search_term", "")
            source = passage.get("source", "")
            debug = passage.get("debug", "")
        
        # Insert into database
        cursor.execute('''
        INSERT INTO raw_passages (
            nro, title, signature, document_type, publication_date, enactment_date, 
            effective_date, publisher, passage, article, full_context, article_context, 
            search_term, source, debug
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            str(nro), str(title), str(signature), str(document_type), 
            str(publication_date), str(enactment_date), str(effective_date), 
            publisher, str(passage_text), str(article), str(full_context), 
            str(article_context), str(search_term), str(source), str(debug)
        ))
        
        conn.commit()
        return True
    except Exception as e:
        print(f"Error storing passage in database: {str(e)}")
        print(traceback.format_exc())
        conn.rollback()
        return False

# Function to generate monthly date ranges without dateutil
def generate_monthly_ranges(start_year, start_month, end_year, end_month):
    """Generate monthly date ranges in reverse order"""
    ranges = []
    
    # Generate all months from start to end
    current_year = end_year
    current_month = end_month
    
    while (current_year > start_year) or (current_year == start_year and current_month >= start_month):
        # Last day of current month
        if current_month == 12:
            next_year = current_year + 1
            next_month = 1
        else:
            next_year = current_year
            next_month = current_month + 1
            
        # Get the last day of the current month
        if current_month in [4, 6, 9, 11]:
            last_day = 30
        elif current_month == 2:
            # Check for leap year
            if (current_year % 4 == 0 and current_year % 100 != 0) or (current_year % 400 == 0):
                last_day = 29
            else:
                last_day = 28
        else:
            last_day = 31
            
        start_date = f"{current_year}-{current_month:02d}-01"
        end_date = f"{current_year}-{current_month:02d}-{last_day:02d}"
        
        ranges.append((start_date, end_date))
        
        # Move to the previous month
        if current_month == 1:
            current_year -= 1
            current_month = 12
        else:
            current_month -= 1
            
    return ranges

# Function to handle NEAR queries (finds terms within 100 characters of each other)
def is_near_match(text, term1, term2, max_distance=100):
    """Check if term1 and term2 appear within max_distance characters of each other"""
    text_lower = text.lower()
    term1_lower = term1.lower()
    term2_lower = term2.lower()
    
    pos1 = text_lower.find(term1_lower)
    while pos1 >= 0:
        # Look for term2 within max_distance characters before or after term1
        start_check = max(0, pos1 - max_distance)
        end_check = min(len(text_lower), pos1 + len(term1_lower) + max_distance)
        segment = text_lower[start_check:end_check]
        
        if term2_lower in segment:
            return True
            
        # Move to next occurrence of term1
        pos1 = text_lower.find(term1_lower, pos1 + 1)
    
    return False

# Function to extract relevant passages containing the search phrase
def extract_relevant_passages(content, search_phrase):
    """Extract passages containing the search phrase from document content"""
    passages = []
    full_contexts = []
    structured_contexts = []
    article_contexts = []
    debug_contexts = []
    
    if not content:
        return passages, full_contexts, structured_contexts, article_contexts, debug_contexts
    
    # Preprocess text: replace multiple spaces and line breaks with single ones
    text = re.sub(r'\s+', ' ', content)
    text = re.sub(r'\n+', '\n', text)
    
    # Find all occurrences of the search phrase
    search_regex = re.compile(search_phrase, re.IGNORECASE)
    matches = list(search_regex.finditer(text))
    
    for match in matches:
        start_pos = match.start()
        end_pos = match.end()
        
        # Extract a window around the match for the passage
        passage_window = 100  # characters to include in the passage
        passage_start = max(0, start_pos - passage_window)
        passage_end = min(len(text), end_pos + passage_window)
        passage = text[passage_start:passage_end].strip()
        
        # Find sentence boundaries for full context
        sentence_start = find_sentence_start(text, start_pos)
        sentence_end = find_sentence_end(text, end_pos)
        full_context = text[sentence_start:sentence_end].strip()
        
        # Extract the full article context with additional article information
        article_text, article_info = extract_article_context(text, start_pos)
        
        # Create debug context - include previous and next article references
        debug_context = f"PREVIOUS: {article_info['previous_article_reference'] or 'None'} | " + \
                        f"CURRENT: {article_info['current_article_reference'] or 'Unknown'} | " + \
                        f"NEXT: {article_info['next_article_reference'] or 'None'}\n\n" + \
                        article_text[:300] + "..."  # First 300 chars of article for preview
        
        # Create enhanced structured context data
        structured_context = {
            "sentence": full_context,
            "article_reference": article_info["current_article_reference"] or extract_article_reference(article_text),
            "position_in_doc": start_pos,
            "previous_article": article_info["previous_article_reference"],
            "next_article": article_info["next_article_reference"]
        }
        
        passages.append(passage)
        full_contexts.append(full_context)
        structured_contexts.append(structured_context)
        article_contexts.append(article_text)
        debug_contexts.append(debug_context)
    
    return passages, full_contexts, structured_contexts, article_contexts, debug_contexts

def detect_article_boundaries(text):
    """
    Detects article boundaries in legal text.
    Returns a list of tuples (start_position, end_position, article_number).
    """
    # Regular expressions for different article header patterns
    header_patterns = [
        # Common article patterns
        r'(?:^|\n|\r)\s*Art(?:ykuł|\.)\s+(\d+[a-z]*)\.?',
        r'(?:^|\n|\r)\s*§\s*(\d+[a-z]*)\.?',
        r'(?:^|\n|\r)\s*Artykuł\s+(\d+[a-z]*)\.?',
        # Numbered sections
        r'(?:^|\n|\r)\s*(\d+)\.\s+',  # e.g., "1. Some text"
        # Special section formats
        r'(?:^|\n|\r)\s*Rozdział\s+(\d+[a-z]*)\.?',
    ]
    
    # Combine patterns with OR
    combined_pattern = '|'.join(f'({p})' for p in header_patterns)
    
    # Find all potential article headers
    candidates = []
    for match in re.finditer(combined_pattern, text):
        # Extract the full header and the numeric portion
        full_header = match.group(0)
        
        # Determine which pattern matched and extract the numeric value
        for i, group in enumerate(match.groups()):
            if group:
                # Extract the numeric value from whichever group matched
                # This will be in a capture group inside the pattern
                numeric_part = re.search(r'(\d+[a-z]*)', group)
                if numeric_part:
                    numeric_value = numeric_part.group(1)
                    candidates.append({
                        'position': match.start(),
                        'header': full_header.strip(),
                        'numeric_value': numeric_value,
                        'pattern_index': i
                    })
                break
    
    # Sort candidates by position in the text
    candidates.sort(key=lambda x: x['position'])
    
    # Process candidates to find valid sequences
    valid_articles = []
    current_sequence = []
    
    for i, candidate in enumerate(candidates):
        # Start a new sequence or extend the current one
        if not current_sequence:
            current_sequence.append(candidate)
            continue
        
        # Check if this candidate follows a consistent pattern with the sequence
        pattern_consistent = candidate['pattern_index'] == current_sequence[0]['pattern_index']
        
        # Add to current sequence if pattern is consistent
        if pattern_consistent:
            current_sequence.append(candidate)
        else:
            # If we've found a substantial sequence, save it
            if len(current_sequence) >= 2:
                valid_articles.extend(current_sequence)
            # Start a new sequence
            current_sequence = [candidate]
    
    # Add the last sequence if valid
    if len(current_sequence) >= 2:
        valid_articles.extend(current_sequence)
    
    # Convert to boundary tuples (start, end, article_number)
    boundaries = []
    for i, article in enumerate(valid_articles):
        start_pos = article['position']
        # End position is the start of the next article or the end of text
        end_pos = valid_articles[i+1]['position'] if i < len(valid_articles)-1 else len(text)
        boundaries.append((start_pos, end_pos, article['numeric_value']))
    
    return boundaries

def extract_article_for_position(text, position, article_boundaries):
    """
    Extract the article text containing the given position.
    """
    for start_pos, end_pos, article_number in article_boundaries:
        if start_pos <= position < end_pos:
            return text[start_pos:end_pos].strip()
    
    # If no specific article is found, return a reasonable context
    context_size = 1000  # characters
    context_start = max(0, position - context_size//2)
    context_end = min(len(text), position + context_size//2)
    return text[context_start:context_end].strip()

def extract_full_article_containing_position(text, position):
    """
    Extract the complete article that contains the given position in the text.
    
    Args:
        text (str): The full text of the document
        position (int): The position of the matched term
    
    Returns:
        str: The complete article text
    """
    # First, find the article header before the position
    article_pattern = r'Art\.\s*\d+[a-z]?\.?'
    
    # Find all article headers in the text
    article_matches = list(re.finditer(article_pattern, text))
    
    if not article_matches:
        # No article headers found, return a larger context around the position
        start = max(0, position - 500)
        end = min(len(text), position + 500)
        return text[start:end]
    
    # Find the article that contains the position
    current_article_start = None
    next_article_start = None
    
    for i, match in enumerate(article_matches):
        if match.start() <= position:
            current_article_start = match.start()
        else:
            next_article_start = match.start()
            break
    
    if current_article_start is None:
        # Position is before the first article, return a context window
        end = next_article_start if next_article_start else min(len(text), position + 500)
        start = max(0, position - 500)
        return text[start:end]
    
    # Extract from current article start to next article start (or end of text)
    start = current_article_start
    end = next_article_start if next_article_start else len(text)
    
    # Try to find the article number for reference
    article_number_match = re.search(article_pattern, text[start:start+30])
    article_ref = article_number_match.group(0) if article_number_match else "Unknown Article"
    
    return f"{article_ref}: {text[start:end].strip()}"

# Function to extract article/section number from text using patterns
def extract_article_reference(text):
    # Common patterns for article references in Polish legal acts
    patterns = [
        r'Art\.\s*\d+[a-z]?(\s*[§\.]?\s*\d+[a-z]?)*',
        r'§\s*\d+[a-z]?(\s*\w{3}\.\s*\d+[a-z]?)*',
        r'\w{3}\.\s*\d+[a-z]?(\s*pkt\s*\d+[a-z]?)*',
        r'pkt\s*\d+[a-z]?(\s*lit\.\s*[a-z])*',
        r'Art\.\s*\d+[a-z]?\s*§\s*\d+[a-z]?',
        r'Art\.\s*\d+[a-z]?\s*\w{3}\.\s*\d+[a-z]?'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            match_pos = text.find(matches[0])
            start_pos = max(0, match_pos - 10)
            context = text[start_pos:match_pos + len(matches[0]) + 5]
            
            for p in patterns:
                ref_match = re.search(p, context, re.IGNORECASE)
                if ref_match:
                    return ref_match.group(0)
            
            return matches[0]
    
    return "Not identified"

def extract_article_context(text, position):
    """
    Extract the full article context containing the given position.
    Returns the complete text between article boundaries with enhanced detection.
    Also identifies previous and next articles.
    """
    # Expanded patterns to detect article headers in various formats
    article_patterns = [
        # Standard formats
        r'(?:\n|^)Art(?:icle|icuł|\.)\s+(\d+[a-zA-Z]?)(?:\.|\s)',  # Art. 123, Article 123, etc.
        r'(?:\n|^)§\s+(\d+[a-zA-Z]?)(?:\.|\s)',                    # § 123
        r'(?:\n|^)Artykuł\s+(\d+[a-zA-Z]?)(?:\.|\s)',              # Artykuł 123
        
        # Common Polish legal numbering
        r'(?:\n|^)Art\.\s*(\d+[a-z]?)(?:\.|:|$)',                  # Art. 123.
        r'(?:\n|^)Art\.\s*(\d+[a-z]?[a-z]?)(?:\.|:|$)',            # Art. 123a.
        
        # EU legal numbering
        r'(?:\n|^)(?:Article|Artykuł)\s+(\d+[a-z]?)(?:\.|\s|$)',   # Article 123
        
        # Numbered sections with indentation or formatting
        r'\n\s*(\d+)\.\s+[A-ZŁŚĆŻŹĘ]',                            # 1. [Capital letter start]
        
        # Paragraphs and ustępy
        r'(?:\n|^)ust\.\s+(\d+[a-z]?)(?:\.|\s|$)',                 # ust. 1
        r'(?:\n|^)pkt\s+(\d+[a-z]?)(?:\.|\s|$)',                   # pkt 1
    ]
    
    # Combine patterns
    combined_pattern = '|'.join(article_patterns)
    
    # Find all article headers in the text
    article_matches = list(re.finditer(combined_pattern, text))
    
    # Initialize article context structures
    article_info = {
        "previous_article": None,
        "current_article": None,
        "next_article": None,
        "current_article_text": "",
        "previous_article_reference": None,
        "current_article_reference": None,
        "next_article_reference": None
    }
    
    if not article_matches:
        # Handle cases with no article matches (same as before)
        simpler_patterns = [
            r'\n\s*\d+\.\s+',                # Numbered paragraphs: 1. 
            r'\n\s*[A-Z]\.\s+',              # Lettered sections: A.
            r'\n\s*\(\d+\)\s+',              # Parenthesized numbers: (1)
            r'\n\s*[IVX]+\.\s+'              # Roman numerals: IV.
        ]
        combined_simple = '|'.join(simpler_patterns)
        article_matches = list(re.finditer(combined_simple, text))
        
        if not article_matches:
            # Still no matches, fall back to paragraph breaks or window approach
            paragraph_breaks = list(re.finditer(r'\n\s*\n', text))
            if paragraph_breaks:
                return extract_from_paragraph_breaks(text, position, paragraph_breaks), article_info
            else:
                # Absolute fallback to window approach
                context_size = 1500
                start = max(0, position - context_size//2)
                end = min(len(text), position + context_size//2)
                return text[start:end].strip(), article_info
    
    # Find which article contains our position
    current_article_idx = None
    for i, match in enumerate(article_matches):
        if match.start() <= position:
            current_article_idx = i
        else:
            break
    
    # Extract current article text and references
    if current_article_idx is None:
        # Position is before any article - extract preamble
        start = 0
        end = article_matches[0].start() if article_matches else len(text)
        article_info["next_article_reference"] = extract_article_number(article_matches[0].group(0)) if article_matches else None
    else:
        # Current article info
        current_match = article_matches[current_article_idx]
        article_info["current_article_reference"] = extract_article_number(current_match.group(0))
        
        # Set previous article reference if it exists
        if current_article_idx > 0:
            prev_match = article_matches[current_article_idx - 1]
            article_info["previous_article_reference"] = extract_article_number(prev_match.group(0))
        
        # Set next article reference if it exists
        if current_article_idx < len(article_matches) - 1:
            next_match = article_matches[current_article_idx + 1]
            article_info["next_article_reference"] = extract_article_number(next_match.group(0))
        
        # Extract article text
        start = current_match.start()
        if current_article_idx + 1 < len(article_matches):
            end = article_matches[current_article_idx + 1].start()
        else:
            end = len(text)
    
    # Get the article text
    article_text = text[start:end].strip()
    article_info["current_article_text"] = article_text
    
    # Validate and adjust content as before
    if len(article_text) < 10:
        expanded_start = max(0, start - 100)
        expanded_end = min(len(text), end + 100)
        article_text = text[expanded_start:expanded_end].strip()
    
    # Check for overly long articles
    if len(article_text) > 10000:
        # Same logic as before
        pass
    
    # Extract previous and next article text when available (limited to first part)
    if current_article_idx is not None and current_article_idx > 0:
        prev_start = article_matches[current_article_idx - 1].start()
        prev_end = start
        article_info["previous_article"] = text[prev_start:prev_end].strip()[:1000]  # Limit to first 1000 chars
    
    if current_article_idx is not None and current_article_idx < len(article_matches) - 1:
        next_start = end
        if current_article_idx + 2 < len(article_matches):
            next_end = article_matches[current_article_idx + 2].start()
        else:
            next_end = len(text)
        article_info["next_article"] = text[next_start:next_end].strip()[:1000]  # Limit to first 1000 chars
    
    return article_text, article_info

def extract_article_number(article_header):
    """Extract just the number from an article header."""
    # Find numeric part with optional letter suffix
    match = re.search(r'(\d+[a-zA-Z]?[a-zA-Z]?)', article_header)
    if match:
        # Get the prefix (Art., §, etc.)
        prefix_match = re.search(r'(Art\.|Artykuł|Article|§|ust\.|pkt)', article_header)
        prefix = prefix_match.group(0) if prefix_match else "Art."
        return f"{prefix} {match.group(1)}"
    return None

def extract_from_paragraph_breaks(text, position, paragraph_breaks):
    """Extract content using paragraph breaks when article headers are not found."""
    current_para_idx = None
    for i, match in enumerate(paragraph_breaks):
        if match.start() <= position:
            current_para_idx = i
        else:
            break
    
    if current_para_idx is None:
        # Position is in the first paragraph
        start = 0
        end = paragraph_breaks[0].start() if paragraph_breaks else len(text)
    else:
        # Start from beginning of current paragraph
        start = paragraph_breaks[current_para_idx].end()
        
        # End at beginning of next paragraph or end of text
        if current_para_idx + 1 < len(paragraph_breaks):
            end = paragraph_breaks[current_para_idx + 1].start()
        else:
            end = len(text)
    
    return text[start:end].strip()

def is_legal_article_boundary(text, pos):
    """Check if a position in text is a valid article boundary."""
    # Get a small window around the potential boundary
    start = max(0, pos - 20)
    end = min(len(text), pos + 20)
    window = text[start:end]
    
    # Check for typical boundary markers
    boundary_markers = [
        # End of article markers
        r'\.\s*$',
        r'\.\s*\n',
        
        # Beginning of new article markers
        r'^\s*Art\.',
        r'^\s*Artykuł',
        r'^\s*§'
    ]
    
    for marker in boundary_markers:
        if re.search(marker, window):
            return True
    
    return False

# Function to collect passages for a date range
def collect_passages_for_date_range(start_date, end_date, search_phrase):
    """Collect legal passages for a specific date range and search phrase"""
    all_passages = []
    processed_nros = set()
    
    # Handle special cases for search phrases with NEAR
    is_near_query = "NEAR" in search_phrase
    if is_near_query:
        parts = search_phrase.split("NEAR")
        term1 = parts[0].strip()
        term2 = parts[1].strip()
        print(f"  NEAR query: Looking for '{term1}' near '{term2}'")
        # For NEAR queries, we'll search for the first term in the API
        api_search_term = term1
    else:
        api_search_term = search_phrase
    
    pagination_offset = 0
    more_results = True
    
    while more_results:
        try:
            # STEP 1: First, search for documents WITHOUT requesting content
            metadata_query = {
                "from": pagination_offset,
                "size": 45,
                "query": {
                    "term": api_search_term,
                    "filters": [
                        {
                            "field": "documentMainType",
                            "eq": "ACT"
                        },
                        {
                            "field": "actEffectiveDate",
                            "gte": start_date,
                            "lte": end_date
                        },
                        {
                            "field": "actValidity",
                            "eq": 134217729
                        },
                        {
                            "field": "documentType",
                            "eq": "COMMON_ACT"
                        },
                        {
                            "field": "uniformIndexDomain",
                            "in": [4686]
                        }
                    ],
                    # Only request metadata fields, NOT plainTextContent
                    "fields": ["nro", "title", "signature", "documentType", "actEffectiveDate", "enactmentDate", "effectiveDate", "publisher"]
                }
            }
            
            # Execute metadata query
            print(f"  Fetching documents {pagination_offset}-{pagination_offset+44}")
            document_response = requests.post(documents_api_url, json=metadata_query, headers=headers, timeout=60)
            
            # Add rate limiting delay after each API request
            time.sleep(API_REQUEST_DELAY)
            
            # Check for token expiration
            if document_response.status_code == 401:
                print("Token expired. Attempting to refresh...")
                if refresh_access_token():
                    # Retry the request with the new token
                    document_response = requests.post(documents_api_url, json=metadata_query, headers=headers, timeout=60)
                else:
                    print("Failed to refresh token. Aborting.")
                    break
            
            if document_response.status_code != 200:
                print(f"  Error response: {document_response.status_code}")
                print(f"  Error details: {document_response.text}")
                break
            
            data = document_response.json()
            document_list = data.get("results", [])
            if not document_list:
                print("  No documents found in this date range.")
                break
                
            total_count = data.get("allDocumentCount", 0)
            print(f"  Found {len(document_list)} documents (total available: {total_count})")
            
            # Process each document
            for doc_info in document_list:
                nro = doc_info.get("nro")
                
                # Skip if we've already processed this document
                if nro in processed_nros:
                    continue
                
                processed_nros.add(nro)
                
                title = doc_info.get("title", "")
                print(f"  Processing document: {title} (NRO: {nro})")
                
                # STEP 2: Now make a SEPARATE request JUST for the content
                # Use GET method with query parameters instead of POST
                content_endpoint = f"https://api.lex.pl/v4/documents/{nro}"

                try:
                    print(f"    Fetching content using GET request with parameters")
                    # Use query parameters to specify fields
                    params = {"fields": "plainTextContent"}
                    content_response = requests.get(
                        content_endpoint, 
                        params=params,
                        headers=headers, 
                        timeout=60
                    )
                    
                    # Add rate limiting delay after each content request
                    time.sleep(API_REQUEST_DELAY)
                    
                    # Check for token expiration
                    if content_response.status_code == 401:
                        print("Token expired while fetching document content. Attempting to refresh...")
                        if refresh_access_token():
                            # Retry the request with the new token
                            content_response = requests.get(content_endpoint, params=params, headers=headers, timeout=60)
                        else:
                            print("Failed to refresh token. Skipping document.")
                            continue
                    
                    # Add detailed debugging for response
                    print(f"    Response status: {content_response.status_code}")
                    
                    if content_response.status_code == 200:
                        # Try to get the response JSON and debug its structure
                        try:
                            doc = content_response.json()
                            print(f"    Response keys: {list(doc.keys()) if isinstance(doc, dict) else 'Not a dictionary'}")
                            
                            # Check for content under various possible fields
                            content = None
                            
                            # Try different possible field names for content
                            for field in ["plainTextContent", "content", "text", "documentContent"]:
                                if field in doc and doc[field]:
                                    content = doc[field]
                                    print(f"    Found content in field: {field} (length: {len(content)})")
                                    break
                            
                            if not content:
                                # Try looking in nested structures
                                print(f"    No direct content field found, checking response structure")
                                if "results" in doc and doc["results"] and isinstance(doc["results"], list):
                                    result = doc["results"][0]
                                    if isinstance(result, dict):
                                        for field in ["plainTextContent", "content", "text", "documentContent"]:
                                            if field in result and result[field]:
                                                content = result[field]
                                                print(f"    Found content in results[0].{field} (length: {len(content)})")
                                                break
                            
                            if not content:
                                print(f"    No content found. Full response: {str(doc)[:500]}...")
                                continue
                            
                            # Extract passages with the search phrase
                            passages, full_contexts, structured_contexts, article_contexts, debug_contexts = extract_relevant_passages(content, search_phrase)
                            
                            if passages:
                                print(f"    Found {len(passages)} matching passages")
                                print(f"    First passage (sample): {passages[0][:100]}...")
                                
                                for passage, full_context, structured_context, article_context, debug_context in zip(
                                    passages, full_contexts, structured_contexts, article_contexts, debug_contexts):
                                    # Extract article reference from the full context
                                    article_ref = extract_article_reference(full_context)
                                    
                                    passage_record = {
                                        "document_metadata": {
                                            "nro": nro,
                                            "title": title,
                                            "signature": doc_info.get("signature", ""),
                                            "documentType": doc_info.get("documentType", ""),
                                            "publicationDate": doc_info.get("actEffectiveDate", ""),
                                            "enactmentDate": doc_info.get("enactmentDate", ""),
                                            "effectiveDate": doc_info.get("effectiveDate", ""),
                                            "publisher": doc_info.get("publisher", "")
                                        },
                                        "match_data": {
                                            "search_term": search_phrase,
                                            "source": f"{title} ({doc_info.get('signature', 'No signature')})",
                                            "article": article_ref,
                                            "passage": passage.strip(),
                                            "full_context": full_context.strip(),
                                            "article_context": article_context.strip(),
                                            "structured_context": structured_context,
                                            "debug": debug_context.strip()
                                        }
                                    }
                                    
                                    all_passages.append(passage_record)
                            else:
                                print(f"    No matching passages found in document")
                        
                        except json.JSONDecodeError as je:
                            print(f"    Error parsing document JSON: {str(je)}")
                            print(f"    Response content: {content_response.text[:200]}")
                    
                except Exception as e:
                    print(f"    Error processing document content: {str(e)}")
                
                # Add a small delay between documents to avoid rate limits
                time.sleep(0.05)
            
            # Update pagination
            pagination_offset += len(document_list)
            if len(document_list) < 45 or pagination_offset >= total_count:
                more_results = False
            
        except Exception as e:
            print(f"  Error processing document list: {str(e)}")
            print(f"  Full error: {traceback.format_exc()}")
            break
    
    return all_passages, len(processed_nros)

# Add this function to handle token refresh
def refresh_access_token():
    """Request a new access token when the current one expires"""
    print("Access token expired. Requesting a new token...")
    try:
        token_response = requests.post(token_url, data=token_data, timeout=30)
        token_response.raise_for_status()
        token_json = token_response.json()
        new_access_token = token_json.get("access_token")
        
        if new_access_token:
            print("New access token obtained:", new_access_token[:30] + "...")
            # Update the global headers with the new token
            headers["Authorization"] = f"Bearer {new_access_token}"
            return True
        else:
            print("Failed to get new access token")
            return False
    except Exception as e:
        print(f"Error refreshing token: {str(e)}")
        print(traceback.format_exc())
        return False

# Main execution
try:
    # Initialize the database
    print("Initializing database...")
    db_conn = initialize_fetcher_database()
    
    # Generate monthly date ranges in reverse order (from current date back to start date)
    date_ranges = generate_monthly_ranges(
        start_year=SEARCH_START_YEAR1, 
        start_month=SEARCH_START_MONTH1, 
        end_year=SEARCH_END_YEAR1, 
        end_month=SEARCH_END_MONTH1
    )
    
    print(f"Generated {len(date_ranges)} monthly ranges from {SEARCH_END_MONTH1}/{SEARCH_END_YEAR1} back to {SEARCH_START_MONTH1}/{SEARCH_START_YEAR1}")
    print(f"Will search for {len(SEARCH_PHRASES)} different time-related phrases")
    
    all_passages = []
    total_unique_docs = 0
    
    # Process each search phrase
    for search_idx, search_phrase in enumerate(SEARCH_PHRASES):
        print(f"\n[{search_idx+1}/{len(SEARCH_PHRASES)}] SEARCHING FOR: '{search_phrase}'")
        
        # Collect passages across all date ranges for this search phrase
        for date_idx, (start_date, end_date) in enumerate(date_ranges):
            max_retries = 3
            retry_count = 0
            success = False
            
            while not success and retry_count < max_retries:
                try:
                    print(f"\nDate range {date_idx+1}/{len(date_ranges)}: {start_date} to {end_date}")
                    
                    passages, unique_docs = collect_passages_for_date_range(start_date, end_date, search_phrase)
                    
                    # If we got here, the request was successful
                    all_passages.extend(passages)
                    total_unique_docs += unique_docs
                    
                    # Store each passage in the database
                    for passage in passages:
                        store_passage_in_database(db_conn, passage)
                    
                    # Save intermediate results
                    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
                        writer.writeheader()
                        
                        for passage in all_passages:
                            # Flatten the nested structure for CSV
                            flat_record = {}
                            if "document_metadata" in passage and "match_data" in passage:
                                meta = passage["document_metadata"]
                                match = passage["match_data"]
                                
                                flat_record = {
                                    "nro": meta.get("nro", ""),
                                    "title": meta.get("title", ""),
                                    "signature": meta.get("signature", ""),
                                    "documentType": meta.get("documentType", ""),
                                    "publicationDate": meta.get("publicationDate", ""),
                                    "enactmentDate": meta.get("enactmentDate", ""),
                                    "effectiveDate": meta.get("effectiveDate", ""),
                                    "publisher": meta.get("publisher", ""),
                                    "passage": match.get("passage", ""),
                                    "article": match.get("article", ""),
                                    "full_context": match.get("full_context", ""),
                                    "article_context": match.get("article_context", ""),
                                    "search_term": match.get("search_term", ""),
                                    "source": match.get("source", ""),
                                    "debug": match.get("debug", "")
                                }
                            else:
                                flat_record = passage
                                
                            writer.writerow(flat_record)
                    
                    print(f"Progress: {len(all_passages)} passages from {total_unique_docs} documents saved")
                    
                    success = True  # Mark as successful to exit retry loop
                    
                except Exception as e:
                    retry_count += 1
                    print(f"Error processing date range (attempt {retry_count}/{max_retries}): {str(e)}")
                    if retry_count < max_retries:
                        print("Refreshing token and retrying...")
                        refresh_access_token()
                        time.sleep(2)  # Brief pause before retry
                    else:
                        print("Max retries reached. Moving to next date range.")
    
    print(f"\nSearch complete! Found {len(all_passages)} relevant passages across {total_unique_docs} unique documents.")
    print(f"Results saved to {os.path.abspath(csv_filename)}")
    
    # Print sample of first 3 passages from different documents if possible
    if all_passages:
        print("\nSample passages:")
        seen_docs = set()
        sample_count = 0
        
        for passage in all_passages:
            # Use .get() instead of direct dictionary access to handle missing keys
            passage_nro = None
            if isinstance(passage, dict):
                if "document_metadata" in passage:
                    passage_nro = passage["document_metadata"].get("nro")
                else:
                    passage_nro = passage.get("nro")
            
            if passage_nro is not None and passage_nro not in seen_docs and sample_count < 3:
                seen_docs.add(passage_nro)
                sample_count += 1
                
                # Get source and date safely
                source = ""
                pub_date = ""
                search_term = ""
                article = ""
                passage_text = ""
                
                if "document_metadata" in passage and "match_data" in passage:
                    meta = passage["document_metadata"]
                    match = passage["match_data"]
                    source = match.get("source", meta.get("title", "Unknown"))
                    pub_date = meta.get("publicationDate", "No date")
                    search_term = match.get("search_term", "")
                    article = match.get("article", "")
                    passage_text = match.get("passage", "")
                else:
                    source = passage.get("source", passage.get("title", "Unknown"))
                    pub_date = passage.get("publicationDate", "No date")
                    search_term = passage.get("search_term", "")
                    article = passage.get("article", "")
                    passage_text = passage.get("passage", "")
                
                print(f"\n{sample_count}. From: {source} ({pub_date})")
                print(f"   Search Term: {search_term}")
                print(f"   Article: {article}")
                print(f"   Text: {passage_text[:150]}..." if passage_text else "   Text: No text available")

    # Save all passages as JSON for Perplexity processing
    with open(json_output_file, 'w', encoding='utf-8') as jsonfile:
        json.dump(all_passages, jsonfile, ensure_ascii=False, indent=2)
    print(f"\nSaved all passages to {os.path.abspath(json_output_file)} for Perplexity analysis")
    print(f"Run 'perplexity_analyzer.py' to process these passages with Perplexity AI.")

finally:
    if 'db_conn' in locals():
        db_conn.close()

print("\nLex API fetcher script completed.") 