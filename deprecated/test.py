import requests
import json
import sys
import traceback
import csv
import os
import re
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import time  # Add this import for delay functionality
from openai import OpenAI  # Import for Perplexity API (which uses OpenAI-compatible endpoints)
import sqlite3  # For database storage

# ===== SEARCH CONFIGURATION =====
# Start and end dates for document search
SEARCH_START_YEAR = 2024
SEARCH_START_MONTH = 6  # January
SEARCH_END_YEAR = 2024
SEARCH_END_MONTH = 12   # December

# Remove dateutil dependency since it's causing issues
# Instead, we'll implement our own monthly range generator

print("Starting script...")

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
             "enactmentDate", "effectiveDate", "publisher", "passage", "article", "full_context", "article_context", "search_term", "source"]

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

# Add Perplexity API configuration
PERPLEXITY_API_KEY = "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"
PERPLEXITY_MODEL = "r1-1776"  # The model specified in requirements

# Initialize database
def initialize_database():
    """Create SQLite database and necessary tables if they don't exist"""
    conn = sqlite3.connect('legal_passages.db')
    cursor = conn.cursor()
    
    # Create table for legal passages with analysis - now with binding_conversion field
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS passages (
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
        classification TEXT,
        reasoning TEXT,
        confidence TEXT,
        subject TEXT,
        binding_conversion TEXT,
        json_data TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    conn.commit()
    return conn

# Updated prompt to include binding conversion analysis
def get_condensed_prompt():
    return """Analyze this Polish legal passage to identify if it contains an instructional term (termin instrukcyjny) and evaluate if converting it to a binding deadline would make sense.

Instructional terms typically:
1. Set timeframes for authorities/officials without strict consequences
2. Use expressions like "bezzwłocznie", "niezwłocznie", "bez zbędnej zwłoki"
3. Are directed at officials rather than private parties

Also evaluate if the deadline could be converted to a binding one with automatic approval ("milcząca zgoda") if the deadline is missed. For example, in Prawo budowlane, if the 60-day instructional deadline for building permits became binding, an automatic permit would be granted if officials don't respond in 60 days.

Provide your analysis in JSON format:
{
  "search_term": "term that triggered the search",
  "classification": "Instructional Term / Preclusive Term / Other Time Limit / Not a Time Limit",
  "subject": "Who the term applies to - authority or individual",
  "reasoning": "Brief explanation of your classification",
  "confidence": "High/Medium/Low",
  "binding_conversion": {
    "feasible": true/false,
    "rationale": "Why conversion would/wouldn't make sense",
    "impact": "Potential impact of making this deadline binding with automatic approval"
  }
}
"""

# Updated function to extract JSON from AI response
def extract_json_from_analysis(ai_analysis):
    """Extract JSON data from the AI analysis response"""
    try:
        # Try to find JSON structure in the response
        json_match = re.search(r'```json(.*?)```|(\{.*\})', ai_analysis, re.DOTALL)
        if json_match:
            json_str = json_match.group(1) or json_match.group(2)
            # Clean up the string and parse it
            json_str = json_str.strip()
            return json.loads(json_str)
        
        # Fallback: convert the text format to structured data
        structured_data = {
            "search_term": extract_field_from_analysis(ai_analysis, "SEARCH TERM"),
            "classification": extract_field_from_analysis(ai_analysis, "CLASSIFICATION"),
            "subject": extract_field_from_analysis(ai_analysis, "SUBJECT"),
            "reasoning": extract_field_from_analysis(ai_analysis, "REASONING"),
            "confidence": extract_field_from_analysis(ai_analysis, "CONFIDENCE"),
            "binding_conversion": {
                "feasible": "Yes" in extract_field_from_analysis(ai_analysis, "BINDING CONVERSION"),
                "rationale": extract_field_from_analysis(ai_analysis, "RATIONALE"),
                "impact": extract_field_from_analysis(ai_analysis, "IMPACT")
            }
        }
        return structured_data
        
    except json.JSONDecodeError:
        # If JSON parsing fails, return structured data from regex extraction
        return {
            "search_term": extract_field_from_analysis(ai_analysis, "SEARCH TERM"),
            "classification": extract_field_from_analysis(ai_analysis, "CLASSIFICATION"),
            "subject": extract_field_from_analysis(ai_analysis, "SUBJECT"),
            "reasoning": extract_field_from_analysis(ai_analysis, "REASONING"),
            "confidence": extract_field_from_analysis(ai_analysis, "CONFIDENCE"),
            "binding_conversion": {
                "feasible": False,
                "rationale": "Unable to determine from AI response",
                "impact": "Not analyzed"
            }
        }

# Updated function to analyze a passage with Perplexity AI
def analyze_passage_with_perplexity(passage_data):
    """Send passage to Perplexity AI for analysis and return the enriched data"""
    
    # Extract relevant information from the passage record
    if "document_metadata" in passage_data and "match_data" in passage_data:
        meta = passage_data["document_metadata"]
        match = passage_data["match_data"]
        
        passage_text = match.get("passage", "")
        search_term = match.get("search_term", "")
        article = match.get("article", "")
        context = match.get("full_context", "")
        source = match.get("source", "")
    else:
        passage_text = passage_data.get("passage", "")
        search_term = passage_data.get("search_term", "")
        article = passage_data.get("article", "")
        context = passage_data.get("full_context", "")
        source = passage_data.get("source", "")
    
    # Create the content for analysis
    content = f"""
Passage: {passage_text}
Search Term: {search_term}
Article: {article}
Source: {source}
Additional Context: {context[:500]}...
"""
    
    # Initialize Perplexity client (using OpenAI-compatible interface)
    try:
        client = OpenAI(api_key=PERPLEXITY_API_KEY, base_url="https://api.perplexity.ai")
        
        messages = [
            {
                "role": "system",
                "content": get_condensed_prompt()
            },
            {
                "role": "user",
                "content": content
            }
        ]
        
        print(f"Sending passage to Perplexity AI for analysis: '{passage_text[:100]}...'")
        
        # Make the API call
        response = client.chat.completions.create(
            model=PERPLEXITY_MODEL,
            messages=messages,
        )
        
        ai_analysis = response.choices[0].message.content
        print(f"Received analysis from Perplexity AI: '{ai_analysis[:100]}...'")
        
        # Try to extract structured JSON data
        analysis_data = extract_json_from_analysis(ai_analysis)
        
        # Extract key fields, with fallbacks to the old extraction method
        classification = analysis_data.get("classification", extract_field_from_analysis(ai_analysis, "CLASSIFICATION"))
        reasoning = analysis_data.get("reasoning", extract_field_from_analysis(ai_analysis, "REASONING"))
        confidence = analysis_data.get("confidence", extract_field_from_analysis(ai_analysis, "CONFIDENCE"))
        subject = analysis_data.get("subject", extract_field_from_analysis(ai_analysis, "SUBJECT"))
        
        # Get binding conversion info
        binding_conversion = analysis_data.get("binding_conversion", {})
        binding_feasible = binding_conversion.get("feasible", False)
        binding_rationale = binding_conversion.get("rationale", "Not analyzed")
        binding_impact = binding_conversion.get("impact", "Not analyzed")
        
        # Format the binding conversion info as a string for the database
        binding_conversion_text = f"Feasible: {'Yes' if binding_feasible else 'No'}\nRationale: {binding_rationale}\nImpact: {binding_impact}"
        
        # Add the analysis results to the passage data
        if "document_metadata" in passage_data and "match_data" in passage_data:
            if "legal_analysis" not in passage_data["match_data"]:
                passage_data["match_data"]["legal_analysis"] = {}
                
            passage_data["match_data"]["legal_analysis"]["classification"] = classification
            passage_data["match_data"]["legal_analysis"]["reasoning"] = reasoning
            passage_data["match_data"]["legal_analysis"]["confidence"] = confidence
            passage_data["match_data"]["legal_analysis"]["subject"] = subject
            passage_data["match_data"]["legal_analysis"]["binding_conversion"] = binding_conversion
            passage_data["match_data"]["legal_analysis"]["ai_response"] = ai_analysis
        else:
            passage_data["classification"] = classification
            passage_data["reasoning"] = reasoning
            passage_data["confidence"] = confidence
            passage_data["subject"] = subject
            passage_data["binding_conversion"] = binding_conversion
            passage_data["ai_response"] = ai_analysis
        
        return passage_data
        
    except Exception as e:
        print(f"Error analyzing passage with Perplexity AI: {str(e)}")
        print(traceback.format_exc())
        
        # Return original data if analysis fails
        return passage_data

# Helper function to extract fields from AI analysis
def extract_field_from_analysis(analysis, field_name):
    """Extract a specific field from the AI analysis response"""
    field_pattern = f"{field_name}: (.*?)(?:\n|$)"
    match = re.search(field_pattern, analysis, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return "Not specified"

# Updated function to store a passage in the database
def store_passage_in_database(conn, passage_data):
    """Store the passage data in the SQLite database"""
    cursor = conn.cursor()
    
    # Extract the data in a flat structure
    if "document_metadata" in passage_data and "match_data" in passage_data:
        meta = passage_data["document_metadata"]
        match = passage_data["match_data"]
        analysis = match.get("legal_analysis", {})
        
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
            # If it's a dictionary, convert to a string representation
            publisher = json.dumps(publisher_data, ensure_ascii=False)
        else:
            publisher = str(publisher_data)
        
        passage = match.get("passage", "")
        article = match.get("article", "")
        full_context = match.get("full_context", "")
        article_context = match.get("article_context", "")
        search_term = match.get("search_term", "")
        source = match.get("source", "")
        
        classification = analysis.get("classification", "")
        reasoning = analysis.get("reasoning", "")
        confidence = analysis.get("confidence", "")
        subject = analysis.get("subject", "")
        
        # Extract binding conversion data
        binding_conversion = analysis.get("binding_conversion", {})
        binding_feasible = binding_conversion.get("feasible", False)
        binding_rationale = binding_conversion.get("rationale", "Not analyzed")
        binding_impact = binding_conversion.get("impact", "Not analyzed")
        binding_conversion_text = f"Feasible: {'Yes' if binding_feasible else 'No'}\nRationale: {binding_rationale}\nImpact: {binding_impact}"
    else:
        nro = passage_data.get("nro", "")
        title = passage_data.get("title", "")
        signature = passage_data.get("signature", "")
        document_type = passage_data.get("documentType", "")
        publication_date = passage_data.get("publicationDate", "")
        enactment_date = passage_data.get("enactmentDate", "")
        effective_date = passage_data.get("effectiveDate", "")
        
        # Ensure publisher is a string here too
        publisher_data = passage_data.get("publisher", "")
        if isinstance(publisher_data, dict):
            publisher = json.dumps(publisher_data, ensure_ascii=False)
        else:
            publisher = str(publisher_data)
        
        passage = passage_data.get("passage", "")
        article = passage_data.get("article", "")
        full_context = passage_data.get("full_context", "")
        article_context = passage_data.get("article_context", "")
        search_term = passage_data.get("search_term", "")
        source = passage_data.get("source", "")
        
        classification = passage_data.get("classification", "")
        reasoning = passage_data.get("reasoning", "")
        confidence = passage_data.get("confidence", "")
        subject = passage_data.get("subject", "")
        
        # Extract binding conversion data from the flat structure
        binding_conversion = passage_data.get("binding_conversion", {})
        if isinstance(binding_conversion, dict):
            binding_feasible = binding_conversion.get("feasible", False)
            binding_rationale = binding_conversion.get("rationale", "Not analyzed")
            binding_impact = binding_conversion.get("impact", "Not analyzed")
        else:
            binding_feasible = False
            binding_rationale = "Not analyzed"
            binding_impact = "Not analyzed"
        binding_conversion_text = f"Feasible: {'Yes' if binding_feasible else 'No'}\nRationale: {binding_rationale}\nImpact: {binding_impact}"
    
    # Store the complete JSON data
    json_data = json.dumps(passage_data, ensure_ascii=False)
    
    try:
        cursor.execute('''
        INSERT INTO passages (
            nro, title, signature, document_type, publication_date, enactment_date, 
            effective_date, publisher, passage, article, full_context, article_context, 
            search_term, source, classification, reasoning, confidence, subject, 
            binding_conversion, json_data
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            str(nro), str(title), str(signature), str(document_type), 
            str(publication_date), str(enactment_date), str(effective_date), 
            publisher, str(passage), str(article), str(full_context), 
            str(article_context), str(search_term), str(source), 
            str(classification), str(reasoning), str(confidence), 
            str(subject), str(binding_conversion_text), json_data
        ))
        
        conn.commit()
        return True
    except Exception as e:
        print(f"Error storing passage in database: {str(e)}")
        print(traceback.format_exc())
        conn.rollback()
        return False

# Add processing function for all passages
def process_passages_with_perplexity_and_store(passages):
    """Process all passages with Perplexity AI and store in database"""
    try:
        # Check for previously processed passages
        checkpoint_file = "perplexity_checkpoint.json"
        processed_ids = set()
        success_count = 0  # Initialize the success counter

        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'r') as f:
                    processed_ids = set(json.load(f))
                print(f"Found checkpoint with {len(processed_ids)} already processed passages")
            except Exception as e:
                print(f"Error loading checkpoint: {e}")

        # Initialize the database
        db_conn = initialize_database()
        
        print(f"Processing {len(passages)} passages with Perplexity AI...")
        
        # Process in smaller batches to avoid overwhelming the API
        batch_size = 10
        for i in range(0, len(passages), batch_size):
            batch = passages[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(passages)-1)//batch_size + 1} (passages {i+1}-{min(i+batch_size, len(passages))})...")
            
            # Process each passage in the batch
            for j, passage in enumerate(batch):
                try:
                    print(f"  Processing passage {i+j+1}/{len(passages)}...")
                    
                    # Analyze with Perplexity AI
                    enriched_passage = analyze_passage_with_perplexity(passage)
                    
                    # Store in database
                    passage_id = None
                    if "document_metadata" in passage and "match_data" in passage:
                        passage_id = f"{passage['document_metadata'].get('nro', '')}_{passage['match_data'].get('passage', '')[:50]}"
                    else:
                        passage_id = f"{passage.get('nro', '')}_{passage.get('passage', '')[:50]}"

                    # Skip already processed passages
                    if passage_id in processed_ids:
                        print(f"  Skipping already processed passage {i+j+1}")
                        continue

                    if store_passage_in_database(db_conn, enriched_passage):
                        success_count += 1
                        print(f"  Successfully processed and stored passage {i+j+1}")
                    else:
                        print(f"  Failed to store passage {i+j+1} in database")
                    
                    # Add a delay to avoid rate limits
                    time.sleep(1)
                    
                    # After successful processing:
                    processed_ids.add(passage_id)
                    # Save checkpoint
                    with open(checkpoint_file, 'w') as f:
                        json.dump(list(processed_ids), f)
                    
                except Exception as e:
                    print(f"  Error processing passage {i+j+1}: {str(e)}")
                    traceback.print_exc()
            
            # Longer pause between batches
            print(f"Batch {i//batch_size + 1} complete. Pausing before next batch...")
            time.sleep(5)
        
    except Exception as e:
        print(f"Error in Perplexity processing: {str(e)}")
        traceback.print_exc()
    finally:
        if 'db_conn' in locals():
            db_conn.close()

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
    passages = []
    full_contexts = []
    structured_contexts = []
    article_contexts = []  # New list for complete article contexts
    
    if not content:
        return passages, full_contexts, structured_contexts, article_contexts
    
    # Process HTML content if present
    if '<' in content and '>' in content:
        try:
            soup = BeautifulSoup(content, 'html.parser')
            text = soup.get_text(" ", strip=True)
        except Exception as e:
            print(f"Error parsing HTML: {e}")
            text = content
    else:
        text = content
    
    # For regular search terms (not NEAR or regex)
    if "NEAR" not in search_phrase and "\\d+" not in search_phrase:
        # Find all occurrences of the search phrase
        term = search_phrase.lower()
        text_lower = text.lower()
        
        # Find all positions of the search term
        positions = []
        pos = text_lower.find(term)
        while pos != -1:
            positions.append(pos)
            pos = text_lower.find(term, pos + 1)
        
        for pos in positions:
            # Get more precise context: the sentence containing the term plus one before and after
            # Find sentence boundaries
            sentence_start = max(0, text.rfind('. ', 0, pos) + 2)
            if sentence_start == 1:  # No period found before
                sentence_start = max(0, text.rfind('\n', 0, pos) + 1)
                if sentence_start == 0:  # No newline found before
                    sentence_start = 0
            
            sentence_end = text.find('. ', pos + len(term))
            if sentence_end == -1:  # No period found after
                sentence_end = text.find('\n', pos + len(term))
                if sentence_end == -1:  # No newline found after
                    sentence_end = len(text)
            else:
                sentence_end += 2  # Include the period and space
            
            # Get one more sentence before and after if available
            prev_sentence_start = max(0, text.rfind('. ', 0, sentence_start - 2) + 2)
            next_sentence_end = text.find('. ', sentence_end)
            if next_sentence_end != -1:
                next_sentence_end += 2
            else:
                next_sentence_end = len(text)
            
            # Extract the focused passage (just the sentence with the term)
            passage = text[sentence_start:sentence_end].strip()
            
            # Extract the wider context (sentence before and after)
            full_context = text[prev_sentence_start:next_sentence_end].strip()
            
            # Create structured context for LLM analysis
            structured_context = {
                "exact_match": text[pos:pos+len(term)],
                "sentence": passage,
                "preceding_sentence": text[prev_sentence_start:sentence_start].strip(),
                "following_sentence": text[sentence_end:next_sentence_end].strip(),
                "article_context": extract_article_reference(full_context),
                "position_in_document": pos,
                "match_length": len(term)
            }
            
            passages.append(passage)
            full_contexts.append(full_context)
            structured_contexts.append(structured_context)
            
            # Now extract the entire article containing this position
            article_text = extract_full_article_containing_position(text, pos)
            article_contexts.append(article_text)
    
    # Similar improvements for NEAR and regex searches...
    
    return passages, full_contexts, structured_contexts, article_contexts

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

# Function to collect passages for a date range
def collect_passages_for_date_range(start_date, end_date, search_phrase):
    print(f"Searching for documents between {start_date} and {end_date} with '{search_phrase}'")
    
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
            response = requests.post(documents_api_url, json=metadata_query, headers=headers, timeout=60)
            
            if response.status_code != 200:
                print(f"  Error response: {response.status_code}")
                print(f"  Error details: {response.text}")
                break
            
            data = response.json()
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
                            passages, full_contexts, structured_contexts, article_contexts = extract_relevant_passages(content, search_phrase)
                            
                            if passages:
                                print(f"    Found {len(passages)} matching passages")
                                print(f"    First passage (sample): {passages[0][:100]}...")
                                
                                for passage, full_context, structured_context, article_context in zip(passages, full_contexts, structured_contexts, article_contexts):
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
                                            "legal_analysis": {
                                                "contains_deadline": None,  # To be filled by LLM
                                                "deadline_type": None,      # To be filled by LLM 
                                                "termin_instrukcyjny": None, # To be filled by LLM
                                                "obligation_actor": None,    # To be filled by LLM (who has the obligation)
                                                "deadline_period": None,     # To be filled by LLM (e.g., "14 days")
                                                "deadline_trigger": None     # To be filled by LLM (what starts the countdown)
                                            }
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
                time.sleep(0.5)
            
            # Update pagination
            pagination_offset += len(document_list)
            if len(document_list) < 45 or pagination_offset >= total_count:
                more_results = False
            
        except Exception as e:
            print(f"  Error processing document list: {str(e)}")
            print(f"  Full error: {traceback.format_exc()}")
            break
    
    return all_passages, len(processed_nros)

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

# Main execution
try:
    # Generate monthly date ranges in reverse order (from current date back to start date)
    date_ranges = generate_monthly_ranges(
        start_year=SEARCH_START_YEAR, 
        start_month=SEARCH_START_MONTH, 
        end_year=SEARCH_END_YEAR, 
        end_month=SEARCH_END_MONTH
    )
    
    print(f"Generated {len(date_ranges)} monthly ranges from {SEARCH_END_MONTH}/{SEARCH_END_YEAR} back to {SEARCH_START_MONTH}/{SEARCH_START_YEAR}")
    print(f"Will search for {len(SEARCH_PHRASES)} different time-related phrases")
    
    all_passages = []
    total_unique_docs = 0
    
    # Process each search phrase
    for search_idx, search_phrase in enumerate(SEARCH_PHRASES):
        print(f"\n[{search_idx+1}/{len(SEARCH_PHRASES)}] SEARCHING FOR: '{search_phrase}'")
        
        # Collect passages across all date ranges for this search phrase
        for date_idx, (start_date, end_date) in enumerate(date_ranges):
            print(f"\nDate range {date_idx+1}/{len(date_ranges)}: {start_date} to {end_date}")
            
            passages, unique_docs = collect_passages_for_date_range(start_date, end_date, search_phrase)
            all_passages.extend(passages)
            total_unique_docs += unique_docs
            
            # Save intermediate results after each month with proper flattening
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
                            "source": match.get("source", "")
                        }
                    else:
                        flat_record = passage
                        
                    writer.writerow(flat_record)
            
            print(f"Progress: {len(all_passages)} passages from {total_unique_docs} documents saved")
    
    print(f"\nSearch complete! Found {len(all_passages)} relevant passages across {total_unique_docs} unique documents.")
    print(f"Results saved to {os.path.abspath(csv_filename)}")
    
    # Fix the sample passages printing section to handle missing keys
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

    # Now the Perplexity analysis will run after fixing the samples
    if all_passages:
        print("\nStarting analysis with Perplexity AI and database storage...")
        process_passages_with_perplexity_and_store(all_passages)
        print("Analysis and storage complete!")

except Exception as e:
    print(f"Error in search: {str(e)}")
    print(f"Full error: {traceback.format_exc()}")
    
    # Save any results collected so far
    if all_passages:
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
                        "source": match.get("source", "")
                    }
                else:
                    flat_record = passage
                    
                writer.writerow(flat_record)
        
        print(f"Partial results saved to {os.path.abspath(csv_filename)}")

print("\nScript completed.")