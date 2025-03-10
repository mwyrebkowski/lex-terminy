import json
import os
import sys
import traceback
import time
from openai import OpenAI
import sqlite3
import re
from concurrent.futures import ThreadPoolExecutor
# import asyncio
# import aiohttp
# from functools import partial
import argparse
import threading

# Perplexity API configuration
PERPLEXITY_API_KEY = "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"
PERPLEXITY_MODEL = "r1-1776"  # The model specified in requirements

# Input/output files
JSON_INPUT_FILE = "passages_for_perplexity.json"
RESULTS_DB_FILE = "legal_passages.db"

# Add these constants near the top of your file
MAX_WORKERS = 20  # Number of parallel requests
REQUESTS_PER_MINUTE = 500  # Your rate limit
MIN_REQUEST_INTERVAL = 60 / REQUESTS_PER_MINUTE  # Minimum time between requests

# Add these constants
MAX_CONCURRENT_REQUESTS = 20
RATE_LIMIT = 500  # requests per minute
RATE_LIMIT_PERIOD = 60  # seconds

print("Starting Perplexity analyzer script...")

# Create a thread-local storage
thread_local = threading.local()

# Initialize database
def initialize_database():
    """Create SQLite database and necessary tables if they don't exist"""
    conn = sqlite3.connect(RESULTS_DB_FILE)
    cursor = conn.cursor()
    
    # Create table for legal passages with analysis
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

# Get the prompt for Perplexity AI
def get_condensed_prompt():
    return """Analyze this Polish legal passage to identify if it contains an instructional term (termin instrukcyjny) and evaluate if converting it to a binding deadline would make sense.

Instructional terms typically:
1. Set timeframes for authorities/officials without strict consequences
2. Use expressions like "bezzwłocznie", "niezwłocznie", "bez zbędnej zwłoki"
3. Are directed at officials rather than private parties

Also evaluate if the deadline could be converted to a binding one with automatic approval ("milcząca zgoda") if the deadline is missed. For example, in Prawo budowlane, if the 60-day instructional deadline for building permits became binding, an automatic permit would be granted if officials don't respond in 60 days.

Additionally, identify the specific article reference (e.g., "Art. 123", "§ 5", "ust. 2 pkt 3") that contains this passage, based on the context provided.

Provide your analysis in JSON format. Output the analysis in Polish language:
{
  "search_term": "term that triggered the search",
  "classification": "Instructional Term / Preclusive Term / Other Time Limit / Not a Time Limit",
  "subject": "Who the term applies to - authority or individual",
  "reasoning": "Brief explanation of your classification",
  "confidence": "High/Medium/Low",
  "article_reference": "The specific article number (Art. X) that contains this provision",
  "binding_conversion": {
    "feasible": true/false,
    "rationale": "Why conversion would/wouldn't make sense",
    "impact": "Potential impact of making this deadline binding with automatic approval"
  }
}
"""

# Improved function to extract JSON from AI response with </think> handling
def extract_json_from_analysis(ai_analysis):
    """Extract JSON data from the AI analysis response, focusing on content after </think>"""
    try:
        # Check if we have a </think> tag and extract content after it
        think_match = re.search(r'</think>(.*)', ai_analysis, re.DOTALL)
        if think_match:
            # Use only the content after </think>
            content = think_match.group(1).strip()
        else:
            # No </think> tag, use the full response
            content = ai_analysis
        
        # Try to find JSON structure in the content
        json_match = re.search(r'```json(.*?)```|(\{.*\})', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1) or json_match.group(2)
            # Clean up the string and parse it
            json_str = json_str.strip()
            return json.loads(json_str)
        
        # Fallback: convert the text format to structured data
        structured_data = {
            "search_term": extract_field_from_analysis(content, "SEARCH TERM"),
            "classification": extract_field_from_analysis(content, "CLASSIFICATION"),
            "subject": extract_field_from_analysis(content, "SUBJECT"),
            "reasoning": extract_field_from_analysis(content, "REASONING"),
            "confidence": extract_field_from_analysis(content, "CONFIDENCE"),
            "binding_conversion": {
                "feasible": "Yes" in extract_field_from_analysis(content, "BINDING CONVERSION"),
                "rationale": extract_field_from_analysis(content, "RATIONALE"),
                "impact": extract_field_from_analysis(content, "IMPACT")
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

# Helper function to extract fields from AI analysis
def extract_field_from_analysis(analysis, field_name):
    """Extract a specific field from the AI analysis response"""
    field_pattern = f"{field_name}: (.*?)(?:\n|$)"
    match = re.search(field_pattern, analysis, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return "Not specified"

# Function to analyze a passage with Perplexity AI
def analyze_passage_with_perplexity(passage):
    """Send a passage to Perplexity AI for analysis and return the result"""
    try:
        # Extract details from the passage
        if "document_metadata" in passage and "match_data" in passage:
            # Extract from structured data
            passage_text = passage["match_data"].get("passage", "")
            search_term = passage["match_data"].get("search_term", "")
            article = passage["match_data"].get("article", "")
            context = passage["match_data"].get("full_context", "")
            article_context = passage["match_data"].get("article_context", "")
            source = passage["match_data"].get("source", "Unknown")
            
            # Get document metadata
            meta = passage["document_metadata"]
            nro = meta.get("nro", "")
            title = meta.get("title", "")
            signature = meta.get("signature", "")
            document_type = meta.get("documentType", "")
            publication_date = meta.get("publicationDate", "")
            enactment_date = meta.get("enactmentDate", "")
            effective_date = meta.get("effectiveDate", "")
            publisher = meta.get("publisher", "")
        else:
            # Extract from flat structure
            passage_text = passage.get("passage", "")
            search_term = passage.get("search_term", "")
            article = passage.get("article", "")
            context = passage.get("full_context", "")
            article_context = passage.get("article_context", "")
            source = passage.get("source", "Unknown")
            
            nro = passage.get("nro", "")
            title = passage.get("title", "")
            signature = passage.get("signature", "")
            document_type = passage.get("documentType", "")
            publication_date = passage.get("publicationDate", "")
            enactment_date = passage.get("enactmentDate", "")
            effective_date = passage.get("effectiveDate", "")
            publisher = passage.get("publisher", "")
        
        # Create prompt with full context for article identification
        prompt = get_condensed_prompt()
        full_prompt = f"{prompt}\n\nPassage to analyze (from {source}):\n{passage_text}\n\nFull context:\n{article_context}\n\nCurrent article reference (if known): {article}"
        
        # Create OpenAI client with the correct base_url for Perplexity
        client = OpenAI(
            api_key=PERPLEXITY_API_KEY,
            base_url="https://api.perplexity.ai"
        )
        
        # Call the API
        response = client.chat.completions.create(
            model=PERPLEXITY_MODEL,
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.2,
            max_tokens=3000
        )
        
        # Get the raw response content
        raw_response = response.choices[0].message.content if response.choices else ""
        
        # Extract JSON after the thinking section
        try:
            # First, check if there's a thinking section to remove
            if "<think>" in raw_response and "</think>" in raw_response:
                # Extract content after the </think> tag
                json_content = raw_response.split("</think>", 1)[1].strip()
            else:
                # Use the full response if no thinking tags
                json_content = raw_response
            
            # Look for JSON content between triple backticks
            json_match = re.search(r'```json\s*(.*?)\s*```', json_content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
            else:
                # Try to find any JSON-like structure
                json_match = re.search(r'({[\s\S]*})', json_content)
                if json_match:
                    json_str = json_match.group(1).strip()
                else:
                    # If all else fails, try using the raw content
                    json_str = json_content
            
            # Clean up any unwanted prefixes
            if json_str.startswith("```json"):
                json_str = json_str[7:].strip()
            
            # Fix common truncation issues
            # If JSON ends with a property name and colon but no value, add null or empty string
            json_str = re.sub(r'"([^"]+)"\s*:\s*$', r'"\1": ""', json_str)
            
            # If the JSON ends with a comma, remove it
            json_str = re.sub(r',\s*$', '', json_str)
            
            # If the JSON is missing a closing brace(s), add them
            open_braces = json_str.count('{')
            close_braces = json_str.count('}')
            if open_braces > close_braces:
                json_str += '}' * (open_braces - close_braces)
            
            # Try to parse the JSON
            analysis_result = json.loads(json_str)
            
            # Get the AI-determined article reference if available
            ai_article_reference = analysis_result.get("article_reference", "")
            
            # Use AI-determined article reference if it exists and looks valid
            # Otherwise fall back to the original article reference
            if ai_article_reference and ai_article_reference not in ["Not identified", "Unknown", "None"]:
                article_to_use = ai_article_reference
            else:
                article_to_use = article
            
            # Add the passage data to the analysis result
            analysis_result["nro"] = nro
            analysis_result["title"] = title
            analysis_result["signature"] = signature
            analysis_result["document_type"] = document_type
            analysis_result["publication_date"] = publication_date
            analysis_result["enactment_date"] = enactment_date
            analysis_result["effective_date"] = effective_date
            analysis_result["publisher"] = publisher
            analysis_result["passage"] = passage_text
            analysis_result["article"] = article_to_use  # Use AI-determined article reference when available
            analysis_result["original_article"] = article  # Keep the original reference for reference
            analysis_result["full_context"] = context
            analysis_result["article_context"] = article_context
            analysis_result["source"] = source
            
            return {"analysis_result": analysis_result, "raw_response": raw_response}
            
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {str(e)}")
            print(f"Attempted to parse: {json_str}")
            print(f"Raw response: {raw_response}")
            
            # In test mode, show a more detailed error
            if hasattr(thread_local, 'test_mode') and thread_local.test_mode:
                print("\nDetailed JSON parsing debug:")
                print(f"Raw response length: {len(raw_response)} chars")
                print(f"First 100 chars: {raw_response[:100]}")
                print(f"Last 100 chars: {raw_response[-100:] if len(raw_response) > 100 else raw_response}")
            
            return {"analysis_result": None, "raw_response": raw_response}
            
    except Exception as e:
        print(f"Error analyzing passage: {str(e)}")
        print(traceback.format_exc())
        return {"analysis_result": None, "raw_response": str(e)}

# Function to store a passage in the database
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

# Modified process_passages_with_perplexity_and_store function
def process_passages_with_perplexity_and_store(passages):
    """Process all passages with Perplexity AI and store in database"""
    global processed_passages
    
    print(f"Will process {len(passages)} passages with parallel execution.")
    
    # Initialize database
    initialize_database()
    
    # Load already processed passages
    load_checkpoint()
    
    # Filter passages that haven't been processed yet
    unprocessed_passages = []
    for passage in passages:
        passage_id = passage.get("id", "")
        if passage_id not in processed_passages:
            unprocessed_passages.append(passage)
    
    print(f"Found {len(unprocessed_passages)} passages that need processing.")
    
    # Process in batches to manage memory
    batch_size = 100
    total_processed = 0
    
    # Process passages in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for i in range(0, len(unprocessed_passages), batch_size):
            batch = unprocessed_passages[i:i+batch_size]
            print(f"\nProcessing batch {i//batch_size + 1} of {(len(unprocessed_passages) + batch_size - 1) // batch_size} ({len(batch)} passages)...")
            
            # Submit all tasks to the executor
            futures = [executor.submit(process_single_passage, passage) for passage in batch]
            
            # Wait for all futures to complete
            for future in futures:
                result = future.result()
                if result:
                    total_processed += 1
            
            print(f"Batch complete. Processed {total_processed} passages so far.")
            
            # Optional: Add a small delay between batches
            time.sleep(1)
    
    print(f"\nParallel processing complete!")
    return total_processed

# Create a rate limiter class
class RateLimiter:
    def __init__(self, rate_limit, period):
        self.rate_limit = rate_limit
        self.period = period
        self.tokens = rate_limit
        self.last_refill = time.time()
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        async with self.lock:
            now = time.time()
            elapsed = now - self.last_refill
            
            # Refill tokens based on elapsed time
            if elapsed > 0:
                new_tokens = int(elapsed * (self.rate_limit / self.period))
                if new_tokens > 0:
                    self.tokens = min(self.rate_limit, self.tokens + new_tokens)
                    self.last_refill = now
            
            # Wait if no tokens available
            if self.tokens <= 0:
                wait_time = (1.0 / (self.rate_limit / self.period))
                await asyncio.sleep(wait_time)
                return await self.acquire()
            
            self.tokens -= 1
            return True

# Main processing function using asyncio
async def process_passages_async(passages, db_conn):
    # Initialize rate limiter
    limiter = RateLimiter(RATE_LIMIT, RATE_LIMIT_PERIOD)
    
    # Create a semaphore to limit concurrent connections
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    # Create client session
    async with aiohttp.ClientSession() as session:
        # Create tasks for all passages
        tasks = []
        for passage in passages:
            if passage.get("id") not in processed_passages:
                task = asyncio.create_task(
                    process_passage_async(passage, session, limiter, semaphore, db_conn)
                )
                tasks.append(task)
        
        # Process tasks in batches to avoid memory issues
        batch_size = 100
        total_processed = 0
        
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i+batch_size]
            results = await asyncio.gather(*batch, return_exceptions=True)
            
            # Count successful results
            for result in results:
                if result is True:
                    total_processed += 1
            
            print(f"Completed batch {i//batch_size + 1}. Processed {total_processed} passages so far.")
    
    return total_processed

# Update the test_process_5_records function to set test_mode flag
def test_process_5_records():
    """Process just 5 records and display detailed output in terminal for testing"""
    print("RUNNING IN TEST MODE: Processing 5 records only")
    
    # Set test mode flag in thread_local
    thread_local.test_mode = True
    
    # Initialize database
    db_conn = initialize_database()
    print("Database initialized for test.")
    
    # Load passages
    try:
        print(f"Loading passages from {JSON_INPUT_FILE}...")
        with open(JSON_INPUT_FILE, 'r', encoding='utf-8') as f:
            all_passages = json.load(f)
        
        print(f"Loaded {len(all_passages)} passages. Will process first 5 for testing.")
        
        # Take only the first 5 records
        test_passages = all_passages[:5]
        
        # Process each record sequentially with detailed output
        for i, passage in enumerate(test_passages):
            print(f"\n{'='*80}\nTEST RECORD {i+1}/5\n{'='*80}")
            
            # Extract metadata
            if "document_metadata" in passage and "match_data" in passage:
                nro = passage["document_metadata"].get("nro", "")
                title = passage["document_metadata"].get("title", "")
                passage_text = passage["match_data"].get("passage", "")
                search_term = passage["match_data"].get("search_term", "")
                source = passage["match_data"].get("source", "Unknown")
            else:
                nro = passage.get("nro", "")
                title = passage.get("title", "")
                passage_text = passage.get("passage", "")
                search_term = passage.get("search_term", "")
                source = passage.get("source", "Unknown")
            
            print(f"Document: {title} (ID: {nro})")
            print(f"Search Term: '{search_term}'")
            print(f"Source: {source}")
            print(f"\nPassage Text:\n{passage_text}\n")
            
            # Call Perplexity API
            print("Sending to Perplexity API...")
            start_time = time.time()
            result = analyze_passage_with_perplexity(passage)
            elapsed = time.time() - start_time
            
            # Get the raw response and analysis result
            raw_response = result.get("raw_response", "No response")
            analysis_result = result.get("analysis_result")
            
            # Print the raw API response
            print(f"\nRAW API RESPONSE (received in {elapsed:.1f}s):")
            print("="*40)
            print(raw_response)
            print("="*40)
            
            # Extract and print the analysis if available
            if analysis_result:
                classification = analysis_result.get("classification", "")
                subject = analysis_result.get("subject", "")
                reasoning = analysis_result.get("reasoning", "")
                confidence = analysis_result.get("confidence", "")
                
                binding_conversion = analysis_result.get("binding_conversion", {})
                feasible = binding_conversion.get("feasible", False)
                rationale = binding_conversion.get("rationale", "")
                impact = binding_conversion.get("impact", "")
                
                print(f"\nParsed Analysis Results:")
                print(f"Classification: {classification}")
                print(f"Subject: {subject}")
                print(f"Confidence: {confidence}")
                print(f"Reasoning: {reasoning}")
                print(f"\nBinding Conversion:")
                print(f"Feasible: {feasible}")
                print(f"Rationale: {rationale}")
                print(f"Impact: {impact}")
                
                # Store in database
                stored = store_passage_in_database(db_conn, analysis_result)
                print(f"\nStored in database: {'Success' if stored else 'Failed'}")
            else:
                print("\nFailed to parse JSON from Perplexity API response")
                print("Check the raw response above for format issues")
            
            # Add a short delay between test records
            if i < 4:  # Don't delay after the last one
                print("\nWaiting 2 seconds before next test record...")
                time.sleep(2)
        
        print(f"\n{'='*80}")
        print(f"TEST COMPLETE: Processed 5 records")
        print(f"Results stored in {RESULTS_DB_FILE}")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"Error in test mode: {str(e)}")
        traceback.print_exc()
    finally:
        if 'db_conn' in locals():
            db_conn.close()

# Add this function to process a batch of records in parallel
def process_batch_parallel(batch_size, max_workers, skip_records=0):
    """Process a specific batch of records using parallel processing"""
    print(f"RUNNING IN BATCH MODE: Processing {batch_size} records with {max_workers} parallel workers")
    print(f"Skipping first {skip_records} records")
    
    # Initialize database schema (but don't keep this connection for the workers)
    db_conn = initialize_database()
    db_conn.close()  # Close immediately after schema setup
    print("Database initialized for batch processing.")
    
    # Create a ThreadPoolExecutor with the specified number of workers
    executor = ThreadPoolExecutor(max_workers=max_workers)
    
    try:
        # Load passages from JSON
        print(f"Loading passages from {JSON_INPUT_FILE}...")
        with open(JSON_INPUT_FILE, 'r', encoding='utf-8') as f:
            all_passages = json.load(f)
        
        # Skip the specified number of records and take only the batch_size
        batch_passages = all_passages[skip_records:skip_records + batch_size]
        print(f"Loaded {len(all_passages)} total passages")
        print(f"Will process {len(batch_passages)} passages (records {skip_records+1} to {skip_records+len(batch_passages)})")
        
        # Use smaller internal batches to better monitor progress
        internal_batch_size = min(50, batch_size)
        
        # Function to process a single passage
        def process_single(passage):
            try:
                result = analyze_passage_with_perplexity(passage)
                if result and 'analysis_result' in result and result['analysis_result']:
                    # Get or create a thread-local database connection
                    if not hasattr(thread_local, 'db_conn'):
                        thread_local.db_conn = sqlite3.connect(RESULTS_DB_FILE)
                        # Enable foreign keys
                        thread_local.db_conn.execute('PRAGMA foreign_keys = ON')
                    
                    # Store in database using the thread-local connection
                    stored = store_passage_in_database(thread_local.db_conn, result['analysis_result'])
                    return True
                return False
            except Exception as e:
                print(f"Error processing passage: {str(e)}")
                return False
            finally:
                # Close the thread-local connection when done
                if hasattr(thread_local, 'db_conn'):
                    thread_local.db_conn.close()
                    del thread_local.db_conn
        
        # Process passages in internal batches
        total_processed = 0
        total_batches = (len(batch_passages) + internal_batch_size - 1) // internal_batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * internal_batch_size
            end_idx = min(start_idx + internal_batch_size, len(batch_passages))
            current_batch = batch_passages[start_idx:end_idx]
            
            print(f"\nProcessing internal batch {batch_idx+1}/{total_batches} ({len(current_batch)} passages)...")
            
            # Submit all tasks to the executor
            future_to_passage = {executor.submit(process_single, passage): passage for passage in current_batch}
            
            # Process completed tasks as they finish
            batch_success = 0
            for future in future_to_passage:
                try:
                    if future.result():
                        batch_success += 1
                except Exception as e:
                    print(f"Task generated an exception: {str(e)}")
            
            total_processed += batch_success
            
            print(f"Completed internal batch {batch_idx+1}/{total_batches}")
            print(f"Successfully processed {batch_success}/{len(current_batch)} in this batch")
            print(f"Total processed so far: {total_processed}/{len(batch_passages)}")
        
        print(f"\n{'='*80}")
        print(f"BATCH PROCESSING COMPLETE")
        print(f"Successfully processed {total_processed}/{len(batch_passages)} passages")
        print(f"{'='*80}")
        
        return total_processed
        
    except Exception as e:
        print(f"Error in batch processing: {str(e)}")
        traceback.print_exc()
        return 0
    finally:
        if 'db_conn' in locals():
            db_conn.close()
        executor.shutdown()

# Add this function to directly test raw API response
def test_raw_api_response():
    """Test the raw API response format for a single passage"""
    print("TESTING RAW API RESPONSE FORMAT")
    
    try:
        # Load the first passage from the JSON file
        with open(JSON_INPUT_FILE, 'r', encoding='utf-8') as f:
            all_passages = json.load(f)
            
        if not all_passages:
            print("No passages found in the input file.")
            return
            
        test_passage = all_passages[0]
        
        # Extract some basic info for display
        if "document_metadata" in test_passage and "match_data" in test_passage:
            passage_text = test_passage["match_data"].get("passage", "")
            search_term = test_passage["match_data"].get("search_term", "")
        else:
            passage_text = test_passage.get("passage", "")
            search_term = test_passage.get("search_term", "")
        
        print(f"Testing with passage containing search term: '{search_term}'")
        print(f"First 100 chars of passage: {passage_text[:100]}...")
        
        # Create OpenAI client with Perplexity base URL
        client = OpenAI(
            api_key=PERPLEXITY_API_KEY,
            base_url="https://api.perplexity.ai"
        )
        
        # Create a very simple prompt to test the response format
        prompt = """Analyze this passage and return a valid JSON response with this format:
        {
          "result": "success",
          "analysis": "brief analysis here"
        }"""
        
        # Call the API
        response = client.chat.completions.create(
            model=PERPLEXITY_MODEL,
            messages=[{"role": "user", "content": prompt + "\n\nPassage: " + passage_text[:200]}],
            temperature=0.2,
            max_tokens=500
        )
        
        # Get the raw response
        raw_response = response.choices[0].message.content if response.choices else ""
        
        # Print the raw response for analysis
        print("\nRAW API RESPONSE:")
        print("="*80)
        print(raw_response)
        print("="*80)
        
        # Try various JSON extraction methods
        methods = [
            ("Direct JSON parsing", raw_response),
            ("After </think>", raw_response.split("</think>", 1)[1].strip() if "</think>" in raw_response else "No </think> tag found"),
            ("JSON between backticks", re.search(r'```(?:json)?\s*(.*?)\s*```', raw_response, re.DOTALL).group(1) if re.search(r'```(?:json)?\s*(.*?)\s*```', raw_response, re.DOTALL) else "No JSON in backticks found"),
            ("First JSON-like structure", re.search(r'({[\s\S]*})', raw_response).group(1) if re.search(r'({[\s\S]*})', raw_response) else "No JSON-like structure found")
        ]
        
        print("\nTRYING DIFFERENT EXTRACTION METHODS:")
        for method_name, content in methods:
            print(f"\n{method_name}:")
            print("-"*40)
            print(content[:200] + "..." if len(content) > 200 else content)
            try:
                parsed = json.loads(content)
                print("PARSING SUCCESS!")
                print(f"Parsed JSON: {json.dumps(parsed, indent=2, ensure_ascii=False)}")
            except json.JSONDecodeError as e:
                print(f"Parsing failed: {str(e)}")
        
        print("\nAPI RESPONSE TEST COMPLETE")
        
    except Exception as e:
        print(f"Error testing API response: {str(e)}")
        traceback.print_exc()

# Main execution
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze legal passages with Perplexity AI')
    parser.add_argument('--test', action='store_true', help='Run in test mode (process 5 records only)')
    parser.add_argument('--batch', type=int, help='Process a specific number of records (e.g., 200)')
    parser.add_argument('--parallel', type=int, default=20, 
                        help='Maximum number of concurrent requests (default: 20, max: 400)')
    parser.add_argument('--skip', type=int, default=0,
                        help='Skip this many records from the start (useful for resuming)')
    parser.add_argument('--test-api', action='store_true', 
                        help='Test the raw API response format')
    args = parser.parse_args()
    
    # Add validation for parallel argument
    if args.parallel and (args.parallel < 1 or args.parallel > 400):
        print("Error: --parallel must be between 1 and 400")
        sys.exit(1)
    
    # Check if the input file exists
    if not os.path.exists(JSON_INPUT_FILE):
        print(f"Error: Input file '{JSON_INPUT_FILE}' not found!")
        print("Please run lex_api_fetcher.py first to generate the input data.")
        sys.exit(1)
    
    # Run API response test if specified
    if args.test_api:
        test_raw_api_response()
        sys.exit(0)
    
    # Run in test mode if specified
    if args.test:
        test_process_5_records()
        sys.exit(0)
    
    # Run in batch mode if specified
    if args.batch:
        # Update MAX_WORKERS constant if parallel argument is provided
        if args.parallel:
            MAX_WORKERS = args.parallel
        
        # Process the specified batch
        success_count = process_batch_parallel(args.batch, MAX_WORKERS, args.skip)
        
        print("\nBatch processing complete!")
        print(f"Successfully analyzed and stored {success_count} passages in {RESULTS_DB_FILE}")
        
        # Print instructions for viewing results
        print("\nTo view results:")
        print(f"1. Open the SQLite database: {os.path.abspath(RESULTS_DB_FILE)}")
        print("2. Query the passages table for results")
        print("   Example SQL: SELECT * FROM passages WHERE classification = 'Instructional Term' AND binding_conversion LIKE 'Feasible: Yes%'")
        
        sys.exit(0)
    
    try:
        # Load passages from the JSON file
        print(f"Loading passages from {JSON_INPUT_FILE}...")
        with open(JSON_INPUT_FILE, 'r', encoding='utf-8') as f:
            passages = json.load(f)
        
        print(f"Loaded {len(passages)} passages for analysis")
        
        # Process passages with Perplexity and store in database
        success_count = process_passages_with_perplexity_and_store(passages)
        
        print("\nPerplexity analysis complete!")
        print(f"Successfully analyzed and stored {success_count} passages in {RESULTS_DB_FILE}")
        
        # Print instructions for viewing results
        print("\nTo view results:")
        print(f"1. Open the SQLite database: {os.path.abspath(RESULTS_DB_FILE)}")
        print("2. Query the passages table for results")
        print("   Example SQL: SELECT * FROM passages WHERE classification = 'Instructional Term' AND binding_conversion LIKE 'Feasible: Yes%'")
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        print(traceback.format_exc())
        sys.exit(1)

print("\nPerplexity analyzer script completed.") 