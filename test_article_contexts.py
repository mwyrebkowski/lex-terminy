import sqlite3
import re
import argparse

# Database file
DB_FILE = "legal_passages.db"

def test_article_extraction(num_samples=5):
    """
    Simple test to fetch and print article contexts from the database
    """
    try:
        # Connect to the database
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Fetch sample records
        cursor.execute("""
            SELECT id, search_term, passage, article, full_context 
            FROM passages
            LIMIT ?
        """, (num_samples,))
        
        records = cursor.fetchall()
        print(f"Found {len(records)} records to examine\n")
        
        # Process each record
        for record in records:
            record_id, search_term, passage, article, full_context = record
            
            print(f"=== RECORD ID: {record_id} ===")
            print(f"Search Term: {search_term}")
            print(f"Current Article Marker: {article}")
            print(f"Passage: {passage[:100]}..." if len(passage) > 100 else f"Passage: {passage}")
            
            # Find article markers in the full context
            article_markers = []
            for match in re.finditer(r'(Art(?:ykuł)?\s*\d+[a-z]?\.?|§\s*\d+[a-z]?\.?)', full_context, re.IGNORECASE):
                article_markers.append({
                    'marker': match.group(0),
                    'position': match.start()
                })
            
            print(f"\nFull Context Length: {len(full_context)} characters")
            print(f"Found {len(article_markers)} article markers in the full context")
            
            if article_markers:
                print("\nArticle Markers Found:")
                for i, marker in enumerate(article_markers):
                    print(f"  {i+1}. '{marker['marker']}' at position {marker['position']}")
                    
                    # Show some context around this marker
                    start = max(0, marker['position'] - 20)
                    end = min(len(full_context), marker['position'] + 60)
                    context = full_context[start:end]
                    print(f"      Context: ...{context}...")
            
            # Try to identify which article contains the passage
            article_containing_passage = None
            if article_markers:
                # Sort by position
                article_markers.sort(key=lambda x: x['position'])
                
                # Find which article segment contains the passage
                passage_pos = full_context.find(passage)
                if passage_pos >= 0:
                    for i, marker in enumerate(article_markers):
                        start_pos = marker['position']
                        end_pos = article_markers[i+1]['position'] if i < len(article_markers) - 1 else len(full_context)
                        
                        if start_pos <= passage_pos < end_pos:
                            article_containing_passage = {
                                'marker': marker['marker'],
                                'start': start_pos,
                                'end': end_pos
                            }
                            break
            
            if article_containing_passage:
                print(f"\nPassage appears to be in: {article_containing_passage['marker']}")
                # Extract just the article content
                article_text = full_context[article_containing_passage['start']:article_containing_passage['end']]
                print(f"This article's length: {len(article_text)} characters")
                print(f"Article excerpt: {article_text[:100]}..." if len(article_text) > 100 else f"Article: {article_text}")
            else:
                print("\nCould not determine which article contains the passage")
                
            print("\n" + "=" * 80 + "\n")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple test for article context extraction')
    parser.add_argument('--samples', '-n', type=int, default=5, 
                      help='Number of records to sample (default: 5)')
    
    args = parser.parse_args()
    test_article_extraction(args.samples) 