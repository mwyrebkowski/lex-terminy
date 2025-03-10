import sqlite3
import csv
import json
import os
import sys
import argparse

# Database file path
DB_FILE = "legal_passages.db"
DEFAULT_OUTPUT_FILE = "perplexity_results.csv"

def export_perplexity_results(output_file, filter_classification=None, filter_feasible=None):
    """
    Export results from the database to a CSV file
    
    Parameters:
    - output_file: Path to the output CSV file
    - filter_classification: Optional filter for specific classification (e.g., "Instructional Term")
    - filter_feasible: Optional filter for binding conversion feasibility (True/False)
    """
    try:
        # Connect to the database
        print(f"Connecting to database: {DB_FILE}")
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Build query based on filters
        query = "SELECT * FROM passages WHERE 1=1"
        params = []
        
        if filter_classification:
            query += " AND classification = ?"
            params.append(filter_classification)
        
        if filter_feasible is not None:
            query += " AND json_data LIKE ?"
            # The JSON contains the feasible field
            feasible_value = "true" if filter_feasible else "false"
            params.append(f'%"feasible": {feasible_value}%')
        
        # Execute the query
        print(f"Executing query: {query}")
        cursor.execute(query, params)
        
        # Get column names
        column_names = [description[0] for description in cursor.description]
        
        # Fetch all matching records
        rows = cursor.fetchall()
        print(f"Found {len(rows)} matching records")
        
        if not rows:
            print("No matching records found.")
            return
        
        # Write to CSV
        print(f"Writing results to {output_file}")
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header row
            writer.writerow(column_names)
            
            # Write data rows
            for row in rows:
                writer.writerow(row)
        
        print(f"Successfully exported {len(rows)} records to {output_file}")
        
        # Extract some statistics
        if rows:
            # Count by classification
            classification_counts = {}
            feasible_counts = {"true": 0, "false": 0}
            
            for row in rows:
                # Find the classification column index
                class_idx = column_names.index('classification')
                classification = row[class_idx]
                
                # Count by classification
                if classification not in classification_counts:
                    classification_counts[classification] = 0
                classification_counts[classification] += 1
                
                # Count by feasibility
                json_idx = column_names.index('json_data')
                if json_idx >= 0 and row[json_idx]:
                    try:
                        json_data = json.loads(row[json_idx])
                        if "binding_conversion" in json_data and "feasible" in json_data["binding_conversion"]:
                            if json_data["binding_conversion"]["feasible"]:
                                feasible_counts["true"] += 1
                            else:
                                feasible_counts["false"] += 1
                    except json.JSONDecodeError:
                        pass
            
            # Print statistics
            print("\nStatistics:")
            print("Classification Counts:")
            for classification, count in classification_counts.items():
                print(f"  {classification}: {count}")
            
            print("\nBinding Conversion Feasibility:")
            print(f"  Feasible: {feasible_counts['true']}")
            print(f"  Not Feasible: {feasible_counts['false']}")
            
    except Exception as e:
        print(f"Error exporting results: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export Perplexity analysis results to CSV')
    parser.add_argument('--output', '-o', default=DEFAULT_OUTPUT_FILE,
                      help=f'Output CSV file (default: {DEFAULT_OUTPUT_FILE})')
    parser.add_argument('--classification', '-c', 
                      help='Filter by classification (e.g., "Instructional Term")')
    parser.add_argument('--feasible', '-f', action='store_true',
                      help='Filter for feasible binding conversion')
    parser.add_argument('--not-feasible', '-n', action='store_true',
                      help='Filter for not feasible binding conversion')
    
    args = parser.parse_args()
    
    # Check if database file exists
    if not os.path.exists(DB_FILE):
        print(f"Error: Database file '{DB_FILE}' not found!")
        sys.exit(1)
    
    # Handle incompatible arguments
    if args.feasible and args.not_feasible:
        print("Error: Cannot specify both --feasible and --not-feasible")
        sys.exit(1)
    
    # Determine feasibility filter
    feasible_filter = None
    if args.feasible:
        feasible_filter = True
    elif args.not_feasible:
        feasible_filter = False
    
    # Run export
    export_perplexity_results(args.output, args.classification, feasible_filter) 