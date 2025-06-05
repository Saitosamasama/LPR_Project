import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / 'vehicle.db'

def view_tables():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get list of tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    print("\n=== Tables in Database ===")
    for table in tables:
        table_name = table[0]
        print(f"\n--- Table: {table_name} ---")
        
        # Get column names
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        column_names = [col[1] for col in columns]
        print("Columns:", ", ".join(column_names))
        
        # Get all rows
        cursor.execute(f"SELECT * FROM {table_name}")
        rows = cursor.fetchall()
        print(f"\nTotal rows: {len(rows)}")
        print("\nFirst 10 rows:")
        for row in rows[:10]:
            print(row)
    
    conn.close()

if __name__ == "__main__":
    view_tables() 
