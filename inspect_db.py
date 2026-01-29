import sqlite3

db_path = "databases/onepager_credito.db"

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    print(f"Tables in {db_path}:")
    for t in tables:
        table_name = t[0]
        print(f"\n--- Table: {table_name} ---")
        # Get column info
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [info[1] for info in cursor.fetchall()]
        print("Columns:", columns)
        
    conn.close()
except Exception as e:
    print(f"Error: {e}")
