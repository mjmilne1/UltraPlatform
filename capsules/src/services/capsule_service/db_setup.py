import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import os
from dotenv import load_dotenv
load_dotenv()

db_url = os.getenv('DATABASE_URL')
parts = db_url.replace('postgresql://', '').split('@')
user_pass = parts[0].split(':')
host_port_db = parts[1].split('/')

user, password = user_pass[0], user_pass[1]
host = host_port_db[0].split(':')[0]
dbname = host_port_db[1]

print(f"Connecting to PostgreSQL at {host}")
try:
    conn = psycopg2.connect(dbname='postgres', user=user, password=password, host=host)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()
    cursor.execute(f"SELECT 1 FROM pg_database WHERE datname='{dbname}'")
    if not cursor.fetchone():
        cursor.execute(f'CREATE DATABASE {dbname}')
        print(f"✓ Created: {dbname}")
    else:
        print(f"✓ Exists: {dbname}")
    cursor.close()
    conn.close()
    from models import init_db
    init_db()
except Exception as e:
    print(f"✗ Error: {e}")
