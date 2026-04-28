import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

uri = os.getenv("NEO4J_URI")
# Try 'neo4j' as username first, then fallback to .env value
usernames = ['neo4j', os.getenv("NEO4J_USERNAME")]
password = os.getenv("NEO4J_PASSWORD")

print(f"Connecting to {uri}...")

for user in usernames:
    print(f"Attempting with user: {user}")
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session() as session:
            result = session.run("RETURN 1 as result")
            print(f"✅ Success with user {user}!")
            driver.close()
            break
    except Exception as e:
        print(f"❌ Failed with user {user}: {e}")
