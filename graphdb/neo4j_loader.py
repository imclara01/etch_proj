import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Neo4jLoader:
    def __init__(self):
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def clear_database(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("🧹 Database cleared.")

    def load_initial_data(self):
        """
        Loads a mockup schema: Sensor -> monitors -> Part -> affects -> Fault -> fixed_by -> SOP
        """
        cypher_query = """
        // 1. Create Sensors
        CREATE (s1:Sensor {name: 'TCP Top Pwr', description: 'Transformer Coupled Plasma Top Power'})
        CREATE (s2:Sensor {name: 'RF Btm Pwr', description: 'Radio Frequency Bottom Power'})
        CREATE (s3:Sensor {name: 'Cl2 Flow', description: 'Chlorine Gas Flow Rate'})
        CREATE (s4:Sensor {name: 'Pressure', description: 'Chamber Pressure'})
        CREATE (s5:Sensor {name: 'Vat Valve', description: 'Vacuum Valve Position'})
        CREATE (s6:Sensor {name: 'TCP Load', description: 'TCP Matcher Load Position'})

        // 2. Create Parts
        CREATE (p1:Part {name: 'TCP Matcher', description: 'Power matching network for top electrode'})
        CREATE (p2:Part {name: 'ESC', description: 'Electrostatic Chuck'})
        CREATE (p3:Part {name: 'MFC', description: 'Mass Flow Controller'})
        CREATE (p4:Part {name: 'APC Valve', description: 'Automatic Pressure Control Valve'})

        // 3. Create Faults
        CREATE (f1:Fault {name: 'TCP Top Pwr Fault', severity: 'High'})
        CREATE (f2:Fault {name: 'Cl2 Flow Fault', severity: 'Medium'})
        CREATE (f3:Fault {name: 'Pressure Fault', severity: 'High'})

        // 4. Create SOPs (Standard Operating Procedures)
        CREATE (sop1:SOP {title: 'TCP Matcher Inspection', steps: '1. Check RF cable connection. 2. Inspect capacitor state. 3. Recalibrate matcher.'})
        CREATE (sop2:SOP {title: 'Gas Line Purge', steps: '1. Shut down gas supply. 2. Purge line with N2. 3. Replace MFC filter.'})
        CREATE (sop3:SOP {title: 'Vacuum Leak Test', steps: '1. Close VAT valve. 2. Monitor pressure rise rate. 3. Check O-ring seal.'})

        // 5. Define Relationships
        CREATE (s1)-[:MONITORS]->(p1)
        CREATE (s6)-[:MONITORS]->(p1)
        CREATE (p1)-[:AFFECTS]->(f1)-[:FIXED_BY]->(sop1)

        CREATE (s3)-[:MONITORS]->(p3)-[:AFFECTS]->(f2)-[:FIXED_BY]->(sop2)

        CREATE (s4)-[:MONITORS]->(p4)
        CREATE (s5)-[:MONITORS]->(p4)
        CREATE (p4)-[:AFFECTS]->(f3)-[:FIXED_BY]->(sop3)
        """
        with self.driver.session() as session:
            session.run(cypher_query)
            print("🚀 Initial knowledge graph loaded successfully.")

    def get_recommendation(self, fault_name):
        """
        Sample query for testing
        """
        query = """
        MATCH (f:Fault {name: $fault_name})-[:FIXED_BY]->(sop:SOP)
        RETURN sop.title as title, sop.steps as steps
        """
        with self.driver.session() as session:
            result = session.run(query, fault_name=fault_name)
            return [dict(record) for record in result]

if __name__ == "__main__":
    loader = Neo4jLoader()
    try:
        loader.clear_database()
        loader.load_initial_data()
        
        # Test Query
        recs = loader.get_recommendation('TCP Top Pwr Fault')
        print(f"Test Recommendation: {recs}")
    finally:
        loader.close()
