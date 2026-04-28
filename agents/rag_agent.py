from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class GraphRAGAgent:
    def __init__(self, model_name="gpt-4o"):
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        
        # Neo4j Driver
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Senior Semiconductor Maintenance Expert. 
Your goal is to provide specific troubleshooting recommendations and Standard Operating Procedures (SOPs) based on information retrieved from a Knowledge Graph.

Retrieved Context:
{context}

Detected Fault: {fault_name}

Guidelines:
1. Clearly identify which part is likely causing the issue.
2. Summarize the steps for the engineer to follow based on the SOP.
3. Be concise and practical.
4. Output should be in Korean."""),
            ("user", "What are the recommended countermeasures for the {fault_name}?")
        ])
        self.chain = self.prompt | self.llm | StrOutputParser()

    def close(self):
        self.driver.close()

    def get_context_from_graph(self, fault_name):
        """
        Retrieves Fault -> Part and Fault -> SOP relationships from Neo4j
        """
        query = """
        MATCH (f:Fault {name: $fault_name})
        OPTIONAL MATCH (f)<-[:DETECTS]-(task:Task)
        OPTIONAL MATCH (f)-[:FIXED_BY]->(sop:SOP)
        OPTIONAL MATCH (cause:Cause)-[:RECOMMENDED_ACTION]->(sop)
        OPTIONAL MATCH (cause)-[:AFFECTS_COMPONENT]->(comp:Component)
        OPTIONAL MATCH (vsm:VirtualSensorModel)-[:PREDICTS]->(ws:WaferState)-[:INDICATES]->(risk:FaultRisk)
        WHERE (f.name CONTAINS "Pressure" AND risk.name CONTAINS "Over Etch") OR f.name = "TCP Top Pwr Fault"
        
        RETURN 
            f.name as fault,
            sop.title as sop_title, 
            sop.steps as sop_steps,
            collect(DISTINCT comp.name) as components,
            cause.name as cause_name,
            ws.name as impacted_quality
        """
        with self.driver.session() as session:
            result = session.run(query, fault_name=fault_name)
            records = [dict(record) for record in result]
            
            if not records or not records[0]['sop_title']:
                return "No specific SOP found in the knowledge graph for this fault."
            
            context = ""
            for r in records:
                context += f"- Fault: {r['fault']}\n"
                context += f"- SOP Title: {r['sop_title']}\n"
                context += f"- SOP Steps: {r['sop_steps']}\n"
                if r['components']:
                    context += f"- Affected Components: {', '.join(r['components'])}\n"
                if r['cause_name']:
                    context += f"- Potential Cause: {r['cause_name']}\n"
                if r['impacted_quality']:
                    context += f"- Impacted Wafer Quality: {r['impacted_quality']}\n"
            return context

    def get_recommendation(self, fault_name):
        # 1. Retrieve data from Neo4j
        context = self.get_context_from_graph(fault_name)
        
        # 2. Generate response using LLM
        try:
            response = self.chain.invoke({
                "context": context,
                "fault_name": fault_name
            })
            return response
        except Exception as e:
            return f"Error generating recommendation: {str(e)}"

if __name__ == "__main__":
    agent = GraphRAGAgent()
    try:
        # Test with an existing fault
        fault = "TCP Top Pwr Fault"
        print(f"--- Recommendation for {fault} ---")
        print(agent.get_recommendation(fault))
    finally:
        agent.close()
