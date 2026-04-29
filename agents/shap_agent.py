from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

if "OPEN_AI_API_KEY" in os.environ and "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = os.environ["OPEN_AI_API_KEY"]

class SHAPAgent:
    def __init__(self, model_name="gpt-4o"):
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert Semiconductor Process Engineer.
Your task is to analyze sensor data anomalies based on SHAP values and raw sensor statistics.

Context:
- Process: Plasma Etching
- You will receive a list of top sensors that contributed to a fault detection.
- Each sensor entry includes: Current Value, Mean (Normal), Normal Range, Status (High/Low), and SHAP Direction.

Guidelines:
1. Explain the physical impact of the sensor being 'High' or 'Low' for the specific fault.
2. Link the sensor status to potential equipment issues (e.g., "Pressure is High, suggesting a Throttle Valve or Vacuum Pump issue").
3. Use the SHAP direction to confirm if the sensor's current state is what primarily drove the AI's decision.
4. Output should be in Korean."""),
            ("user", "Detected Fault: {fault_name}\nSensor Analysis Data:\n{analysis_json}\n\nPlease provide a deep technical analysis.")
        ])
        self.chain = self.prompt | self.llm | StrOutputParser()

    def explain_fault(self, fault_name, analysis_results):
        """
        analysis_results: list of dicts from SHAPExplainer.explain
        """
        # Convert list to a readable string for the LLM
        analysis_str = json.dumps(analysis_results, indent=2, ensure_ascii=False)
        
        try:
            explanation = self.chain.invoke({
                "fault_name": fault_name,
                "analysis_json": analysis_str
            })
            return explanation
        except Exception as e:
            return f"Error generating explanation: {str(e)}"

if __name__ == "__main__":
    agent = SHAPAgent()
    print("SHAPAgent updated.")
