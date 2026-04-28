from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Handle case where key might be named OPEN_AI_API_KEY instead of OPENAI_API_KEY
if "OPEN_AI_API_KEY" in os.environ and "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = os.environ["OPEN_AI_API_KEY"]

class SHAPAgent:
    def __init__(self, model_name="gpt-4o"):
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert Semiconductor Process Engineer specializing in Plasma Etching (Dry Etching).
Your task is to provide a clear, professional explanation of why a specific fault was detected based on SHAP (SHapley Additive exPlanations) values.

Context:
- Process: Etching
- Fault Types: TCP Top Pwr, RF Btm Pwr, Cl2 Flow, BCl3 Flow, Pressure, He Press, etc.
- Input: Top 3 sensors contributing to the prediction and their SHAP values (positive values increase the probability of that fault).

Guidelines:
1. Explain the physical meaning of the top sensors in the context of the detected fault.
2. Provide actionable insights or potential hardware issues (e.g., MFC malfunction, ESC problem, RF Matcher instability).
3. Keep the tone professional and technical.
4. Output should be in Korean as the primary users are Korean engineers."""),
            ("user", "Detected Fault: {fault_name}\nTop Contributing Sensors:\n{sensor_info}\n\nPlease provide an expert explanation.")
        ])
        self.chain = self.prompt | self.llm | StrOutputParser()

    def explain_fault(self, fault_name, contributions):
        """
        contributions: dict of {sensor_name: shap_value}
        """
        sensor_info = "\n".join([f"- {sensor}: {val:.4f}" for sensor, val in contributions.items()])
        
        try:
            explanation = self.chain.invoke({
                "fault_name": fault_name,
                "sensor_info": sensor_info
            })
            return explanation
        except Exception as e:
            return f"Error generating explanation: {str(e)}"

if __name__ == "__main__":
    # Test (requires OPENAI_API_KEY environment variable)
    agent = SHAPAgent()
    sample_contribs = {
        "TCP Top Pwr": 0.45,
        "TCP Load": 0.32,
        "Pressure": 0.12
    }
    # print(agent.explain_fault("TCP Top Pwr Fault", sample_contribs))
    print("SHAPAgent initialized. (Set OPENAI_API_KEY to run full test)")
