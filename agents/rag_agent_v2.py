"""
GraphRAG Agent v2 — Two-stage retrieval + SHAP fingerprint matching

Token optimization techniques (estimated 40-60 percent saving vs v1):
  1. SHAP context: JSON dump -> compact natural-language line (~25 vs ~120 tokens)
  2. SOP: abstract first, full steps fetched only on follow-up (~70 vs ~400 tokens)
  3. Cypher returns trimmed property set, not whole node dicts
  4. System prompt: 4-bullet guideline -> 2-line role + format directive

Retrieval design (paper-grounded):
  Stage 1 (Coarse):
     SHAP top-k sensors -> match against Fault.sensors_low/high fingerprints
     -> rank candidate Faults by overlap score
     -> aggregate up to FaultCategory if no single fault dominates
  Stage 2 (Fine, only if user asks for detail):
     For top-1 fault, traverse Fault -> Mechanism -> Component -> SOP (abstract)
     Full SOPStep fetch only on explicit user request

References:
  - Wise 1999 fault catalog -> fingerprint matching
  - Sofge 1997 g_model -> wafer-state risk signals
  - Microsoft GraphRAG (Edge et al. 2024) -> coarse-to-fine pattern
"""

import os
from typing import Optional, Dict, List, Any, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()


# ------------------------------------------------------------------
# SHAP -> compact natural language (token saver #1)
# ------------------------------------------------------------------
def compact_shap(shap_analysis: Optional[Dict[str, Any]]) -> str:
    """
    Convert SHAP dict into a single short line.

    Input:
      {"He_Pressure":   {"value": -2.3, "direction": "Low",  "rank": 1},
       "TCP_Top_Power": {"value":  1.8, "direction": "High", "rank": 2}}
    Output:
      "Top SHAP deviations: He Pressure (Low, -2.3, rank 1); TCP Top Power (High, +1.8, rank 2)"
    """
    if not shap_analysis:
        return "No SHAP analysis provided."
    ranked = sorted(shap_analysis.items(),
                    key=lambda kv: kv[1].get("rank", 99))
    parts = []
    for sensor, info in ranked[:5]:  # cap at top-5; Sofge showed >5 vars degrade models
        direction = info.get("direction", "?")
        value = info.get("value", 0.0)
        rank = info.get("rank", "?")
        sign = "+" if value >= 0 else ""
        parts.append(f"{sensor} ({direction}, {sign}{value:.2f}, rank {rank})")
    return "Top SHAP deviations: " + "; ".join(parts)


def parse_shap_to_lists(shap_analysis: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """Split SHAP into (low_sensors, high_sensors) for Cypher matching.
    SHAP dict often uses underscores; KG uses spaces. Normalize.
    """
    low, high = [], []
    if not shap_analysis:
        return low, high
    for raw_name, info in shap_analysis.items():
        sensor_name = raw_name.replace("_", " ")
        d = info.get("direction", "").lower()
        if d == "low":
            low.append(sensor_name)
        elif d == "high":
            high.append(sensor_name)
    return low, high


class GraphRAGAgentV2:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        # Default to mini for cost.
        # Switch to gpt-4o only when the user asks for deep root-cause analysis.
        self.llm = ChatOpenAI(model=model_name, temperature=0)

        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

        # Slim system prompt — guidelines compressed.
        self.prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a senior semiconductor maintenance expert for the Lam 9600 metal etcher. "
             "Use the retrieved Knowledge Graph context and SHAP signals to answer in Korean. "
             "Output format: (1) most likely fault and confidence, (2) root cause mechanism, "
             "(3) recommended SOP (title + abstract), (4) the 2-3 most relevant steps. "
             "Cite the KG reference field where present. Be concise."),
            ("user",
             "Detected fault candidate: {fault_name}\n"
             "{shap_line}\n\n"
             "Knowledge Graph context:\n{kg_context}\n\n"
             "Question: {question}")
        ])
        self.chain = self.prompt | self.llm | StrOutputParser()

    def close(self):
        self.driver.close()

    # ------------------------------------------------------------------
    # Stage 1 — fingerprint matching
    # ------------------------------------------------------------------
    def stage1_match_fault(self, shap_analysis: Dict[str, Any],
                           top_n: int = 3) -> List[Dict]:
        low, high = parse_shap_to_lists(shap_analysis)
        if not low and not high:
            return []

        query = """
        MATCH (f:Fault)
        WITH f,
             size([s IN f.sensors_low  WHERE s IN $low ]) AS low_hits,
             size([s IN f.sensors_high WHERE s IN $high]) AS high_hits,
             size(f.sensors_low) + size(f.sensors_high)   AS fp_size
        WITH f, low_hits + high_hits AS hits, fp_size
        WHERE hits > 0
        WITH f, hits, fp_size,
             toFloat(hits) /
               CASE fp_size WHEN 0 THEN 1 ELSE fp_size END AS score
        OPTIONAL MATCH (f)-[:BELONGS_TO]->(cat:FaultCategory)
        RETURN f.fault_id    AS fault_id,
               f.name        AS name,
               f.induced_via AS induced_via,
               f.region      AS region,
               cat.name      AS category,
               score, hits
        ORDER BY score DESC, hits DESC
        LIMIT $top_n
        """
        with self.driver.session() as session:
            result = session.run(query, low=low, high=high, top_n=top_n)
            return [dict(r) for r in result]

    # ------------------------------------------------------------------
    # Stage 2 — causal chain + SOP abstract (no steps yet)
    # ------------------------------------------------------------------
    def stage2_fetch_chain(self, fault_id: str) -> Dict:
        query = """
        MATCH (f:Fault {fault_id:$fid})
        OPTIONAL MATCH (f)-[:CAUSED_BY]->(m:Mechanism)
        OPTIONAL MATCH (m)-[:DEGRADES]->(c:Component)
        OPTIONAL MATCH (m)-[:REMEDIATED_BY]->(sop:SOP)
        RETURN f.name      AS fault_name,
               f.region    AS region,
               f.reference AS fault_ref,
               collect(DISTINCT {
                 mechanism: m.name,
                 timescale: m.timescale,
                 symptom: m.symptom,
                 reference: m.reference
               }) AS mechanisms,
               collect(DISTINCT c.name) AS components,
               collect(DISTINCT {
                 sop_id: sop.sop_id,
                 title: sop.title,
                 abstract: sop.abstract,
                 reference: sop.reference
               }) AS sops
        """
        with self.driver.session() as session:
            result = session.run(query, fid=fault_id).single()
            if not result:
                return {}
            return dict(result)

    def fetch_sop_steps(self, sop_id: str) -> List[Dict]:
        """Expensive: only call when user asks for full procedure."""
        query = """
        MATCH (s:SOP {sop_id:$sop_id})-[:HAS_STEP]->(st:SOPStep)
        RETURN st.step_no AS step_no, st.text AS text
        ORDER BY st.step_no
        """
        with self.driver.session() as session:
            return [dict(r) for r in session.run(query, sop_id=sop_id)]

    # ------------------------------------------------------------------
    # Format chain to compact context (token saver #2)
    # ------------------------------------------------------------------
    @staticmethod
    def _format_chain(chain: Dict) -> str:
        if not chain:
            return "No causal chain found."
        lines = [
            f"Fault: {chain.get('fault_name','?')} "
            f"(region: {chain.get('region','?')}, ref: {chain.get('fault_ref','?')})"
        ]
        mechs = [m for m in chain.get("mechanisms", []) if m.get("mechanism")]
        if mechs:
            lines.append("Mechanisms:")
            for m in mechs:
                lines.append(
                    f"  - {m['mechanism']} "
                    f"(timescale: {m.get('timescale','?')}; "
                    f"symptom: {m.get('symptom','?')})"
                )
        comps = [c for c in chain.get("components", []) if c]
        if comps:
            lines.append(f"Affected components: {', '.join(comps)}")
        sops = [s for s in chain.get("sops", []) if s.get("sop_id")]
        if sops:
            lines.append("Recommended SOPs:")
            for s in sops:
                lines.append(f"  - [{s['sop_id']}] {s['title']}: {s['abstract']}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def recommend(self,
                  shap_analysis: Optional[Dict[str, Any]] = None,
                  fault_name_hint: Optional[str] = None,
                  question: str = "What is the recommended countermeasure?",
                  include_full_steps: bool = False) -> Dict:
        """
        Returns:
            'candidates'      : stage-1 ranking
            'chain'           : stage-2 causal chain for top-1
            'answer'          : LLM final text
            'token_estimate'  : rough context token count
        """
        # Stage 1
        candidates = self.stage1_match_fault(shap_analysis or {}, top_n=3)
        if not candidates and fault_name_hint:
            with self.driver.session() as session:
                rec = session.run(
                    "MATCH (f:Fault) WHERE f.name CONTAINS $name "
                    "RETURN f.fault_id AS fault_id LIMIT 1",
                    name=fault_name_hint
                ).single()
                if rec:
                    candidates = [{"fault_id": rec["fault_id"],
                                   "name": fault_name_hint,
                                   "score": 0.0}]

        if not candidates:
            return {
                "candidates": [],
                "chain": {},
                "answer": "SHAP에 매칭되는 fault가 KG에 없습니다. 센서 이름 정규화를 확인하세요.",
                "token_estimate": 0,
            }

        # Stage 2 for top-1
        top = candidates[0]
        chain = self.stage2_fetch_chain(top["fault_id"])

        full_steps_text = ""
        if include_full_steps and chain.get("sops"):
            sop_id = chain["sops"][0]["sop_id"]
            steps = self.fetch_sop_steps(sop_id)
            full_steps_text = "\n\nFull SOP steps:\n" + "\n".join(
                f"  {s['step_no']}. {s['text']}" for s in steps
            )

        kg_context = self._format_chain(chain) + full_steps_text
        shap_line = compact_shap(shap_analysis)
        token_estimate = (len(kg_context) + len(shap_line)) // 4

        try:
            answer = self.chain.invoke({
                "fault_name": top["name"],
                "shap_line": shap_line,
                "kg_context": kg_context,
                "question": question,
            })
        except Exception as e:
            answer = f"LLM error: {e}"

        return {
            "candidates": candidates,
            "chain": chain,
            "answer": answer,
            "token_estimate": token_estimate,
        }


# ------------------------------------------------------------------
# Demo
# ------------------------------------------------------------------
if __name__ == "__main__":
    agent = GraphRAGAgentV2(model_name="gpt-4o-mini")
    try:
        # Wise 1999 Fault #9 (He Chuck) pattern
        shap = {
            "Helium_Pressure":  {"value": -2.8, "direction": "Low",  "rank": 1},
            "Chamber_Pressure": {"value":  0.4, "direction": "High", "rank": 2},
        }
        out = agent.recommend(
            shap_analysis=shap,
            question="이 이상 징후의 원인과 점검 절차를 알려주세요."
        )
        print("=" * 60)
        print("Stage-1 candidates:")
        for c in out["candidates"]:
            print(f"  {c['fault_id']} {c['name']:15s} "
                  f"score={c['score']:.2f} cat={c.get('category')}")
        print(f"\nContext tokens (est): {out['token_estimate']}")
        print("\nLLM answer:\n")
        print(out["answer"])
    finally:
        agent.close()
