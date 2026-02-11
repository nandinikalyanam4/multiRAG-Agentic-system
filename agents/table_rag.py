

# ======================== FILE: agents/table_rag.py ========================
"""
AGENT 6: TABLE RAG
--------------------
WHAT IT DOES: For structured data. Retrieves schema, then generates pandas code to answer.
WHY LEARN IT: Standard RAG fails on "what's the average X grouped by Y?" questions.
KEY INSIGHT: Don't retrieve rows — generate code that queries the data.
"""
from agents.base import BaseRAGAgent
from vectorstore import store_manager
from llm import llm_call
import pandas as pd

class TableRAGAgent(BaseRAGAgent):
    name = "table_rag"
    description = "Handles structured data (CSV/Excel). Generates pandas code to answer analytical questions."

    COLLECTION = "table_rag"

    async def retrieve(self, query, top_k=5):
        # Always retrieve schema first
        results = store_manager.search(
            self.COLLECTION, query, k=top_k,
            filter_dict=None  # get all types including schema
        )
        return results

    async def generate(self, query, context):
        # Find schema chunk and CSV path
        schema_doc = None
        csv_path = None
        for doc in context:
            if doc.metadata.get("chunk_type") == "schema":
                schema_doc = doc
                csv_path = doc.metadata.get("csv_path")
                break

        if not csv_path or not schema_doc:
            # Fallback to text-based answer
            ctx = "\n\n".join([d.page_content for d in context])
            return llm_call(
                "Answer based on the tabular data in context.",
                f"Context:\n{ctx}\n\nQuestion: {query}"
            )

        # Generate pandas code
        code = llm_call(
            f"""You are a pandas expert. Given this table schema, write Python code to answer the question.
Schema:
{schema_doc.page_content}

Rules:
- Assume df is already loaded as a pandas DataFrame
- Write ONLY the code, no explanations
- Store the final result in a variable called `result`
- Handle missing values with .fillna() or .dropna()
- Use .to_string() for the final result""",
            f"Question: {query}"
        )

        # Execute the code safely
        try:
            code_clean = code.replace("```python", "").replace("```", "").strip()
            df = pd.read_csv(csv_path) if csv_path.endswith(".csv") else pd.read_excel(csv_path)
            local_vars = {"df": df, "pd": pd}
            exec(code_clean, {}, local_vars)
            result = local_vars.get("result", "No result computed")

            return f"**Analysis Result:**\n{result}\n\n**Code Used:**\n```python\n{code_clean}\n```"

        except Exception as e:
            # Fallback: use context directly
            ctx = "\n\n".join([d.page_content for d in context])
            fallback = llm_call(
                "The code execution failed. Answer based on the data shown in context.",
                f"Context:\n{ctx}\n\nQuestion: {query}\n\nError: {str(e)}"
            )
            return f"{fallback}\n\n[Note: Pandas execution failed: {str(e)}]"

