from gradio_client import Client
import json

SPACE_ID = "dylanebert/research-tracker-mcp"

def _normalize(result):
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except Exception:
            pass

    if isinstance(result, dict):
        paper_url = result.get("paper") or ""
        code_repo = result.get("code") or ""
        name = result.get("name") or ""
        return paper_url, name, code_repo

    if isinstance(result, (list, tuple)):
        vals = list(result) + ["", "", ""]
        return vals[0], vals[2], vals[1] 

    return "", "", ""

def find_paper(paper_link: str):
    client = Client(SPACE_ID)
    raw = client.predict(input_data=paper_link, api_name="/find_research_relationships")
    paper_url, paper_name, code_repo = _normalize(raw)
    return paper_url, paper_name, code_repo