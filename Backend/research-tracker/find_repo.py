from gradio_client import Client
from find_paper_link import find_link
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


def get_repo_link(paper_path: str):
    paper_link = find_link(paper_path)
    if not paper_link:
        print("Could not find paper link.")
        return None
    paper_url, paper_name, code_repo = find_paper(paper_link)
    if paper_url is None and paper_name is None and code_repo is None:
        print("Could not find paper information from MCP.")
        return None
    if not code_repo:
        print("Could not find code repository for: " + paper_name + " (" + paper_url + ")")
        return None
    print("Found code repository for: " + paper_name + " (" + paper_url + "): " + code_repo)
    return code_repo

if __name__ == "__main__":
    get_repo_link("../papercodesync/example/paper.pdf")