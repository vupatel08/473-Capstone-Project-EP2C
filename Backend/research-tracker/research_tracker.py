from gradio_client import Client
import json

SPACE_ID = "dylanebert/research-tracker-mcp"

def normalize(result):
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

def main():
    query = input("Enter paper name or URL: ").strip()
    if not query:
        print("No input provided.")
        return

    client = Client(SPACE_ID)
    print("Searching...")

    raw = client.predict(input_data=query, api_name="/find_research_relationships")
    paper_url, paper_name, code_repo = normalize(raw)

    print("\nResearch Tracker Results")
    print("Paper URL: ", paper_url or "(none found)")
    print("Paper Name:", paper_name or "(none found)")
    print("Code Repo: ", code_repo or "(none found)")

if __name__ == "__main__":
    main()
