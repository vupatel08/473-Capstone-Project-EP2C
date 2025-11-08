from find_paper_link import find_link
from find_repo import find_paper


PAPER_PATH = "../papercodesync/example/paper.pdf"

def get_repo_link():
    paper_link = find_link(PAPER_PATH)
    if not paper_link:
        print("Could not find paper link.")
        return None, None, None
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
    get_repo_link()