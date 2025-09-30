from gradio_client import Client

# Instantiate the client with the HuggingFace model
client = Client("dylanebert/research-tracker-mcp")

# Function to get comprehensive research metadata
def get_research_metadata(input_data):
    result = client.predict(
        input_data=input_data,
        api_name="/find_research_relationships"
    )
    return result


if __name__ == "__main__":
	result = get_research_metadata(("https://arxiv.org/abs/2404.15592"))
	print("\n\nPaper Metadata:\n")
	print("Paper URL:", result['paper'])
	print("Title:", result['name'])
	print("Authors:", result['authors'])
	print("Date Published:", result['date'])
	print("Code Repo:", result['code'])
	print("Datasets:", result['dataset'])
	print("Model:", result['model'])
