# EP2C Prototype: Paper-to-Code Prototype

This is the **Flask-based frontend** for the **EP2C (Explainable Paper-to-Code)** project. It allows users to upload academic papers (PDF format) and see the generated code side by side, with an explanation layer linking the code to sections of the paper.

## Prototype Structure
```bash
prototype/
├── static/ # CSS, JS, uploaded PDFs
│ ├── css/ # Stylesheets
│ ├── js/ # JavaScript for interactivity
│ └── uploads/ # User-uploaded PDF files
├── templates/ # HTML templates (index.html, viewer.html)
├── app.py # Flask app files (routes, logic)
├── requirements.txt # Dependencies for the frontend app
└── README.md # Project setup guide and information
```

---

## Running the Prototype

### Clone the repository: 
```bash
git clone https://github.com/vupatel08/473-Capstone-Project-EP2C.git
cd 473-Capstone-Project-EP2C/prototype
```

### Set up a virtual environment

#### For macOS/Linux: 
```bash 
python3 -m venv .venv
source .venv/bin/activate
``` 
#### For Windows: 
```bash
python -m venv .venv 
.\.venv\Scripts\activate
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the app: 
```bash
python app.py
```

Visit `http://localhost:5000` to try it out!
