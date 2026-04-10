# AI Smart Site Selection - Streamlit Demo

A demo Streamlit app for an AI-driven clinical trial smart site selection workflow.

## What is included
- 8 workflow screens matching the uploaded wireframe
- mock AI ranking and explainability
- feasibility distribution and response tracking
- site feasibility deep-dive
- qualification workflow with CDA + CRA flags
- final site selection summary with export buttons
- chatbot panel
- CRA notifications center

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes
- This is a frontend-heavy demo with seeded mock data.
- Export to Excel currently downloads CSV for simplicity.
- PDF export is stubbed as a text summary and can be replaced with ReportLab or WeasyPrint later.
- Best next step: split into multipage modules and connect to real APIs / database.
