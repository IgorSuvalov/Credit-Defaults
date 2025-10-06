# End-to-End Credit Risk Demo for TrustBank 


## Business Goal

Trust Bank wants to roll out a loan campaign to target new customers. The goal is to create a simple-to-fill form that would either approve the loan for a customer or not.

We need to:
- Create a simple web form for customers to fill out.
- Create a machine learning model to predict whether a customer should be approved for a loan or not.
- Create a backend to serve the model and handle form submissions.
## System Architecture

![A diagram showing the flow of data from the React frontend, through the FastAPI backend, to the MLflow server and storage.](./images/Diagram.drawio.svg)

## Tech Stack

- **Frontend**: 
  - React (Vite)
  - Axios (HTTP client)
  - Nginx (static hosting)
- **Backend**: 
  - FastAPI (API)
  - Uvicorn (server)
  - Pydantic (data validation)
- **Machine Learning**:
  - MLflow (model tracking and deployment)
  - Scikit-learn (data preprocessing and model evaluation)
  - XGBoost (model training)
  - PyTorch (deep learning models)
  - Pandas (data manipulation)
- **Containerization / Orchestration**:
  - Docker and Docker Compose
  - Built-in healthchecks
- **Testing**:
  - Pytest (unit and integration tests)
  - FastAPI TestClient (for API endpoint testing)
 
## Video Demo

https://github.com/user-attachments/assets/3f5448bc-9e4e-47c5-bfef-030c1b279e0f

## Quickstart

```bash
# 1) Clone the repository
git clone https://github.com/IgorSuvalov/Credit-Defaults/

# 2) Run the notebook to train and log the model in MLflow (uncomment PyTorch in requirements.txt first)
Jupyter Notebook Notebooks/Notebook.ipynb
http://localhost:5000 #to check the model run ID

# 3) Build and run everything with Docker Compose
docker-compose up --build

# 4) Access the apps 
Frontend: http://localhost:5173
Backend: http://localhost:8004
MLflow: http://localhost:5000



```

## Testing
```bash
# To run tests
pip install -r requirements.txt
pytest -q
```

## Project Structure
```angular2html
Directory structure:
└── igorsuvalov-credit-defaults/
    ├── README.md
    ├── docker-compose.yml
    ├── requirements.txt
    ├── backend/
    │   ├── __init__.py
    │   ├── Dockerfile
    │   ├── service.py
    │   └── .dockerignore
    ├── frontend/
    │   ├── Dockerfile
    │   ├── index.html
    │   ├── nginx.conf
    │   ├── .dockerignore
    │   └── src/
    │       ├── api.js
    │       ├── App.css
    │       ├── App.jsx
    │       ├── index.css
    │       ├── main.jsx
    │       └── components/
    │           └── EnterDetailsForm.jsx
    ├── Notebooks/
    │   └── Notebook.ipynb
    └── tests/
        ├── conftest.py
        ├── test_endpoint.py
        └── test_validation.py
```
