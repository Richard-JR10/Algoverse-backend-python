## Installation

Follow these steps to set up AlgoVerse on your local machine:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Richard-JR10/Algoverse-backend-python.git
   cd Algoverse-backend-python
   ```

2. **Create a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   Assuming the project uses a `requirements.txt` file for dependencies (e.g., FastAPI, Uvicorn, etc.):
   ```bash
   pip install -r requirements.txt
   ```
   *Note*: If the repository doesn't include a `requirements.txt`, typical dependencies for a FastAPI project might include:
   ```bash
   pip install fastapi uvicorn python-dotenv
   ```

4. **Verify Setup**:
   - Check Python version:
     ```bash
     python --version
     ```
   - Ensure all dependencies installed correctly by running:
     ```bash
     python -m pytest
     ```
     *Note*: This assumes the project includes tests in a `tests/` directory. If no tests exist, you can skip this step or check the repository for specific test instructions.

---

## Running AlgoVerse Locally

To run AlgoVerse on your machine:

1. **Start the Development Server**:
   Using Uvicorn (a common ASGI server for FastAPI):
   ```bash
   uvicorn main:app --reload
   ```
   - `main:app` refers to the FastAPI app instance in a file named `main.py`. Replace `main` with the actual filename if different (e.g., `app.py` → `app:app`).
   - The `--reload` flag enables auto-reload for development, similar to `npm run dev`.
   - This will launch the app, typically at `http://localhost:8000` (Uvicorn's default port).

2. **Access AlgoVerse**:
   - Open your browser and navigate to `http://localhost:8000` (or the port specified in the terminal, e.g., if changed via `--port`).
   - For FastAPI, you can also access the interactive API docs at `http://localhost:8000/docs`.
