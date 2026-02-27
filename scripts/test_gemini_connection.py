
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from wordtolatex.gemini_llm import _init_gemini

def test_connection():
    print("Initializing Gemini...")
    try:
        model = _init_gemini()
        print("Model initialized.")
        response = model.generate_content("Hello, simply reply with 'Connected'.")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_connection()
