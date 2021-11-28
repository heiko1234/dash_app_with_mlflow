"""Main application file"""

from app.embeded_app import app

if __name__ == "__main__":
    app.run_server(port=8050, debug=True)
