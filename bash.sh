pip install mkdocs
mkdocs new .
mv docs/* into mkdocs-project/docs/
mkdocs serve       # local
mkdocs gh-deploy   # pushes to GitHub Pages

docker compose up --build
# → open http://localhost:8501

pip install streamlit pandas scikit-learn joblib
