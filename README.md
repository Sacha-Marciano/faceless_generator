To run in dev :
create venv :
python -m venv venv
build env :
source venv/Scripts/activate
then:
uvicorn main:app --reload

install requirements:
pip install -r requirements.txt
freeze requirements:
pip freeze > requirements.txt

To build :
gcloud builds submit --tag gcr.io/drive-viewer-app-460607/fastapi-gemini

To deploy on cloud run :
gcloud run deploy fastapi-gemini --image gcr.io/drive-viewer-app-460607/fastapi-gemini --platform managed --region us-central1 --allow-unauthenticated --port 8080
