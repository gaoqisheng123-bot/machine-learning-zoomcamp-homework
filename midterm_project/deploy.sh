set -e


export PROJECT_ID="ml-zoomcamp-midterm"

export SERVICE_NAME="heart-disease-predictor"

export REGION="us-central1"

echo "--- CONFIGURING GCLOUD CLI ---"
gcloud config set project ${PROJECT_ID}

echo "--- ENABLING NECESSARY APIS ---"
gcloud services enable cloudbuild.googleapis.com run.googleapis.com

echo "--- BUILDING CONTAINER IMAGE WITH CLOUD BUILD ---"
gcloud builds submit --tag gcr.io/${PROJECT_ID}/${SERVICE_NAME}

echo "--- DEPLOYING TO CLOUD RUN ---"
gcloud run deploy ${SERVICE_NAME} \
  --image gcr.io/${PROJECT_ID}/${SERVICE_NAME} \
  --platform managed \
  --region ${REGION} \
  --port 9696 \
  --allow-unauthenticated

echo "--- DEPLOYMENT COMPLETE ---"
