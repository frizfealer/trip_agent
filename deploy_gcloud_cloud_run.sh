#!/usr/bin/env bash
# Set options for increased script robustness:
# -e: Exit immediately if a command exits with a non-zero status.
# -u: Treat unset variables as an error when substituting.
# -o pipefail: The return value of a pipeline is the status of the last command
#               to exit with a non-zero status, or zero if all commands exit
#               successfully.
set -euo pipefail

# ==========================================================
#  Cloud Run – end‑to‑end bootstrap for your Gradio + FastAPI
#  container (mounting .env secret) — amd64‑compatible version
# ==========================================================
#  This script automates the deployment process for a Dockerized
#  application (like Gradio/FastAPI) to Google Cloud Run.
#  It handles:
#  • Building and pushing a Docker image optimized for Cloud Run (linux/amd64)
#    using docker buildx.
#  • Reading your local .env file content.
#  • Storing the .env content as a single secret in Google Secret Manager.
#  • Creating necessary IAM bindings for the Cloud Run service account
#    to access the secret.
#  • Creating or updating the Cloud Run service.
#  • Configuring Cloud Run to mount the .env secret as a file inside the container.
# ==========================================================

# ---------- USER VARIABLES --------------------------------
# Configure these variables according to your Google Cloud project and application.

PROJECT_ID="striped-bonfire-257419"   # Your Google Cloud project ID.
REGION="us-east1"                     # The Google Cloud region where you want to deploy Cloud Run and Artifact Registry.
REPO="trip-agent"                     # The name for your Artifact Registry Docker repository.
IMAGE_NAME="app"                      # The name for your Docker image within the repository.
TAG="v1"                              # The tag for your Docker image (e.g., latest, v1, commit SHA).
LOCAL_DOCKERFILE_DIR="."             # The local directory containing your Dockerfile.

# Local .env file path
DOTENV_PATH="./.env" # Path to your local .env file containing environment variables and secrets.

# Secret Manager Name for the entire .env file content
SECRET_NAME="dotenv-file" # The name for the secret in Google Secret Manager that will store your .env content.

# Path where the secret will be mounted as a file inside the container
ENV_MOUNT_PATH="/app/.env" # The absolute path inside the container where the .env file will be mounted.

SERVICE_NAME="trip-agent"             # The name for your Cloud Run service.
SA_NAME="run-exec-sa"                 # The name for the dedicated service account for Cloud Run execution.
PORT="8080"                           # The container port that Cloud Run will send traffic to.
                                      # Cloud Run injects the PORT env var, typically 8080.
                                      # Ensure your application (specifically the frontend) listens on this port.
# -----------------------------------------------------------

# Check if .env file exists
# This is crucial as the script relies on reading this file.
if [ ! -f "$DOTENV_PATH" ]; then
    echo "Error: .env file not found at $DOTENV_PATH"
    echo "Please create a .env file with your environment variables and secrets."
    exit 1 # Exit the script if the .env file is missing.
fi

echo "📖 Using .env file at $DOTENV_PATH"

# 1. Authenticate and set the Google Cloud project -------------------------
echo "🔑 Authenticating and setting project..."
# Authenticates the user (interactive login may be required the first time).
gcloud auth login
# Sets the default project for subsequent gcloud commands.
gcloud config set project "$PROJECT_ID"

# 2. Enable required Google Cloud APIs -----------------------------------
echo "🚀 Enabling required APIs..."
# Ensures the necessary APIs are enabled for the project.
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  secretmanager.googleapis.com

# 3. Create Artifact Registry Docker repository if missing --------------------------
echo "📦 Checking/creating Docker repository..."
# Checks if the repository exists by attempting to describe it.
# Redirects stdout and stderr to /dev/null to keep the output clean.
if ! gcloud artifacts repositories describe "$REPO" --location="$REGION" >/dev/null 2>&2; then
  # If describe fails (repository not found), create it.
  gcloud artifacts repositories create "$REPO" \
    --repository-format=docker \
    --location="$REGION" \
    --description="Docker repo for $SERVICE_NAME"
  echo "✨ Created repository: $REPO in $REGION"
else
  echo "🔄 Repository '$REPO' already exists."
fi

# 4. Configure Docker to use gcloud as a credential helper ---------------------
echo "🐳 Configuring Docker auth helper..."
# This allows Docker to push/pull images to/from Artifact Registry without manual login.
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

# 5. Build and push linux/amd64 image via buildx ------------------------------
REMOTE_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE_NAME}:${TAG}"

echo "🐳 Checking/using buildx builder 'cloudrun_cross'..."
# This section ensures a buildx builder capable of building linux/amd64 is available.
# It first tries to use an existing builder named 'cloudrun_cross'.
# If that fails (e.g., builder doesn't exist or is in a bad state), it creates a new one.
if ! docker buildx use cloudrun_cross >/dev/null 2>&1; then
  echo "✨  Creating buildx builder 'cloudrun_cross'"
  docker buildx create --name cloudrun_cross --use
else
   echo "🔄  Re‑using existing buildx builder 'cloudrun_cross'"
fi

echo -e "\n🏗  Building linux/amd64 image and pushing to $REMOTE_IMAGE…"
# Builds the Docker image for the specified platform and pushes it to Artifact Registry.
docker buildx build \
  --platform linux/amd64 \
  -t "$REMOTE_IMAGE" \
  --push "$LOCAL_DOCKERFILE_DIR"

# 6. Create/Update the .env secret in Secret Manager ------------------------
echo -e "\n🔒 Checking/creating/updating secret '$SECRET_NAME' in Secret Manager..."
# Checks if the secret exists.
# Using a subshell `(...)` ensures that the `set -e` option doesn't cause the script
# to exit if `gcloud secrets describe` fails because the secret doesn't exist.
if (gcloud secrets describe "$SECRET_NAME" >/dev/null 2>&2); then
  # If secret exists, add a new version with the current .env file content.
  # This is how you update the secret value.
  gcloud secrets versions add "$SECRET_NAME" --data-file="$DOTENV_PATH"
  echo "🔄 Added a new version to secret '$SECRET_NAME' with content from $DOTENV_PATH."
else
  # If secret does not exist, create it.
  # --data-file=- reads the secret content from standard input.
  # --replication-policy=automatic ensures the secret is replicated globally.
  gcloud secrets create "$SECRET_NAME" --data-file="$DOTENV_PATH" --replication-policy=automatic
  echo "✨ Created secret: $SECRET_NAME with content from $DOTENV_PATH."
fi


# 7. Create execution Service Account if absent --------------------------
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
echo -e "\n👤 Checking/creating execution service account..."
# Checks if the service account exists.
if ! gcloud iam service-accounts describe "$SA_EMAIL" >/dev/null 2>&2; then
  # If service account not found, create it.
  gcloud iam service-accounts create "$SA_NAME" --display-name="Cloud Run execution SA"
  echo "✨ Created service account: $SA_EMAIL"
else
  echo "🔄 Service account '$SA_EMAIL' already exists."
fi

# 8. Grant the Service Account permission to access the secret --------------------
echo "🔐 Granting secret access to service account '$SA_EMAIL' for secret '$SECRET_NAME'..."
# Binds the 'Secret Manager Secret Accessor' role to the service account for the specific secret.
# This is the principle of least privilege.
gcloud secrets add-iam-policy-binding "$SECRET_NAME" \
  --member="serviceAccount:$SA_EMAIL" \
  --role="roles/secretmanager.secretAccessor" --quiet
echo "🔑 Granted access to $SECRET_NAME"


# 9. Deploy or update the Cloud Run service ------------------------------
echo -e "\n🚀 Deploying/updating Cloud Run service '$SERVICE_NAME'..."
# Deploys the service. If it exists, it creates a new revision.
gcloud run deploy "$SERVICE_NAME" \
  --region "$REGION" \
  --image "$REMOTE_IMAGE" \
  --service-account "$SA_EMAIL" \
  --port "$PORT" # Specifies the container port Cloud Run should send traffic to.
  # Use one of the following flags to configure authentication:
  # --allow-unauthenticated # Allows public access to the service.
  # --no-allow-unauthenticated # Requires authentication (e.g., via Identity Platform or IAP).
  \
  # Mounts the secret as a file inside the container.
  # Format: CONTAINER_PATH=SECRET_NAME:SECRET_VERSION
  --set-secrets "${ENV_MOUNT_PATH}=${SECRET_NAME}:latest" \
  --quiet # Suppresses interactive prompts during deployment.

# 10. Output the deployed service URL -------------------------------------------
echo -e "\n✅ Deployment complete."
echo "🌐 Service URL:"
# Describes the service and extracts the URL using format.
gcloud run services describe "$SERVICE_NAME" --region="$REGION" --format='value(status.url)'

# --------------------  END  --------------------------------
