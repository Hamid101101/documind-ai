
#!/usr/bin/env bash

set -Eeuo pipefail

# ============================================================
# DocuMind AI - MinIO Initialization
# ============================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${PROJECT_ROOT}/.env"
POLICY_FILE="${PROJECT_ROOT}/infra/minio/policies/documind-storage-policy.json"

POLICY_NAME="documind-storage-policy"
ADMIN_ALIAS="documind-admin"
APP_ALIAS="documind-app"

# ------------------------------------------------------------
# 1. Validate prerequisites
# ------------------------------------------------------------

echo "=========================================="
echo " DocuMind AI - MinIO Initialization"
echo "=========================================="

if ! command -v mc >/dev/null 2>&1; then
    echo "ERROR: MinIO Client (mc) is not installed."
    echo "Install mc and ensure it is available in PATH."
    exit 1
fi

if [[ ! -f "$ENV_FILE" ]]; then
    echo "ERROR: .env file not found."
    exit 1
fi

if [[ ! -f "$POLICY_FILE" ]]; then
    echo "ERROR: MinIO policy file not found: $POLICY_FILE"
    exit 1
fi

# ------------------------------------------------------------
# 2. Load environment variables
# ------------------------------------------------------------

set -a
source "$ENV_FILE"
set +a

: "${MINIO_ROOT_USER:?MINIO_ROOT_USER is required}"
: "${MINIO_ROOT_PASSWORD:?MINIO_ROOT_PASSWORD is required}"
: "${MINIO_ACCESS_KEY:?MINIO_ACCESS_KEY is required}"
: "${MINIO_SECRET_KEY:?MINIO_SECRET_KEY is required}"
: "${MINIO_BUCKET:?MINIO_BUCKET is required}"

MINIO_ENDPOINT="${MINIO_ENDPOINT:-localhost:9000}"
MINIO_SECURE="${MINIO_SECURE:-false}"

if [[ "$MINIO_SECURE" == "true" ]]; then
    SCHEME="https"
else
    SCHEME="http"
fi

ENDPOINT="${SCHEME}://${MINIO_ENDPOINT}"

# ------------------------------------------------------------
# 3. Start MinIO
# ------------------------------------------------------------

echo "[1/7] Starting MinIO..."

cd "$PROJECT_ROOT"

docker compose up -d minio

# ------------------------------------------------------------
# 4. Configure administrative connection
# ------------------------------------------------------------

echo "[2/7] Configuring MinIO administrative connection..."

mc alias set "$ADMIN_ALIAS" \
    "$ENDPOINT" \
    "$MINIO_ROOT_USER" \
    "$MINIO_ROOT_PASSWORD" \
    --api S3v4 >/dev/null

echo "Waiting for MinIO to become available..."

READY=false

for attempt in $(seq 1 30); do
    if mc ls "$ADMIN_ALIAS" >/dev/null 2>&1; then
        READY=true
        break
    fi

    sleep 2
done

if [[ "$READY" != "true" ]]; then
    echo "ERROR: MinIO did not become available within 60 seconds."
    docker compose logs --tail=50 minio
    exit 1
fi

echo "MinIO is ready."

# ------------------------------------------------------------
# 5. Create the document bucket
# ------------------------------------------------------------

echo "[3/7] Creating document bucket..."

mc mb --ignore-existing \
    "${ADMIN_ALIAS}/${MINIO_BUCKET}"

# Ensure anonymous access is disabled.

mc anonymous set private \
    "${ADMIN_ALIAS}/${MINIO_BUCKET}"

echo "Bucket '${MINIO_BUCKET}' is ready and private."

# ------------------------------------------------------------
# 6. Create application policy
# ------------------------------------------------------------

echo "[4/7] Configuring application policy..."

if mc admin policy info "$ADMIN_ALIAS" "$POLICY_NAME" \
    >/dev/null 2>&1; then

    echo "Policy '${POLICY_NAME}' already exists."

else

    mc admin policy create \
        "$ADMIN_ALIAS" \
        "$POLICY_NAME" \
        "$POLICY_FILE"

    echo "Policy '${POLICY_NAME}' created."

fi

# ------------------------------------------------------------
# 7. Create application user and attach policy
# ------------------------------------------------------------

echo "[5/7] Configuring application user..."

if mc admin user info "$ADMIN_ALIAS" "$MINIO_ACCESS_KEY" \
    >/dev/null 2>&1; then

    echo "Application user '${MINIO_ACCESS_KEY}' already exists."

else

    mc admin user add \
        "$ADMIN_ALIAS" \
        "$MINIO_ACCESS_KEY" \
        "$MINIO_SECRET_KEY"

    echo "Application user created."

fi

echo "Attaching application policy..."

mc admin policy attach \
    "$ADMIN_ALIAS" \
    "$POLICY_NAME" \
    --user "$MINIO_ACCESS_KEY"

# ------------------------------------------------------------
# 8. Verify application access
# ------------------------------------------------------------

echo "[6/7] Verifying application credentials..."

mc alias set "$APP_ALIAS" \
    "$ENDPOINT" \
    "$MINIO_ACCESS_KEY" \
    "$MINIO_SECRET_KEY" \
    --api S3v4 >/dev/null

mc ls "${APP_ALIAS}/${MINIO_BUCKET}" >/dev/null

echo "Application credentials successfully verified."

# ------------------------------------------------------------
# 9. Final status
# ------------------------------------------------------------

echo "[7/7] MinIO initialization completed."

echo "=========================================="
echo " MinIO is ready!"
echo "=========================================="
echo "Bucket: ${MINIO_BUCKET}"
echo "Application user: ${MINIO_ACCESS_KEY}"
echo "Endpoint: ${ENDPOINT}"
echo "Console: http://localhost:9001"
echo "=========================================="
