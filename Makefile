
.PHONY: db-init db-start db-stop db-restart db-status db-logs db-shell db-down db-migrate \
        minio-start minio-stop minio-restart minio-status minio-logs minio-down minio-console

# ==============================
# PostgreSQL
# ==============================

# First-time database initialization
db-init:
	docker compose up -d postgres
	@echo "Waiting for PostgreSQL to become healthy..."
	@until docker compose exec -T postgres pg_isready -U documind_user -d documind >/dev/null 2>&1; do sleep 2; done
	@echo "PostgreSQL is ready!"
	$(MAKE) db-migrate

# Start database
db-start:
	docker compose up -d postgres

# Stop database (preserve data)
db-stop:
	docker compose stop postgres

# Restart database (preserve data)
db-restart:
	docker compose restart postgres

# Check database status
db-status:
	docker compose ps postgres

# View database logs
db-logs:
	docker compose logs -f postgres

# Open PostgreSQL shell
db-shell:
	docker compose exec postgres psql -U abdh -d documind

# Apply pending SQL migrations
db-migrate:
	python database/migrations.py

# Stop and remove database container (preserve named volumes)
db-down:
	docker compose stop postgres


# ==============================
# MinIO
# ==============================

# Start MinIO
minio-start:
	docker compose up -d minio

# Stop MinIO (preserve data)
minio-stop:
	docker compose stop minio

# Restart MinIO
minio-restart:
	docker compose restart minio

# Check MinIO status
minio-status:
	docker compose ps minio

# View MinIO logs
minio-logs:
	docker compose logs -f minio

# Open MinIO console
minio-console:
	@echo "Opening MinIO Console: http://localhost:9001"
	@xdg-open http://localhost:9001 2>/dev/null || echo "Open http://localhost:9001 manually"

# Stop and remove MinIO container (preserve named volume)
minio-down:
	docker compose stop minio
