#!/bin/bash
# Setup PostgreSQL for MLflow backend

set -e

echo "Setting up PostgreSQL for MLflow..."

# PostgreSQL configuration
DB_NAME="mlflow_db"
DB_USER="mlflow_user"
DB_PASSWORD="mlflow_password"
DB_HOST="localhost"
DB_PORT="5432"

# Check if PostgreSQL is running
if ! command -v psql &> /dev/null; then
    echo "PostgreSQL is not installed. Please install PostgreSQL first."
    echo "For Ubuntu/Debian: sudo apt-get install postgresql postgresql-contrib"
    echo "For macOS: brew install postgresql"
    exit 1
fi

# Start PostgreSQL service if not running
if ! pg_isready -h $DB_HOST -p $DB_PORT &> /dev/null; then
    echo "Starting PostgreSQL service..."
    if command -v systemctl &> /dev/null; then
        sudo systemctl start postgresql
    elif command -v brew &> /dev/null; then
        brew services start postgresql
    else
        echo "Please start PostgreSQL manually"
        exit 1
    fi
fi

# Wait for PostgreSQL to be ready
echo "Waiting for PostgreSQL to be ready..."
for i in {1..30}; do
    if pg_isready -h $DB_HOST -p $DB_PORT &> /dev/null; then
        echo "PostgreSQL is ready!"
        break
    fi
    sleep 1
    if [ $i -eq 30 ]; then
        echo "PostgreSQL failed to start"
        exit 1
    fi
done

# Create database user and database
echo "Creating MLflow database and user..."

# Create user (ignore error if user exists)
sudo -u postgres psql -c "CREATE USER $DB_USER WITH PASSWORD '$DB_PASSWORD';" 2>/dev/null || echo "User $DB_USER already exists"

# Create database (ignore error if database exists)
sudo -u postgres psql -c "CREATE DATABASE $DB_NAME OWNER $DB_USER;" 2>/dev/null || echo "Database $DB_NAME already exists"

# Grant privileges
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_USER;"

# Test connection
echo "Testing database connection..."
PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "SELECT version();" > /dev/null

if [ $? -eq 0 ]; then
    echo "PostgreSQL setup complete!"
    echo "Connection string: postgresql://$DB_USER:$DB_PASSWORD@$DB_HOST:$DB_PORT/$DB_NAME"
else
    echo "Failed to connect to database"
    exit 1
fi

# Create .env file with database credentials
echo "Creating .env file..."
cat > .env << EOF
# MLflow PostgreSQL Configuration
MLFLOW_BACKEND_STORE_URI=postgresql://$DB_USER:$DB_PASSWORD@$DB_HOST:$DB_PORT/$DB_NAME
MLFLOW_ARTIFACT_ROOT=file:./mlruns
MLFLOW_TRACKING_URI=postgresql://$DB_USER:$DB_PASSWORD@$DB_HOST:$DB_PORT/$DB_NAME

# Database credentials
DB_NAME=$DB_NAME
DB_USER=$DB_USER
DB_PASSWORD=$DB_PASSWORD
DB_HOST=$DB_HOST
DB_PORT=$DB_PORT
EOF

echo ".env file created successfully!"
echo ""
echo "To start MLflow UI, run:"
echo "  mlflow server --backend-store-uri postgresql://$DB_USER:$DB_PASSWORD@$DB_HOST:$DB_PORT/$DB_NAME --default-artifact-root file:./mlruns --host 0.0.0.0 --port 5000"
