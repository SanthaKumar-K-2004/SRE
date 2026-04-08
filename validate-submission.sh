#!/bin/bash
# SRE-Bench Pre-Submission Validation Script
# Run this before submitting to verify everything works.

set -e

echo "🔍 SRE-Bench Pre-Submission Validation"
echo "======================================="

# 1. Check all required files exist
echo ""
echo "📁 Checking required files..."
REQUIRED_FILES=(
    "environment.py"
    "state_machine.py"
    "models.py"
    "rewards.py"
    "api.py"
    "inference.py"
    "generate_dataset.py"
    "Dockerfile"
    "requirements.txt"
    "README.md"
    "tasks/__init__.py"
    "tasks/task1.py"
    "tasks/task2.py"
    "tasks/task3.py"
    "data/incidents.json"
    "tests/test_models.py"
    "tests/test_state_machine.py"
    "tests/test_graders.py"
    "tests/test_environment.py"
    "tests/test_api.py"
)

MISSING=0
for f in "${REQUIRED_FILES[@]}"; do
    if [ -f "$f" ]; then
        echo "  ✅ $f"
    else
        echo "  ❌ $f MISSING"
        MISSING=$((MISSING + 1))
    fi
done

if [ $MISSING -gt 0 ]; then
    echo "❌ $MISSING required files missing!"
    exit 1
fi
echo "✅ All required files present"

# 2. Check dataset
echo ""
echo "📊 Checking dataset..."
INCIDENT_COUNT=$(./venv/Scripts/python.exe -c "import json; data=json.load(open('data/incidents.json')); print(len(data))" 2>/dev/null || venv/bin/python -c "import json; data=json.load(open('data/incidents.json')); print(len(data))")
echo "  Incidents: $INCIDENT_COUNT"
if [ "$INCIDENT_COUNT" -ne 90 ]; then
    echo "  ❌ Expected 90 incidents, got $INCIDENT_COUNT"
    exit 1
fi
echo "  ✅ Dataset valid (90 incidents)"

# 3. Run tests
echo ""
echo "🧪 Running tests..."
./venv/Scripts/python.exe -m pytest --cov=. tests/ -q 2>/dev/null || venv/bin/python -m pytest --cov=. tests/ -q
echo "✅ All tests passed"

# 4. Check API starts
echo ""
echo "🌐 Testing API startup..."
./venv/Scripts/uvicorn.exe api:app --host 0.0.0.0 --port 7860 & 2>/dev/null || venv/bin/uvicorn api:app --host 0.0.0.0 --port 7860 &
API_PID=$!
sleep 3

# Health check
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:7860/health)
if [ "$HTTP_STATUS" -eq 200 ]; then
    echo "  ✅ /health returns 200"
else
    echo "  ❌ /health returned $HTTP_STATUS"
    kill $API_PID 2>/dev/null
    exit 1
fi

# Reset check
RESET_STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X POST http://localhost:7860/reset \
    -H 'Content-Type: application/json' \
    -d '{"task": "task1", "seed": 42}')
if [ "$RESET_STATUS" -eq 200 ]; then
    echo "  ✅ /reset returns 200"
else
    echo "  ❌ /reset returned $RESET_STATUS"
    kill $API_PID 2>/dev/null
    exit 1
fi

# Step check
STEP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X POST http://localhost:7860/step \
    -H 'Content-Type: application/json' \
    -d '{"action_type": "inspect_logs", "target_service": "api-gateway"}')
if [ "$STEP_STATUS" -eq 200 ]; then
    echo "  ✅ /step returns 200"
else
    echo "  ❌ /step returned $STEP_STATUS"
    kill $API_PID 2>/dev/null
    exit 1
fi

kill $API_PID 2>/dev/null
echo "✅ API endpoints working"

# 5. Summary
echo ""
echo "======================================="
echo "🎉 All validation checks passed!"
echo "  📁 Files:    ${#REQUIRED_FILES[@]}/${#REQUIRED_FILES[@]}"
echo "  📊 Dataset:  90 incidents"
echo "  🧪 Tests:    All passed"
echo "  🌐 API:      All endpoints responding"
echo "======================================="
echo ""
echo "Ready for submission! 🚀"
