#!/bin/bash
# Run All - Complete pipeline execution
# Sanket-Svasthya Sign Language Recognition

echo "========================================"
echo "  Sanket-Svasthya Pipeline Runner"
echo "========================================"

# Check if data exists
if [ ! -d "data/processed" ]; then
    echo "Step 1: Preprocessing data..."
    python preprocessing/preprocess.py --input data/raw --output data/processed
else
    echo "Step 1: Skipped (processed data exists)"
fi

# Train model if not exists
if [ ! -f "checkpoints/best_model.h5" ]; then
    echo "Step 2: Training model..."
    python training/train.py --config training/config.yaml
else
    echo "Step 2: Skipped (model exists)"
fi

# Evaluate model
echo "Step 3: Evaluating model..."
python training/evaluate.py --model checkpoints/best_model.h5 --data data/processed

# Run tests
echo "Step 4: Running tests..."
pytest tests/ -v

echo ""
echo "========================================"
echo "  Pipeline Complete!"
echo "  Run demo: streamlit run ui/app.py"
echo "========================================"
