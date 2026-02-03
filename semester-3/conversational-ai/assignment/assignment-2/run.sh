#!/bin/bash
# Master script to run the complete Hybrid RAG System pipeline
# Deletes existing data and collects fresh data every run

set -e  # Exit on error
export KMP_DUPLICATE_LIB_OK=TRUE
export TOKENIZERS_PARALLELISM=false

echo "============================================================"
echo "HYBRID RAG SYSTEM - COMPLETE PIPELINE"
echo "============================================================"
echo ""

# Function to print step headers
print_step() {
    echo ""
    echo "------------------------------------------------------------"
    echo "STEP $1: $2"
    echo "------------------------------------------------------------"
}

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    print_step "0" "Creating Virtual Environment"
    python3 -m venv venv
fi

# Activate virtual environment
print_step "0" "Activating Virtual Environment"
source venv/bin/activate

# Install dependencies
print_step "1" "Installing Dependencies"
pip install -q -r requirements.txt
python -c "import nltk; nltk.download('punkt', quiet=True)"

# Clean existing data for fresh collection
print_step "2" "Cleaning Existing Data"
echo "Removing old data files for fresh collection..."
rm -f data/corpus.pkl
rm -f data/chunks.json
rm -f data/faiss_index.bin
rm -f data/faiss_index_metadata.pkl
rm -f data/bm25_index.pkl
rm -f data/questions.json
echo "Cleanup complete!" 

# Generate fixed URLs if not exists
if [ ! -f "data/fixed_urls.json" ]; then
    print_step "3" "Generating Fixed URLs (200 articles)"
    python scripts/generate_fixed_urls.py
else
    echo "Fixed URLs exist. Using existing fixed URLs..."
fi

# Collect Wikipedia data (always fresh)
print_step "4" "Collecting Fresh Wikipedia Data (500 articles)"
echo "This will take 15-20 minutes..."
echo "Collecting 200 fixed + 300 random URLs"
python src/data_collection.py

# Preprocess data
print_step "5" "Preprocessing Text (Chunking)"
python src/preprocessing.py

# Build indices
print_step "6" "Building Retrieval Indices"
python src/hybrid_retrieval.py

echo ""
echo "============================================================"
echo "SETUP COMPLETE!"
echo "============================================================"
echo ""
echo "Choose an option:"
echo "  1. Run Web UI (localhost:5000)"
echo "  2. Run Evaluation Pipeline"
echo "  3. Exit"
echo ""
read -p "Enter choice [1-3]: " choice

case $choice in
    1)
        print_step "7" "Launching Web UI"
        echo "Opening http://localhost:5000"
        cd ui && python app.py
        ;;
    2)
        print_step "7" "Running Evaluation"
        # Generate questions
        echo "Generating evaluation questions..."
        python evaluation/question_generation.py
        # Run evaluation
        python run_evaluation.py
        echo ""
        echo "Evaluation complete! Check reports/evaluation_report.html"
        ;;
    3)
        echo "Exiting..."
        ;;
    *)
        echo "Invalid choice. Exiting..."
        ;;
esac

echo ""
echo "Done!"

