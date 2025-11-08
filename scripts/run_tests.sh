#!/bin/bash
# run_tests.sh
# Script to run all tests with various options

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}  MLOps Phase 3 - Test Runner  ${NC}"
echo -e "${GREEN}================================${NC}"
echo ""

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}Error: pytest not found!${NC}"
    echo "Please install testing dependencies:"
    echo "  pip install -r requirements-dev.txt"
    exit 1
fi

# Function to run tests with specific options
run_tests() {
    local test_type=$1
    local pytest_args=$2
    
    echo -e "${YELLOW}Running ${test_type}...${NC}"
    
    if pytest $pytest_args; then
        echo -e "${GREEN}✓ ${test_type} passed!${NC}"
        echo ""
        return 0
    else
        echo -e "${RED}✗ ${test_type} failed!${NC}"
        echo ""
        return 1
    fi
}

# Parse command line arguments
case "${1:-all}" in
    all)
        echo "Running all tests..."
        run_tests "All Tests" "-v"
        ;;
    
    unit)
        echo "Running unit tests only..."
        run_tests "Unit Tests" "-v -m unit"
        ;;
    
    integration)
        echo "Running integration tests only..."
        run_tests "Integration Tests" "-v -m integration"
        ;;
    
    fast)
        echo "Running fast tests only (excluding slow tests)..."
        run_tests "Fast Tests" "-v -m 'not slow'"
        ;;
    
    coverage)
        echo "Running tests with coverage report..."
        run_tests "Coverage Tests" "--cov=src --cov-report=html --cov-report=term"
        echo -e "${GREEN}Coverage report generated in htmlcov/index.html${NC}"
        ;;
    
    api)
        echo "Running API tests only..."
        run_tests "API Tests" "-v -m api"
        ;;
    
    parallel)
        echo "Running tests in parallel..."
        run_tests "Parallel Tests" "-n auto -v"
        ;;
    
    verbose)
        echo "Running tests with maximum verbosity..."
        run_tests "Verbose Tests" "-vv -s"
        ;;
    
    quick)
        echo "Running quick check (fast tests with coverage)..."
        run_tests "Quick Check" "-v -m 'not slow' --cov=src --cov-report=term-missing"
        ;;
    
    help|--help|-h)
        echo "Usage: ./run_tests.sh [option]"
        echo ""
        echo "Options:"
        echo "  all          Run all tests (default)"
        echo "  unit         Run only unit tests"
        echo "  integration  Run only integration tests"
        echo "  fast         Run fast tests only (exclude slow)"
        echo "  coverage     Run tests with coverage report"
        echo "  api          Run only API tests"
        echo "  parallel     Run tests in parallel"
        echo "  verbose      Run with maximum verbosity"
        echo "  quick        Quick check (fast tests + coverage)"
        echo "  help         Show this help message"
        echo ""
        echo "Examples:"
        echo "  ./run_tests.sh              # Run all tests"
        echo "  ./run_tests.sh unit         # Run unit tests"
        echo "  ./run_tests.sh coverage     # Run with coverage"
        echo "  ./run_tests.sh quick        # Quick check before commit"
        ;;
    
    *)
        echo -e "${RED}Unknown option: $1${NC}"
        echo "Use './run_tests.sh help' for usage information"
        exit 1
        ;;
esac

echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}  Testing Complete!  ${NC}"
echo -e "${GREEN}================================${NC}"