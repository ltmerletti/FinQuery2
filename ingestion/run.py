import time
import sys
import pathlib

# --- Path Adjustments for Standalone Execution ---
# This ensures the script can find the 'ingestion' module when run directly.
project_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Import the core logic from our new pipeline module
from ingestion.pipeline import run_ingestion_process

def main():
    """
    Main entry point for the standalone ingestion script.
    This function now delegates the heavy lifting to the reusable
    ingestion pipeline function.
    """
    print("--- Starting standalone ingestion process ---")
    run_ingestion_process()
    print("\n--- Standalone ingestion process complete ---")


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
