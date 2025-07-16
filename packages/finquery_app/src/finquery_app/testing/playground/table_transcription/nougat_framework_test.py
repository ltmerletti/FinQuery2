import subprocess
import os

pdf_path = "../../test_docs/aapl-20230930.pdf"

output_dir = "nougat_output"

try:
    os.makedirs(output_dir, exist_ok=True)

    command = [
        "nougat",
        pdf_path,
        "-o",
        output_dir
    ]

    print("Running...")
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    print("Nougat processing complete.")

    pdf_basename = os.path.splitext(os.path.basename(pdf_path))[0]
    output_files = [f for f in os.listdir(output_dir) if f.endswith('.mmd') and pdf_basename in f]

    if output_files:
        output_files.sort(key=lambda f: os.path.getmtime(os.path.join(output_dir, f)), reverse=True)
        latest_file = output_files[0]
        output_filepath = os.path.join(output_dir, latest_file)

        print(f"\n--- Raw Markdown Output from Nougat ({output_filepath}) ---")
        with open(output_filepath, 'r') as f:
            markdown_content = f.read()
            if "[MISSING_PAGE_EMPTY" in markdown_content:
                print("\nError: Nougat reported the page was empty or could not be rendered.")
            else:
                print(markdown_content)
    else:
        print(f"Error: No Nougat output file (.mmd) found in '{output_dir}'.")
        if result.stderr:
            print("STDERR from Nougat process:", result.stderr)
finally:
    print("Script finished.")
