import pathlib


def get_file_paths_pathlib_path(content_root_directory: str) -> list[pathlib.Path]:
    print(f"\nGetting list of all PDF files within {content_root_directory}")
    content_root_directory = pathlib.Path(content_root_directory)
    file_paths = []
    for file_path in content_root_directory.glob("*.pdf"):
        file_paths.append(file_path)
    return file_paths


def get_file_paths(content_root_directory: str) -> list[pathlib.Path]:
    print(f"\nGetting list of all PDF files within {content_root_directory}")
    content_root_directory = pathlib.Path(content_root_directory)
    file_paths = []
    for file_path in content_root_directory.glob("*.pdf"):
        file_paths.append(file_path)
    return file_paths
