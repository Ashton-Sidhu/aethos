import os


def _create_project_dir(project_dir: str, name: str):
    """
    Creates the projects directory.
    
    Parameters
    ----------
    project_dir : str
        Full path of the project dir.

    name : str
        Name of the project
    """

    os.system(f"mkdir -p {project_dir}/{name}/app")
