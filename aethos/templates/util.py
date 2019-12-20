import os


def _create_project_dir(project_dir: str, name: str):

    os.system(f'mkdir -p {project_dir}/{name}/app')
