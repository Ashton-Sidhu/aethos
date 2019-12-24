import os
import subprocess
from pathlib import Path

from aethos.templates.util import _create_project_dir
from jinja2 import Environment, PackageLoader


class TemplateGenerator(object):

    # Prepare environment and source data
    env = Environment(
        loader=PackageLoader("aethos", "templates"),
        trim_blocks=True,
        lstrip_blocks=True,
    )

    project_dir = os.path.join(os.path.expanduser("~"), ".aethos", "projects")

    @classmethod
    def generate_service(cls, name: str, filename: str):
        """
        Generates the necessary files to run your model as a service.

        Generates the app.py, Dockerfile and requirements.txt file.
        
        Parameters
        ----------
        name : str
            Project name

        filename : str
            Model file name
        """

        _create_project_dir(cls.project_dir, name=name)

        files = ["main.py", "Dockerfile", "requirements.txt"]

        for file in files:
            script = cls.env.get_template("files/" + file.replace(".py", "")).render(
                name=name, filename=filename, service=True,
            )

            if file.endswith(".py") or file.endswith(".txt"):
                with open(
                    os.path.join(cls.project_dir, name, "app", file),
                    "w",
                    encoding="utf8",
                ) as f:
                    f.write(script)
            else:
                with open(
                    os.path.join(cls.project_dir, name, file), "w", encoding="utf8"
                ) as f:
                    f.write(script)

        print(f"Deployment files can be found at {cls.project_dir}/{name}.")
