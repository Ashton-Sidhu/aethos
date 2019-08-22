import os


class Report():

    def __init__(self, report_name):
        self.report_name = report_name
        self.filename = f"pyautoml_reports/{report_name}.txt"

        #TODO: Move making the directory on first time run to some config file
        if not os.path.exists("pyautoml_reports/"):
            os.makedirs("pyautoml_reports")        

    def write_header(self, header: str):
        """
        Writes a header to a report file
        
        Parameters
        ----------
        header : str
            Header for a report file
            Examples: 'Cleaning', 'Feature Engineering', 'Modelling', etc.
        """
        
        if os.path.exists(self.filename):
            with open(self.filename, 'a+') as f:
                f.write("\n" + header + "\n\n")
        
        else:
            with open(self.filename, 'a+') as f:
                f.write(header + "\n\n")

    def write_contents(self, content: str):
        """
        Write report contents to report.
        
        Parameters
        ----------
        content : str
            Report Content
        """

        with open(self.filename, 'a+') as f:
            f.write(content)

    def report_technique(self, technique: str, list_of_cols: list):
        """
        IN ALPHA V1
        ============

        Writes analytic technique info to the report detailing what analysis was run
        and why it was ran.
        
        Parameters
        ----------
        technique : str
            Analytic technique
        list_of_cols : list
            List of columns
        """

        if list(list_of_cols) != []:        
            for col in list_of_cols:
                log = technique.replace("column_name_placeholder", col)

                self.write_contents(log)

        else:
            self.write_contents(technique)

    def log(self, log: str):
        """
        IN ALPHA V1
        ============

        Logs info to the report file.
        
        Parameters
        ----------
        log : str
            Information to write to the report file
        """

        self.write_contents(log)
