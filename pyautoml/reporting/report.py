import os


class Report():

    def __init__(self, report_name):
        self.report_name = report_name
        self.filename = f"pyautoml_reports/{report_name}.txt"

        #TODO: Move making the directory on first time run to some config file
        if not os.path.exists("pyautoml_reports/"):
            os.makedirs("pyautoml_reports")        

    def WriteHeader(self, header):
        
        if os.path.exists(self.filename):
            with open(self.filename, 'a+') as f:
                f.write("\n" + header + "\n\n")
        
        else:
            with open(self.filename, 'a+') as f:
                f.write(header + "\n\n")

    def WriteContents(self, content):

        with open(self.filename, 'a+') as f:
            f.write(content)

    def ReportTechnique(self, technique, list_of_cols):

        if list(list_of_cols) != []:        
            for col in list_of_cols:
                log = technique.replace("column_name_placeholder", col)

                self.WriteContents(log)

        else:
            self.WriteContents(technique)
