import os


class Report():

    def __init__(self, report_name):
        self.report_name = report_name
        self.filename = f"reports/{report_name}.txt"

        #TODO: Move making the directory on first time run to some config file
        if not os.path.exists("reports/"):
            os.makedirs("reports")        

    def WriteHeader(self, header):

        with open(self.filename, 'a+') as f:
            f.write(header + "\n")

    def WriteContents(self, content):

        with open(self.filename, 'a+') as f:
            f.write(content + "\n")
