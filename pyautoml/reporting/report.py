import os
import platform
import sys
from datetime import datetime


class Report():

    def __init__(self, report_name):
        self.report_name = report_name

        if os.path.exists(self.report_name):
            self.filename = self.report_name
        else:
            self.filename = 'pyautoml_reports/{}{}.txt'.format(report_name, datetime.now().strftime("%d-%m-%Y_%I-%M-%S%p"))
            self.report_environtment()

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

            write = False; 

            with open(self.filename, 'r') as f:
                if header not in f.read():
                    write = True
            
            if write:
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

        write = True; 

        if os.path.exists(self.filename):
            write = False
            
            with open(self.filename, 'r') as f:
                if content not in f.read():
                    write = True

        if write:
            with open(self.filename, 'a+') as f:
                f.writelines(content)

    def report_technique(self, technique: str, list_of_cols=[]):
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

        self.write_contents(log + "\n")

    def report_gridsearch(self, clf, verbose):

        content = "Tuning hyper-parameters:\n\n \
            Best parameters set found on training set:\n \
                {}\n\n \
            Grid scores on training set:\n\n".format(
            str(clf.best_params_))

        self.write_contents("\n".join(map(str.lstrip, content.split('\n'))))
        
        if verbose:
            print(content)

        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            stats = '{:.2%} (+/-{}) for {}\n'.format(float(mean), std*2, params)
            self.write_contents(stats)
            if verbose:
                print(stats)
            
            self.write_contents('\n')

    def report_classification_report(self, report):

        self.write_contents("Detailed classification report:\n")
        self.write_contents("The model is trained on the full training set.")
        self.write_contents("The scores are computed on the full test set.\n\n")
        self.write_contents(report)

    def report_environtment(self):
        """
        Reports local environment info.
        """
        
        system = platform.system()

        self.write_header('Environment')

        self.log('OS Version          : {}'.format(platform.platform()))
        self.log('System              : {}'.format(system))
        self.log('Machine             : {}'.format(platform.machine()))

        if system == 'Windows':
            memory = os.popen('wmic memorychip get capacity').read()
            total_mem = 0
            for m in memory.split("  \r\n")[1:-1]:
                total_mem += int(m)

            self.log('Processor           : {}'.format(platform.processor()))
            self.log('Memory              : {:.2f} GB'.format(round(total_mem / (1024**2))))
        elif platform.system() == 'Darwin':
            command = '/usr/sbin/sysctl -n machdep.cpu.brand_string'
            self.log('Processor           : {}'.format(os.popen(command).read().strip()))
            self.log('Memory              : {}'.format(round(int(os.popen('sysctl -n hw.memsize').read().strip()) / (1024**2))))
        elif platform.system() == 'Linux':
            command = "cat /proc/cpuinfo | grep 'model name'"
            with open("/proc/meminfo", "r") as f:
                lines = f.readlines()
            
            memory = [int(s) for s in lines[0].split() if s.isdigit()][0]
            processor = os.popen(command).read().replace('\t', '').split('model name:')[1].strip()
            self.log('Processor           : {}'.format(processor))
            self.log('Memory              : {:.2f} GB'.format(round(memory / (1024 ** 2))))

        self.log('CPU Count           : {}'.format(os.cpu_count()))
        
        self.log('Python Version      : {}'.format(platform.python_version()))
        self.log('Python Compiler     : {}'.format(platform.python_compiler()))
        self.log('Python Build        : {}'.format(platform.python_build()))
