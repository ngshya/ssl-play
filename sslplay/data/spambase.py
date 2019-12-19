from pandas import read_csv
import numpy as np

class DataSpambase:


    def __init__(self): 
        self.name = "SPAM"


    def load(self, path="data/spambase/spambase.data"):
        self.data = read_csv(path, header=None)
        self.data.columns = [
            "word_freq_make", 
            "word_freq_address", 
            "word_freq_all", 
            "word_freq_3d", 
            "word_freq_our", 
            "word_freq_over", 
            "word_freq_remove", 
            "word_freq_internet", 
            "word_freq_order", 
            "word_freq_mail", 
            "word_freq_receive", 
            "word_freq_will", 
            "word_freq_people", 
            "word_freq_report", 
            "word_freq_addresses", 
            "word_freq_free", 
            "word_freq_business", 
            "word_freq_email", 
            "word_freq_you", 
            "word_freq_credit", 
            "word_freq_your", 
            "word_freq_font", 
            "word_freq_000", 
            "word_freq_money", 
            "word_freq_hp", 
            "word_freq_hpl", 
            "word_freq_george", 
            "word_freq_650", 
            "word_freq_lab", 
            "word_freq_labs", 
            "word_freq_telnet", 
            "word_freq_857", 
            "word_freq_data", 
            "word_freq_415", 
            "word_freq_85", 
            "word_freq_technology", 
            "word_freq_1999", 
            "word_freq_parts", 
            "word_freq_pm", 
            "word_freq_direct", 
            "word_freq_cs", 
            "word_freq_meeting", 
            "word_freq_original", 
            "word_freq_project", 
            "word_freq_re", 
            "word_freq_edu", 
            "word_freq_table", 
            "word_freq_conference", 
            "char_freq_;", 
            "char_freq_(", 
            "char_freq_[", 
            "char_freq_!", 
            "char_freq_$", 
            "char_freq_#", 
            "capital_run_length_average", 
            "capital_run_length_longest", 
            "capital_run_length_total", 
            "target"
        ]

    
    def parse(self):
        pass


    def split(self, port_test=0.2, port_unla=0.0, seed=1102):
        np.random.seed(seed)
        int_size = self.data.shape[0]
        array_sets = np.repeat("L", int_size)
        array_test = np.random.choice([True, False], size=int_size, replace=True, p=[port_test, 1-port_test])
        array_unla = np.random.choice([True, False], size=int_size, replace=True, p=[port_unla, 1-port_unla])
        array_sets[array_test] = "T"
        array_sets[(~array_test) & array_unla] = "U"
        return {
            "train_l": self.data.loc[array_sets == "L", self.data.columns != "target"], 
            "target_train_l": np.array(self.data.loc[array_sets == "L", :]["target"]), 
            "train_u": self.data.loc[array_sets == "U", self.data.columns != "target"], 
            "target_train_u": np.array(self.data.loc[array_sets == "U", :]["target"]),  
            "test": self.data.loc[array_sets == "T", self.data.columns != "target"], 
            "target_test": np.array(self.data.loc[array_sets == "T", :]["target"]),  
        }
    