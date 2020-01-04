import time
import numpy as np
import subprocess
import os
from pathlib import Path

array_perc_u = [40, 45, 50, 55, 60, 65, 70, 75, 79, 79.5, 79.9]
array_class_model = ["nn", "ln", "rf", "krf", "ls"]
array_class_data = [
    "spambase", 
    "creditcard", 
    "splice", 
    "landsat",
    "letter",
    "digits",
    "cifar"
]

if __name__ == "__main__":

    for model in array_class_model:
        for data in array_class_data: 
            for punla in array_perc_u:

                punla = round(punla, 2)
                plabe = round(80-punla, 2)

                file_name = Path(
                    str("outputs/batch_run_output/") \
                    + "d_" + data \
                    + "_m_" + model \
                    + "_t" + str(20) \
                    + "_u" + str(punla) \
                    + "_l" + str(plabe) \
                    + "_f" + str(5) \
                    + "_s" + str(30) \
                    + "_r" + str(1102) \
                    + ".pickle"
                )

                if os.path.isfile(file_name): 
                    print(str(file_name) + " already exists!")
                    continue
                
                time.sleep(np.random.randint(0, 10))

                while True:
                    
                    cpu_available = 100 - float(subprocess.check_output("top -bn2 | grep 'Cpu(s)' | awk END'{print $2}'", shell=True).decode('ascii').replace("\n", "").replace(",", "."))
                    ram_available = float(subprocess.check_output("free | grep Mem | awk '{print $7/$2 * 100.0}'", shell=True).decode('ascii').replace("\n", ""))
                    gpu_available = 100 - float(subprocess.check_output("nvidia-smi --format=csv,noheader --query-gpu=utilization.gpu | awk END'{print $1}'", shell=True).decode('ascii').replace(" %\n", ""))
                    gpu_mem_available = 100 - float(subprocess.check_output("nvidia-smi --format=csv,noheader --query-gpu=utilization.memory | awk END'{print $1}'", shell=True).decode('ascii').replace(" %\n", ""))

                    print(
                        "AVAILABLE: CPU " + str(cpu_available) + 
                        " RAM " + str(ram_available) + 
                        " GPU " + str(gpu_available) + 
                        " GPU-M " + str(gpu_mem_available)
                    )

                    if (cpu_available >= 20) & (ram_available >= 20):
                        if model in ["nn", "ln"]:
                            if (gpu_available >= 50) & (gpu_mem_available >= 50):
                                break
                        else:
                            break

                    time.sleep(np.random.randint(20, 30))

                if model in ["nn", "ln"]:
                    str_background = ""
                else:
                    str_background = "&"

                os.system(
                    "python scripts/ssl-play.py" + " " +
                    "--data=" + data + " " +
                    "--model=" + model + " " +
                    "--ptest=" + str(20) + " " + 
                    "--punla=" + str(punla) + " " + 
                    "--plabe=" + str(plabe) + " " + 
                    "--folds=" + str(5) + " " + 
                    "--samples=" + str(10) + " " +
                    "--outfolder=" + "outputs/batch_run_output/" + " " +
                    "--seed=" + "1102" + " " + 
                    str_background
                )

