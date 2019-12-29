import os
import psutil
import time
import numpy as np

array_perc_u = [40, 45, 50, 55, 60, 65, 70, 75, 79, 79.5, 79.9]
array_class_model = ["rf", "nn", "krf", "ln", "ls"]
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

    for data in array_class_data: 
        for model in array_class_model:
            for punla in array_perc_u:
                
                time.sleep(np.random.randint(0, 120))

                while True:
                    cpu_available = 100-psutil.cpu_percent()
                    ram_available = psutil.virtual_memory().available /  psutil.virtual_memory().total * 100
                    gpu_available = 100 - float(os.system("nvidia-smi --format=csv,noheader --query-gpu=utilization.gpu"))
                    gpu_mem_available = 100 - float(os.system("nvidia-smi --format=csv,noheader --query-gpu=utilization.memory"))
                    if (cpu_available >= 20) & (ram_available >= 20):
                        if model in ["nn", "ln"]:
                            if (gpu_available >= 33) & (gpu_mem_available >= 33):
                                break
                        else:
                            break
                    time.sleep(30)

                os.system(
                    "python scripts/ssl-play.py" + " " +
                    "--data=" + data + " " +
                    "--model=" + model + " " +
                    "--ptest=" + str(20) + " " + 
                    "--punla=" + str(punla) + " " + 
                    "--plabe=" + str(80-punla) + " " + 
                    "--folds=" + str(5) + " " + 
                    "--samples=" + str(30) + " " +
                    "--outfolder=" + "outputs/batch_run_output/" + " " +
                    "--seed=" + "1102" " " + 
                    "&"
                )

