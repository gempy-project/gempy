import os,sys
from datetime import datetime
import numpy as np

import gempy as gp
import gempy_engine
import gempy_viewer as gpv




from helpers import *
from generate_samples import *


import warnings
warnings.filterwarnings("ignore")


def main():
    
    
    # ---------------- 1️⃣ Create the Mesh ----------------
    
    
    n_points = 32  # number of points along each axis

    x = np.linspace(0, 1, n_points)
    z = np.linspace(0, 1, n_points)

    X, Z = np.meshgrid(x, z)  # creates grid of shape (n_points, n_points)

    mesh = np.stack([X.ravel(), Z.ravel()], axis=1)  # shape: (n_points^2, 2)

    # ---------------- 2️⃣ Generate samples for the input paramter of the gempy, output and it's jacobian ----------------
    data, total_time_input, total_time_output = generate_input_output_gempy_data(mesh=mesh, number_samples=50)
    
    
    filename =  "./data_9_parameter.json"
    c ,m_data, dmdc_data = np.array(data["input"]),np.array(data["Gempy_output"]), np.array(data["Jacobian_Gempy"])
    print("Shapes-" , "Gempy Input: ", c.shape, "Gempy Output:", m_data.shape, "Jacobian shape:", dmdc_data.shape, "\n")
    
    print("Time required to generate samples for the input:\n", total_time_input)
    print("Time required to generate samples for the output of gempy and it's Jacobian matrix:\n", total_time_output)
    
    # ---------------- 3️⃣ Save the samples in a file ----------------
    # with open(filename, 'w') as file:
    #         json.dump(data, file)
        
    

if __name__ == "__main__":
    
    # Your main script code starts here
    print("Script started...")
    
    # Record the start time
    start_time = datetime.now()

    main()
    # Record the end time
    end_time = datetime.now()

    # Your main script code ends here
    print("Script ended...")
    
    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    
    print(f"Elapsed time: {elapsed_time}")
