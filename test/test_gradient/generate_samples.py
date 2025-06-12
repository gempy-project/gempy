
import torch
import numpy as np
from datetime import datetime
from torch.autograd.functional import jacobian
import pyro
import pyro.distributions as dist
from pyro.infer import Predictive

from pyro.nn import PyroModule

import gempy as gp
import gempy_engine
from gempy_engine.core.backend_tensor import BackendTensor


from helpers import *


class GempyModel(PyroModule):
    def __init__(self, interpolation_input_, geo_model_test, num_layers, slope, dtype):
        super(GempyModel, self).__init__()
        
        BackendTensor.change_backend_gempy(engine_backend=gp.data.AvailableBackends.PYTORCH)
        self.interpolation_input_ = interpolation_input_
        self.geo_model_test = geo_model_test
        self.num_layers = num_layers
        self.dtype = dtype
        self.geo_model_test.interpolation_options.sigmoid_slope = slope
        ###############################################################################
        # Seed the randomness 
        ###############################################################################
        seed = 42           
        np.random.seed(seed)
        torch.manual_seed(seed)
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Setting the seed for Pyro sampling
        
        pyro.set_rng_seed(42)
        
    
    def create_sample(self):
        
            """
            This Pyro model represents the probabilistic aspects of the geological model.
            It defines a prior distribution for the top layer's location and
            computes the thickness of the geological layer as an observed variable.

            
            interpolation_input_: represents the dictionary of random variables for surface parameters
            geo_model_test : gempy model
            
            num_layers: represents the number of layers we want to include in the model
            
            """

            Random_variable ={}
            
            # Create a random variable based on the provided dictionary used to modify input data of gempy
            counter=1
            for interpolation_input_data in self.interpolation_input_[:self.num_layers]:
                
                # Check if user wants to create random variable based on modifying the surface points of gempy
                if interpolation_input_data["update"]=="interface_data":
                    # Check what kind of distribution is needed
                    if interpolation_input_data["prior_distribution"]=="normal":
                        mean = interpolation_input_data["normal"]["mean"]
                        std  = interpolation_input_data["normal"]["std"]
                        Random_variable["mu_"+ str(counter)] = pyro.sample("mu_"+ str(counter), dist.Normal(mean, std))
                        #print(Random_variable["mu_"+ str(counter)])
                        
                    elif interpolation_input_data["prior_distribution"]=="uniform":
                        min = interpolation_input_data["uniform"]["min"]
                        max = interpolation_input_data["uniform"]["min"]
                        Random_variable["mu_"+ str(interpolation_input_data['id'])] = pyro.sample("mu_"+ str(interpolation_input_data['id']), dist.Uniform(min, max))

                        
                    else:
                        print("We have to include the distribution")
                
                    # # Check which co-ordinates direction we wants to allow and modify the surface point data
                counter=counter+1
                
          
                    
    def GenerateInputSamples(self, number_samples):
        
        pyro.clear_param_store()
        # We can build a probabilistic model using pyro by calling it 
        
        dot = pyro.render_model(self.create_sample, model_args=())
        # Generate 50 samples
        num_samples = number_samples # N
        predictive = Predictive(self.create_sample, num_samples=num_samples)
        samples = predictive()
        
        samples_list=[]
        for i in range(len(self.interpolation_input_)):
            samples_list.append(samples["mu_"+str(i+1)].reshape(-1,1))
        ######store the samples ######
        parameters=torch.hstack(samples_list) # (N, p) = number of sample X number of paramter

        return parameters.detach().numpy()
    
    def GempyForward(self, *params):
        index=0
        interpolation_input = self.geo_model_test.interpolation_input
        
        for interpolation_input_data in self.interpolation_input_[:self.num_layers]:
            # Check which co-ordinates direction we wants to allow and modify the surface point data
            if interpolation_input_data["direction"]=="X":
                interpolation_input.surface_points.sp_coords = torch.index_put(
                    interpolation_input.surface_points.sp_coords,
                    (torch.tensor([interpolation_input_data["id"]]), torch.tensor([0])),
                    params[index])
            elif interpolation_input_data["direction"]=="Y":
                interpolation_input.surface_points.sp_coords = torch.index_put(
                    interpolation_input.surface_points.sp_coords,
                    (torch.tensor([interpolation_input_data["id"]]), torch.tensor([1])),
                    params[index])
            elif interpolation_input_data["direction"]=="Z":
                interpolation_input.surface_points.sp_coords = torch.index_put(
                    interpolation_input.surface_points.sp_coords,
                    (interpolation_input_data["id"], torch.tensor([2])),
                    params[index])
            else:
                print("Wrong direction")
            
            index=index+1
        
        self.geo_model_test.solutions = gempy_engine.compute_model(
                    interpolation_input=interpolation_input,
                    options=self.geo_model_test.interpolation_options,
                    data_descriptor=self.geo_model_test.input_data_descriptor,
                    geophysics_input=self.geo_model_test.geophysics_input,
                )
        
        # Compute and observe the thickness of the geological layer
    
        m_samples = self.geo_model_test.solutions.octrees_output[0].last_output_center.custom_grid_values
        return m_samples
    
    def GenerateOutputSamples(self, Inputs_samples):
        
        Inputs_samples = torch.tensor(Inputs_samples, dtype=self.dtype)
        m_data =[]
        dmdc_data =[]
        for i in range(Inputs_samples.shape[0]):
            params_tuple = tuple([Inputs_samples[i,j].clone().requires_grad_(True) for j in range(Inputs_samples.shape[1])])
           
            m_samples = self.GempyForward(*params_tuple)
            m_data.append(m_samples)
            J = jacobian(self.GempyForward, params_tuple)
            J_matrix = torch.tensor([[J[j][i] for j in range(len(J))] for i in  range(J[0].shape[0])])
            dmdc_data.append(J_matrix)
        
        return torch.stack(m_data).detach().numpy() , torch.stack(dmdc_data).detach().numpy()
    
    


def generate_input_output_gempy_data(mesh, number_samples, slope=200, filename=None):
    # ---------------- 1️⃣ Check and create a 3D custom grid ----------------
    mesh_coordinates = mesh
    data ={}
    geo_model_test = create_initial_gempy_model(refinement=3, save=False)
    if mesh_coordinates.shape[1]==2:
        xyz_coord = np.insert(mesh_coordinates, 1, 0, axis=1)
    elif mesh_coordinates.shape[1]==3:
        xyz_coord = mesh_coordinates
    
    gp.set_custom_grid(geo_model_test.grid, xyz_coord=xyz_coord)
    geo_model_test.interpolation_options.mesh_extraction = False
    
    sp_coords_copy_test = geo_model_test.interpolation_input.surface_points.sp_coords.copy()
    
    ###############################################################################
    # 2️⃣ Make a list of gempy parameter which would be treated as a random variable
    ###############################################################################
    dtype =torch.float64
    test_list=[]
    std = 0.06  
    test_list.append({"update":"interface_data","id":torch.tensor([1]), "direction":"Z", "prior_distribution":"normal","normal":{"mean":torch.tensor(sp_coords_copy_test[1,2],dtype=dtype), "std":torch.tensor(std,dtype=dtype)}})
    test_list.append({"update":"interface_data","id":torch.tensor([4]), "direction":"Z", "prior_distribution":"normal","normal":{"mean":torch.tensor(sp_coords_copy_test[4,2],dtype=dtype), "std":torch.tensor(std,dtype=dtype)}})
    num_layers = len(test_list) # length of the list

    Gempy = GempyModel(test_list, geo_model_test, num_layers, slope=slope,  dtype=torch.float64)
    
    # ---------------- 3️⃣ Generate the samples for input parameters. c ∼ N(µ, Σ)  ----------------
    
    start_sample_input_time = datetime.now()
    c = Gempy.GenerateInputSamples(number_samples=number_samples)
    end_sample_input_time = datetime.now()
    total_time_input = end_sample_input_time - start_sample_input_time
    
    data["input"] = c.tolist() 
    # ---------------- 3️⃣ Generate the samples for output of gempy and the Jacobian matrix. m= Gempy(c) and dm/dc ----------------
    start_sample_output_time = datetime.now()
    m_data, dmdc_data = Gempy.GenerateOutputSamples(Inputs_samples=c)
    end_sample_output_time = datetime.now()
    total_time_output = end_sample_output_time - start_sample_output_time
    
    data["Gempy_output"] = m_data.tolist()
    data["Jacobian_Gempy"] = dmdc_data.tolist()
    
    return data, total_time_input, total_time_output


        
