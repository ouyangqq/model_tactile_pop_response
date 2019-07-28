# model_tactile_pop_response

This is the code for the paper "Real-time Simulation of Populations of Tactile Receptors and Afferents in the Skin"
  
 # ![image] (https://github.com/ouyangqq/model_of_single_tactile_unit/blob/master/Saved_figs/diagram.jpg) 
The source code of current model was presented in the function "population_simulate()" in file of receptor.py, which correctly implements the diagram as illustrated in Fig. 1 in the paper. All the simulation results in the paper were obtained by calling this funtion. 

The result figures in the paper and their corresponding code file are shown as follows:  

 
(1)  Fig.3_Model_fitting.py ---> Fig 3. Results of fitting parameters ...  
(2)  Fig.4_Response_to_vibration.py--->Fig 4. Mean firing rates from the 3 afferent units...  
(3)  Fig.5_RFsize_rec_single_unit.py--->Fig 5. Probe presentation locations relative to the tactile unit location...  
(4)  Fig.6_Textures_dots_single_repeat.py --->Fig 6. Response to doted texture...  
(5)  Fig.7_Form_letters_scaning_sim_single_repeat.py--->Fig 7. Response to embossed letters...  
(6)  Fig.8_Response_to_curve surface.py--->Fig 8. SA1 responses to surface curvature...   
(7)  Fig.9_Computation_efficiency.py --->Fig 9. Evaluation of computation efficiency...
 

The  other code files are described as follow:

simset.py ---> the setup of simulations including: the defination of sampled skin area 
img_to_eqstimuli.py ---> this file includes some functions of constructing EPS from image
ultils.py ---> this file includes some common functions of reading data and calculate R2 value

Besides all the observed data can be found in data/txtdata

