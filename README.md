# model_tactile_pop_response

This is the code for the paper "Real-time Simulation of Populations of Tactile Receptors and Afferents in the Skin"
 
The source code of current model was presented in the function "population_simulate()" in file of receptor.py, which correctly implements the diagram as illustrated in Fig. 1 in the paper. All the simulation results in the paper were obtained by calling this funtion. You can run the code directly on your computer (>=Python3.6, with Numpy, matplotlib and scipy library)

The result figures in the paper and their corresponding code file are shown as follows:  

 
(1)  Fig.4_Model_fitting.py ---> Fig 4. Results of fitting parameters ...  
(2)  Fig.5_Response_to_vibration.py--->Fig 5. Mean firing rates from the 3 afferent units...  
(3)  Fig.6_RFsize_rec_single_unit.py--->Fig 6. Probe presentation locations relative to the tactile unit location...  
(4)  Fig.7_Textures_dots_single_repeat.py --->Fig 7. Response to doted texture...  
(5)  Fig.8_Form_letters_scaning_sim_single_repeat.py--->Fig 8. Response to embossed letters...  
(6)  Fig.9_Response_to_curve surface.py--->Fig 9. SA1 responses to surface curvature and 3-mm bar..   
(7)  Fig.10_Computation_efficiency.py --->Fig 10. Evaluation of computation efficiency...
 

The  other code files are described as follows:

(1) simset.py ---> the setup of simulations including: the defination of sampled skin area, Covering certain area in the receptor image for each tactile unit, generate the martix of resistance network... 

(2) img_to_eqstimuli.py ---> this file includes some functions of constructing EPS from image... 

(3) ultils.py ---> this file includes some common functions of reading data and calculate R2 value... 

Besides all the observed data can be found in data/txtdata

