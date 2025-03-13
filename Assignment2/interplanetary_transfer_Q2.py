""" 
Copyright (c) 2010-2020, Delft University of Technology
All rigths reserved

This file is part of the Tudat. Redistribution and use in source and 
binary forms, with or without modification, are permitted exclusively
under the terms of the Modified BSD license. You should have received
a copy of the license with this file. If not, please or visit:
http://tudat.tudelft.nl/LICENSE.
"""

from interplanetary_transfer_helper_functions import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load spice kernels.
spice.load_standard_kernels()

# Define directory where simulation output will be written
output_directory = "./SimulationOutput/"

###########################################################################
# RUN CODE FOR QUESTION 2 #################################################
###########################################################################

if __name__ == "__main__":

    # Create body objects
    bodies = create_simulation_bodies()

    # Create Lambert arc state model
    lambert_arc_ephemeris = get_lambert_problem_result(
        bodies, target_body, departure_epoch, arrival_epoch
    )
    
    """
    case_i: The initial and final propagation time equal to the initial and final times of the Lambert arc.
    case_ii: The initial and final propagation time shifted forward and backward in time, respectively, by ∆t=1 hour.
    case_iii: The initial and final propagation time shifted forward and backward in time, respectively, by ∆t such that we start/end on the sphere of influence
    case_iv: The initial and final propagation time shifted forward and backward in time, respectively, by ∆t=1 hour. The propagation is started from the middle point in time of the Lambert arc and propagated forward and backward in time.

    """
    # List cases to iterate over. STUDENT NOTE: feel free to modify if you see fit
    cases = ["case_i", "case_ii", "case_iii", "case_iv"]

    # Run propagation for each of cases i-iii
    for case in cases:

        # Define the initial and final propagation time for the current case
        departure_epoch_with_buffer = departure_epoch
        arrival_epoch_with_buffer = arrival_epoch

        # Perform propagation
        termination_settings = propagation_setup.propagator.time_termination(arrival_epoch_with_buffer)
        dynamics_simulator = propagate_trajectory(
        departure_epoch_with_buffer,
        termination_settings,
        bodies,
        lambert_arc_ephemeris,
        use_perturbations=True,
        )
        write_propagation_results_to_file(
            dynamics_simulator,
            lambert_arc_ephemeris,
            "Q2_" + str(cases.index(case)),
            output_directory,
        )

        state_history = dynamics_simulator.propagation_results.state_history
        lambert_history = get_lambert_arc_history(lambert_arc_ephemeris, state_history)

    lambert_acceleration = {}
    time_list = np.array(list(state_history.keys()))  # Extract times

    # Compute numerical derivative of velocity using finite differences
    for i in range(1, len(time_list) - 1):
        t1, t2 = time_list[i-1], time_list[i+1]
        v1, v2 = lambert_history[t1][3:6], lambert_history[t2][3:6]

        # Compute central difference: a = (v2 - v1) / (t2 - t1)
        a_lambert = (v2 - v1) / (t2 - t1)
        lambert_acceleration[time_list[i]] = a_lambert


    position_error(state_history, lambert_history)
    velocity_error(state_history, lambert_history)
    acceleration_error(time_list, lambert_acceleration, dynamics_simulator)
    plotter_3D(state_history)