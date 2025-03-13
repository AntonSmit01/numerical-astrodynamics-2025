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

# Load spice kernels.
spice.load_standard_kernels()

# Define directory where simulation output will be written
output_directory = "./SimulationOutput/"

###########################################################################
# RUN CODE FOR QUESTION 5 #################################################
###########################################################################

if __name__ == "__main__":

    # Create body objects
    bodies = create_simulation_bodies()

    # Create Lambert arc state model
    lambert_arc_ephemeris = get_lambert_problem_result(
        bodies, target_body, departure_epoch, arrival_epoch
    )

    # Set arc length
    number_of_arcs = 10
    arc_length =  (arrival_epoch - departure_epoch) / number_of_arcs

    # Initialize results storage (10 arcs × 6 initial state entries)
    permitted_perturbations_all_arcs = np.zeros((10, 6))

    for arc_index in range(number_of_arcs):

        # Compute start and end time for current arc
        current_arc_initial_time = departure_epoch + arc_index * arc_length
        current_arc_final_time = current_arc_initial_time + arc_length

        # Get propagator settings for perturbed forward arc
        arc_initial_state = lambert_arc_ephemeris.cartesian_state(current_arc_initial_time)

        termination_settings = propagation_setup.propagator.time_termination(current_arc_final_time)

        propagator_settings = get_perturbed_propagator_settings(
            bodies, arc_initial_state, current_arc_initial_time, termination_settings
        )

        # Propagate nominal trajectory and variational equations
        sensitivity_parameters = get_sensitivity_parameter_set(propagator_settings, bodies)
        variational_equations_simulator = numerical_simulation.create_variational_equations_solver(
            bodies, propagator_settings, sensitivity_parameters
        )

        state_transition_result = variational_equations_simulator.state_transition_matrix_history
        nominal_integration_result = variational_equations_simulator.state_history

        # Compute arc initial state before applying variations
        initial_epoch = list(state_transition_result.keys())[0]
        original_initial_state = nominal_integration_result[initial_epoch]

        # Vector to store permitted perturbations
        permitted_perturbations = np.zeros(6)

        # Iterate over all initial state entries
        for entry in range(6):

            # Initialize search range for perturbation
            lower_bound = 1e-3  # Small initial perturbation
            upper_bound = 1.0   # Large value to search up to
            tolerance = 0.1  # 10% accuracy

            best_valid_perturbation = lower_bound  # Keep track of best valid perturbation

            while (upper_bound - lower_bound) / best_valid_perturbation > tolerance:

                # Set current test perturbation (midpoint of search range)
                delta_x0 = np.zeros(6)
                delta_x0[entry] = (upper_bound + lower_bound) / 2.0

                # Apply perturbation to initial state
                perturbed_initial_state = original_initial_state + delta_x0

                # Update propagator settings with perturbed state
                propagator_settings.initial_states = perturbed_initial_state

                # Propagate perturbed trajectory
                dynamics_simulator = numerical_simulation.create_dynamics_simulator(bodies, propagator_settings)
                perturbed_state_history = dynamics_simulator.state_history

                # Compute linearized state deviation Δ˜x(t)
                linearized_state_deviation = {
                    time: state_transition_result[time] @ delta_x0
                    for time in state_transition_result.keys()
                }

                # Compute true state deviation Δx(t)
                true_state_deviation = {
                    time: perturbed_state_history[time] - nominal_integration_result[time]
                    for time in perturbed_state_history.keys()
                }

                # Compute max errors over all times
                max_position_error = max(
                    np.linalg.norm(np.abs(linearized_state_deviation[t] - true_state_deviation[t])[:3]) / 1e3
                    for t in true_state_deviation.keys()
                )
                max_velocity_error = max(
                    np.linalg.norm(np.abs(linearized_state_deviation[t] - true_state_deviation[t])[3:6])
                    for t in true_state_deviation.keys()
                )
                
                # Check validity based on threshold
                if max_position_error < 100 and max_velocity_error < 1:
                    # Perturbation is still valid, increase lower bound
                    best_valid_perturbation = delta_x0[entry]
                    lower_bound = delta_x0[entry]
                else:
                    # Perturbation exceeded limits, decrease upper bound
                    upper_bound = delta_x0[entry]

            # Store best valid perturbation
            permitted_perturbations[entry] = best_valid_perturbation

        # Store computed perturbations for current arc
        permitted_perturbations_all_arcs[arc_index, :] = permitted_perturbations

    # Print results
    print("\nPermitted Initial State Perturbations (Δx0,i) for Each Arc:")
    print("(Values are in the order: x, y, z, vx, vy, vz)\n")
    for arc_index in range(10):
        print(f"Arc {arc_index + 1}: {permitted_perturbations_all_arcs[arc_index]}")

    # Optional: Save to CSV file for analysis
    np.savetxt("permitted_perturbations.csv", permitted_perturbations_all_arcs, delimiter=",", 
            header="x,y,z,vx,vy,vz", comments="")

    