from interplanetary_transfer_helper_functions import *

# Load spice kernels.
spice.load_standard_kernels()

# Define directory where simulation output will be written
output_directory = "./SimulationOutput/"

###########################################################################
# RUN CODE FOR QUESTION 3 #################################################
###########################################################################
# Define tolerance for correction (1 meter in position)
position_tolerance = 1.0  # meters

# Maximum iterations to avoid infinite loops
max_iterations = 10

if __name__ == "__main__":

    # Create body objects
    bodies = create_simulation_bodies()

    # Create Lambert arc state model
    lambert_arc_ephemeris = get_lambert_problem_result(
        bodies, target_body, departure_epoch, arrival_epoch
    )

    ##############################################################
    # Compute number of arcs and arc length
    number_of_arcs = 10
    arc_length = (arrival_epoch - departure_epoch) / number_of_arcs

    ##############################################################

    iteration_counts = []  # Store iterations per arc

    # Compute relevant parameters (dynamics, state transition matrix, Delta V) for each arc
    for arc_index in range(number_of_arcs):

        # Compute initial and final time for arc
        current_arc_initial_time = departure_epoch + arc_index * arc_length
        current_arc_final_time = current_arc_initial_time + arc_length

        ###########################################################################
        # RUN CODE FOR QUESTION 3a ################################################
        ###########################################################################

        # Propagate dynamics on current arc (use propagate_trajectory function)
        dynamics_simulator = propagate_trajectory(current_arc_initial_time, termination_condition=propagation_setup.propagator.time_termination(current_arc_final_time), bodies=bodies, lambert_arc_ephemeris=lambert_arc_ephemeris, use_perturbations=True)
    
        write_propagation_results_to_file(
            dynamics_simulator,
            lambert_arc_ephemeris,
            "Q3_arc_" + str(arc_index),
            output_directory,
        )

        ###########################################################################
        # RUN CODE FOR QUESTION 3c/d/e ############################################
        ###########################################################################
        # Note: for question 3e, part of the code below will be put into a loop
        # for the requested iterations

        # Solve for state transition matrix on current arc
        termination_settings = propagation_setup.propagator.time_termination(
            current_arc_final_time
        )
        variational_equations_solver = propagate_variational_equations(
            current_arc_initial_time,
            termination_settings,
            bodies,
            lambert_arc_ephemeris,
        )
        state_transition_matrix_history = (
            variational_equations_solver.state_transition_matrix_history
        )
        state_history = variational_equations_solver.state_history
        lambert_history = get_lambert_arc_history(lambert_arc_ephemeris, state_history)

        # Get final state transition matrix (and its inverse)
        final_epoch = list(state_transition_matrix_history.keys())[-1]
        final_state_transition_matrix = state_transition_matrix_history[final_epoch]

        # Retrieve final state deviation
        final_state_deviation = (
            state_history[final_epoch] - lambert_history[final_epoch]
        )

        # Compute required velocity change at beginning of arc to meet required final state
        initial_state_correction = (state_history[final_epoch] - lambert_history[final_epoch]) @ np.linalg.inv(final_state_transition_matrix)

        # Propagate with correction to initial state (use propagate_trajectory function),
        # and its optional initial_state_correction input
        dynamics_simulator = propagate_trajectory(
        current_arc_initial_time,
        termination_settings,
        bodies,
        lambert_arc_ephemeris,
        use_perturbations=True,
        initial_state_correction=initial_state_correction
        )  

    corrected_state_history = apply_corrections(state_history, state_transition_matrix_history, lambert_history)

    corrected_position_error(corrected_state_history, lambert_history)



    for arc_index in range(number_of_arcs):

        # Compute initial and final time for arc
        current_arc_initial_time = departure_epoch + arc_index * arc_length
        current_arc_final_time = current_arc_initial_time + arc_length
        
        iteration_count = 0  # Initialize iteration count
        position_error = float(1000)  # Initialize error

        # Start iterative correction process
        initial_state_correction = np.zeros(6)  # No correction initially

        while position_error > position_tolerance and iteration_count < max_iterations:
            print(f"Iteration {iteration_count + 1}, Arc {arc_index + 1}, Position Error: {position_error:.6f} m")
            # Propagate trajectory with current correction
            dynamics_simulator = propagate_trajectory(
                current_arc_initial_time,
                termination_condition=propagation_setup.propagator.time_termination(current_arc_final_time),
                bodies=bodies,
                lambert_arc_ephemeris=lambert_arc_ephemeris,
                use_perturbations=True,
                initial_state_correction=initial_state_correction
            )

            # Compute state transition matrix
            variational_equations_solver = propagate_variational_equations(
                current_arc_initial_time,
                propagation_setup.propagator.time_termination(current_arc_final_time),
                bodies,
                lambert_arc_ephemeris
            )

            state_transition_matrix_history = variational_equations_solver.state_transition_matrix_history
            state_history = variational_equations_solver.state_history
            lambert_history = get_lambert_arc_history(lambert_arc_ephemeris, state_history)

            # Get final state deviation
            final_epoch = list(state_transition_matrix_history.keys())[-1]
            final_state_transition_matrix = state_transition_matrix_history[final_epoch]
            final_state_deviation = state_history[final_epoch] - lambert_history[final_epoch]

            # Compute new correction
            initial_state_correction = np.linalg.solve(final_state_transition_matrix, final_state_deviation)

            # Compute new position error (norm of position deviation)
            position_error = np.linalg.norm(final_state_deviation[:3])

            iteration_count += 1  # Increment iteration count
            print(f"Iteration {iteration_count + 1}, Arc {arc_index + 1}, Applied Correction: {initial_state_correction}")    
        iteration_counts.append(iteration_count)  # Store number of iterations

    # Print iteration results
    print("\nIteration Count Per Arc:")
    print("-----------------------")
    for i, count in enumerate(iteration_counts):
        print(f"Arc {i + 1}: {count} iterations")