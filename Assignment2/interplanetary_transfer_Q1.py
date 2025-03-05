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
# RUN CODE FOR QUESTION 1 #################################################
###########################################################################

if __name__ == "__main__":

    # Create body objects
    bodies = create_simulation_bodies()

    # Create Lambert arc state model
    lambert_arc_ephemeris = get_lambert_problem_result(
        bodies, target_body, departure_epoch, arrival_epoch
    )

    # Create propagation settings and propagate dynamics
    termination_settings = propagation_setup.propagator.time_termination(arrival_epoch)
    dynamics_simulator = propagate_trajectory(
        departure_epoch,
        termination_settings,
        bodies,
        lambert_arc_ephemeris,
        use_perturbations=False,
    )

    # Write results to file
    write_propagation_results_to_file(
        dynamics_simulator, lambert_arc_ephemeris, "Q1", output_directory
    )

    # Extract state history from dynamics simulator
    state_history = dynamics_simulator.propagation_results.state_history

    # Evaluate the Lambert arc model at each of the epochs in the state_history
    lambert_history = get_lambert_arc_history(lambert_arc_ephemeris, state_history)

    positions = []
    velocities = []

    for epoch, state in state_history.items():
        position = state[:3]  # Extract the first three components (x, y, z)
        
        positions.append(position)

    # Convert positions and velocities to numpy arrays for easier manipulation
    positions = np.array(positions)

    # Extract x, y, z coordinates from positions
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the trajectory
    ax.plot(x, y, z, label="Spacecraft Trajectory", color='b')

    # Plot settings
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title('Spacecraft Trajectory in 3D')
    ax.legend()

    # Show the plot
    plt.show()


    # Initialize lists to store differences
    times = []
    diff_x = []
    diff_y = []
    diff_z = []

    # Compute differences in position at each epoch
    for epoch in state_history.keys():
        # Get positions from both histories
        numerical_position = state_history[epoch][:3]  # x, y, z from numerical propagation
        lambert_position = lambert_history[epoch][:3]  # x, y, z from Lambert solution
        
        # Compute the difference
        position_difference = np.array(numerical_position) - np.array(lambert_position)
        
        # Store results
        times.append(epoch)       # Store time
        diff_x.append(position_difference[0])  # Difference in X
        diff_y.append(position_difference[1])  # Difference in Y
        diff_z.append(position_difference[2])  # Difference in Z

    # Convert lists to numpy arrays
    times = np.array(times)
    diff_x = np.array(diff_x)
    diff_y = np.array(diff_y)
    diff_z = np.array(diff_z)

    # Convert times to days for better readability (if needed)
    times_days = (times - times[0]) / (24 * 3600)  # Convert seconds to days

    # Create figure
    plt.figure(figsize=(10, 6))

    # Plot x-component difference
    plt.plot(times_days, diff_x, label="Δx (m)", color="r")
    plt.plot(times_days, diff_y, label="Δy (m)", color="g")
    plt.plot(times_days, diff_z, label="Δz (m)", color="b")

    # Labels and title
    plt.xlabel("Time (days)")
    plt.ylabel("Position Difference (m)")
    plt.title("Difference Between Lambert Solution and Numerical Propagation")
    plt.legend()
    plt.grid()

    # Show plot
    plt.show()