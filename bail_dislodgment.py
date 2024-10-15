# bail_dislodgment.py

import math

class BailDislodgment:
    def __init__(self, mass_ball, velocity_ball, angle_impact, mass_bail, height_bail):
        self.mass_ball = mass_ball  # in kg
        self.velocity_ball = velocity_ball  # in m/s
        self.angle_impact = angle_impact  # in degrees
        self.mass_bail = mass_bail  # in kg
        self.height_bail = height_bail  # in meters
        self.g = 9.8  # Acceleration due to gravity (m/s²)

    def calculate_impact_force(self):
        # Calculate the effective impact force on the stumps
        angle_radians = math.radians(self.angle_impact)
        F_impact = self.mass_ball * self.velocity_ball * math.cos(angle_radians)
        return F_impact

    def calculate_threshold_force(self):
        # Calculate the threshold force required to dislodge the bail
        torque_bail = self.mass_bail * self.g * self.height_bail
        return torque_bail / self.height_bail  # Assuming height is where we measure torque

    def is_bail_dislodged(self):
        # Check if the impact force is greater than or equal to the threshold force
        F_impact = self.calculate_impact_force()
        F_threshold = self.calculate_threshold_force()
        return F_impact >= F_threshold

# Example usage:
# bail = BailDislodgment(mass_ball=0.155, velocity_ball=30, angle_impact=45, mass_bail=0.05, height_bail=0.15)
# bail.is_bail_dislodged()  # This returns True or False based on the calculations
