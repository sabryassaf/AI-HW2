from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random


# TODO: section a : 3
def smart_heuristic(env: WarehouseEnv, robot_id: int):
    robot = env.get_robot(robot_id)

    # Distance to the nearest package
    nearest_package_distance = min(
        manhattan_distance(robot.position, package.position) for package in env.packages
    ) if env.packages else float('inf')

    # Battery level
    battery_level = robot.battery

    # Distance to the nearest charging station
    nearest_charging_station_distance = min(
        manhattan_distance(robot.position, station) for station in env.charge_stations
    ) if env.charge_stations else float('inf')

    # Distance to the package destination (if carrying a package)
    if robot.package:
        package_destination_distance = manhattan_distance(robot.position, robot.package.destination)
    else:
        package_destination_distance = float('inf')

    # Combine the parameters into a heuristic value
    heuristic_value = (
            -nearest_package_distance +  # Prefer closer packages
            battery_level -  # Prefer higher battery levels
            nearest_charging_station_distance -  # Prefer closer charging stations
            package_destination_distance  # Prefer closer package destinations
    )

    return heuristic_value
class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    # TODO: section b : 1

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


class AgentExpectimax(Agent):
    # TODO: section d : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


# here you can check specific paths to get to know the environment
class AgentHardCoded(Agent):
    def __init__(self):
        self.step = 0
        # specifiy the path you want to check - if a move is illegal - the agent will choose a random move
        self.trajectory = ["move north", "move east", "move north", "move north", "pick_up", "move east", "move east",
                           "move south", "move south", "move south", "move south", "drop_off"]

    def run_step(self, env: WarehouseEnv, robot_id, time_limit):
        if self.step == len(self.trajectory):
            return self.run_random_step(env, robot_id, time_limit)
        else:
            op = self.trajectory[self.step]
            if op not in env.get_legal_operators(robot_id):
                op = self.run_random_step(env, robot_id, time_limit)
            self.step += 1
            return op

    def run_random_step(self, env: WarehouseEnv, robot_id, time_limit):
        operators, _ = self.successors(env, robot_id)

        return random.choice(operators)