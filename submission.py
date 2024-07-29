from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random
import time
from func_timeout import func_timeout, FunctionTimedOut


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
    def __init__(self):
        self.start_time = None
        self.time_limit = None

    def check_time_limit(self):
        # Check if the time limit has been reached
        if time.time() - self.start_time >= self.time_limit:
            raise FunctionTimedOut

    def utility(self, env: WarehouseEnv, robot_id):
        # Return the difference in credits between the two robots
        return env.get_robot(robot_id).credit - env.get_robot((robot_id + 1) % 2).credit

    def min_max(self, env: WarehouseEnv, robot_id):
        if env.done() or (time.time() - self.start_time >= self.time_limit):
            # If the game is over or the time limit has been reached, return the utility
            return self.utility(env, robot_id)
        if robot_id == 0:
            # maximizing player turn
            value = float('-inf')
            for op in env.get_legal_operators(robot_id):
                self.check_time_limit()
                # Apply the operator to a clone of the environment
                child = env.clone()
                child.apply_operator(robot_id, op)
                value = max(value, self.min_max(child, 1))
            return value
        else:
            value = float('inf')
            # minimizing player turn
            for op in env.get_legal_operators(robot_id):
                self.check_time_limit()
                # Apply the operator to a clone of the environment
                child = env.clone()
                child.apply_operator(robot_id, op)
                value = min(value, self.min_max(child, 0))
            return value

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        self.start_time = time.time()
        self.time_limit = time_limit
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        values = [self.min_max(child, (agent_id + 1) % 2) for child in children]
        max_value = max(values)
        index_selected = values.index(max_value)
        return operators[index_selected]


class AgentAlphaBeta(Agent):
    def __init__(self):
        self.start_time = None
        self.time_limit = None

    def utility(self, env: WarehouseEnv, robot_id):
        # Return the difference in credits between the two robots
        return env.get_robot(robot_id).credit - env.get_robot((robot_id + 1) % 2).credit

    def check_time_limit(self):
        # Check if the time limit has been reached
        if time.time() - self.start_time >= self.time_limit:
            raise FunctionTimedOut

    def alpha_beta_heuristic(self, env: WarehouseEnv, robot_id):
        if env.done():
            return self.utility(env, robot_id)
        return smart_heuristic(env, robot_id)

    def alpha_beta(self, env: WarehouseEnv, robot_id, alpha, beta):
        if env.done() or (time.time() - self.start_time >= self.time_limit):
            return self.alpha_beta_heuristic(env, robot_id)
        if robot_id == 0:
            value = float('-inf')
            for op in env.get_legal_operators(robot_id):
                self.check_time_limit()
                child = env.clone()
                child.apply_operator(robot_id, op)
                value = max(value, self.alpha_beta(child, 1, alpha, beta))
                alpha = max(value, alpha)
                if value >= beta:
                    return float('inf')
            return value
        else:
            value = float('inf')
            for op in env.get_legal_operators(robot_id):
                self.check_time_limit()
                child = env.clone()
                child.apply_operator(robot_id, op)
                value = min(value, self.alpha_beta(child, 0, alpha, beta))
                beta = min(value, beta)
                if value <= alpha:
                    return float('-inf')
            return value

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        self.start_time = time.time()
        self.time_limit = time_limit
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        values = [self.alpha_beta(child, (agent_id + 1) % 2, float('-inf'), float('inf')) for child in children]
        max_value = max(values)
        index_selected = values.index(max_value)
        return operators[index_selected]

class AgentExpectimax(Agent):
    def __init__(self):
        self.start_time = None
        self.time_limit = None

    def utility(self, env: WarehouseEnv, robot_id):
        # Return the difference in credits between the two robots
        return env.get_robot(robot_id).credit - env.get_robot((robot_id + 1) % 2).credit

    def check_time_limit(self):
        # Check if the time limit has been reached
        if time.time() - self.start_time >= self.time_limit:
            raise FunctionTimedOut

    def AgentExpectimax(self, env: WarehouseEnv, robot_id):
        if env.done():
            return self.utility(env, robot_id)
        return smart_heuristic(env, robot_id)

    def expectimax(self, env: WarehouseEnv, robot_id):
        if env.done() or (time.time() - self.start_time >= self.time_limit):
            return self.AgentExpectimax(env, robot_id)
        operators = env.get_legal_operators(robot_id)
        children = [env.clone() for _ in operators]
        if robot_id == 0:
            value = float('-inf')
            for op in env.get_legal_operators(robot_id):
                self.check_time_limit()
                child = env.clone()
                child.apply_operator(robot_id, op)
                value = max(value, self.expectimax(child, 1))
            return value
        else:
            total_probability = 0
            expected_value = 0
            moves = self.env.get_legal_operators(robot_id)
            move_probabilities = []

            # initiate the probabilities for each move
            for move in moves:
                if move in ["move east", "pick up package"]:
                    probability = 2
                else:
                    probability = 1
                move_probabilities.append(probability)
                total_probability += probability

            for move, probability in zip(moves, move_probabilities):
                child = env.clone()
                child.apply_operator(robot_id, move)
                evaluation, _ = self.expectimax(child, True)
                expected_value += (probability / total_probability) * evaluation
        return expected_value

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        self.start_time = time.time()
        self.time_limit = time_limit
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        values = [self.expectimax(child, (agent_id + 1) % 2) for child in children]
        max_value = max(values)
        index_selected = values.index(max_value)
        return operators[index_selected]


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
