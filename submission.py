from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random
import time
from func_timeout import func_timeout, FunctionTimedOut


def smart_heuristic(env: WarehouseEnv, robot_id: int):
    agent = env.get_robot(robot_id)
    enemy_robot_id = (robot_id + 1) % 2
    enemy_agent = env.get_robot(enemy_robot_id)

    # Extracting agent's characteristics
    agent_credit = agent.credit
    agent_battery = agent.battery
    agent_holding_package = agent.package

    # Extracting enemy's characteristics
    enemy_credit = enemy_agent.credit
    enemy_battery = enemy_agent.battery

    # Weights for features
    credit_weight = 1000
    battery_weight = 10
    enemy_credit_weight = -500
    enemy_battery_weight = -5
    holding_package_bonus = 100
    distance_to_destination_weight = -2
    distance_to_package_weight = -1
    winning_state_bonus = 2000

    heuristic = agent_credit * credit_weight
    heuristic += agent_battery * battery_weight
    heuristic += enemy_credit * enemy_credit_weight
    heuristic += enemy_battery * enemy_battery_weight

    if agent_holding_package:
        distance_to_destination = manhattan_distance(agent.position, agent.package.destination)
        distance_from_package_to_destination = manhattan_distance(agent.package.position, agent.package.destination)
        heuristic += distance_from_package_to_destination
        heuristic += holding_package_bonus
        heuristic += distance_to_destination * distance_to_destination_weight
    else:
        available_packages = [p for p in env.packages if p.on_board]
        if available_packages:
            closest_package = min(available_packages, key=lambda p: manhattan_distance(agent.position, p.position))
            distance_to_package = manhattan_distance(agent.position, closest_package.position)
            heuristic += distance_to_package * distance_to_package_weight

    # Check if the robot is in a winning state
    if agent_credit > enemy_credit and agent_battery >= enemy_battery:
        heuristic += winning_state_bonus

    return heuristic


class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    def __init__(self):
        self.start_time = None
        self.time_limit = None

    def min_max(self, env: WarehouseEnv, robot_id):
        robot = env.get_robot(robot_id)
        if env.done() or robot.battery <= 0 or env.num_steps <= 0:
            # If the game is over or the time limit has been reached, return the utility
            return smart_heuristic(env, robot_id)
        if robot_id == 0:
            # maximizing player turn
            value = float('-inf')
            for op in env.get_legal_operators(0):
                if time.time() - self.start_time >= self.time_limit:
                    return smart_heuristic(env, robot_id)
                # Apply the operator to a clone of the environment
                child = env.clone()
                child.apply_operator(0, op)
                v = self.min_max(child, 1)
                value = max(value, v)
            return value
        else:
            value = float('inf')
            # minimizing player turn
            for op in env.get_legal_operators(1):
                if time.time() - self.start_time >= self.time_limit:
                    return smart_heuristic(env, robot_id)

                # Apply the operator to a clone of the environment
                child = env.clone()
                child.apply_operator(1, op)
                v = self.min_max(child, 0)
                value = min(value, v)
            return value

    def run_step(self, env: WarehouseEnv, robot_id, time_limit):
        self.time_limit = time_limit
        best_step_move = "move north"

        def best_step():
            nonlocal best_step_move
            self.start_time = time.time()
            # Get the legal operators for the robot
            operators = env.get_legal_operators(robot_id)
            # Create a clone of the environment for each operator
            children = [env.clone() for _ in operators]
            # Get the value of each child
            for child, op in zip (children, operators):
                child.apply_operator(robot_id, op)
                print("applied operator", op)
            print("here")
            values = [self.min_max(child, 1 - robot_id) for child in children]
            # Get the index of the child with the maximum value
            max_value = max(values)
            index_selected = values.index(max_value)
            best_step_move = operators[index_selected]
            print("here is best step", best_step_move)

        try:
            func_timeout(time_limit - 0.1, best_step)
        except FunctionTimedOut:
            pass

        return best_step_move


class AgentAlphaBeta(Agent):
    def __init__(self):
        self.start_time = None
        self.time_limit = None

    def utility(self, env: WarehouseEnv, robot_id):
        # Return the difference in credits between the two robots
        return env.get_robot(robot_id).credit - env.get_robot((robot_id + 1) % 2).credit

    def check_time_limit(self):
        # Check if the time limit has been reached
        if (time.time() - self.start_time) >= self.time_limit:
            raise FunctionTimedOut

    def alpha_beta_heuristic(self, env: WarehouseEnv, robot_id):
        if env.done():
            return self.utility(env, robot_id)
        return smart_heuristic(env, robot_id) - smart_heuristic(env, (robot_id + 1) % 2)

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
        if (time.time() - self.start_time) >= self.time_limit:
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
