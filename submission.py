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

    def check_time_limit(self):
        if (time.time() - self.start_time) >= (self.time_limit - 0.2):
            return True
        return False

    def min_max(self, env: WarehouseEnv, robot_id, depth):
        # check if the game is over, if we are close the time limit, or if the robot has no battery
        if env.done() or env.num_steps <= 0 or env.get_robot(robot_id).battery <= 0 or self.check_time_limit() or depth == 0:
            return smart_heuristic(env, robot_id)

        # get the legal operators for the robot
        operators = env.get_legal_operators(robot_id)
        children = [env.clone() for _ in operators]
        if robot_id == 0:  # this is our player we are maximizing it
            cur_max = float('-inf')
            for child, op in zip(children, operators):
                child.apply_operator(robot_id, op)
                v = self.min_max(child, 1, depth - 1)
                cur_max = max(cur_max, v)
            return cur_max

        else:  # this is the enemy player we are minimizing it
            cur_min = float('inf')
            for child, op in zip(children, operators):
                child.apply_operator(robot_id, op)
                v = self.min_max(child, 0, depth - 1)
                cur_min = min(cur_min, v)
            return cur_min

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        # set the time limit for step
        self.time_limit = time_limit
        self.start_time = time.time()

        # set depth
        depth = 3

        # default value for step
        best_step_move = "move north"

        def best_step():
            nonlocal best_step_move
            # get the legal operators for the robot
            operators = env.get_legal_operators(agent_id)
            children = [env.clone() for _ in operators]

            for child, op in zip(children, operators):
                child.apply_operator(agent_id, op)

            # get the value of each child
            values = [self.min_max(child, 1 - agent_id, depth) for child in children]
            # get the index of the child with the maximum value
            max_value = max(values)
            index_selected = values.index(max_value)
            best_step_move = operators[index_selected]

        try:
            func_timeout(time_limit - 0.1, best_step)
        except FunctionTimedOut:
            pass

        return best_step_move


class AgentAlphaBeta(Agent):
    def __init__(self):
        self.start_time = None
        self.time_limit = None

    def check_time_limit(self):
        if (time.time() - self.start_time) >= (self.time_limit - 0.2):
            return True
        return False

    def min_max(self, env: WarehouseEnv, robot_id, depth, alpha, beta):
        # check if the game is over, if we are close the time limit, or if the robot has no battery
        if env.done() or env.num_steps <= 0 or env.get_robot(robot_id).battery <= 0 or self.check_time_limit() or depth == 0:
            return smart_heuristic(env, robot_id)

        # get the legal operators for the robot
        operators = env.get_legal_operators(robot_id)
        children = [env.clone() for _ in operators]
        if robot_id == 0:  # this is our player we are maximizing it
            cur_max = float('-inf')
            for child, op in zip(children, operators):
                child.apply_operator(robot_id, op)
                v = self.min_max(child, 1, depth - 1, alpha, beta)
                cur_max = max(cur_max, v)
                alpha = max(alpha, cur_max)
                if beta <= alpha:
                    return float('inf')
            return cur_max

        else:  # this is the enemy player we are minimizing it
            cur_min = float('inf')
            for child, op in zip(children, operators):
                child.apply_operator(robot_id, op)
                v = self.min_max(child, 0, depth - 1, alpha, beta)
                cur_min = min(cur_min, v)
                beta = min(beta, cur_min)
                if beta <= alpha:
                    return float('-inf')
            return cur_min

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        # set the time limit for step
        self.time_limit = time_limit
        self.start_time = time.time()

        alpha = float('-inf')
        beta = float('inf')

        # set depth
        depth = 3

        # default value for step
        best_step_move = "move north"

        def best_step():
            nonlocal best_step_move, alpha, beta
            # get the legal operators for the robot
            operators = env.get_legal_operators(agent_id)
            children = [env.clone() for _ in operators]

            for child, op in zip(children, operators):
                child.apply_operator(agent_id, op)

            # get the value of each child
            values = [self.min_max(child, 1 - agent_id, depth, alpha, beta) for child in children]
            # get the index of the child with the maximum value
            max_value = max(values)
            index_selected = values.index(max_value)
            best_step_move = operators[index_selected]

        try:
            func_timeout(time_limit - 0.1, best_step)
        except FunctionTimedOut:
            pass

        return best_step_move


class AgentExpectimax(Agent):
    def __init__(self):
        self.start_time = None
        self.time_limit = None

    def check_time_limit(self):
        if (time.time() - self.start_time) >= (self.time_limit - 0.2):
            return True
        return False

    def expectimax(self, env: WarehouseEnv, robot_id, depth):
        if env.done() or env.num_steps <= 0 or env.get_robot(robot_id).battery <= 0 or self.check_time_limit() or depth == 0:
            return smart_heuristic(env, robot_id)

        operators = env.get_legal_operators(robot_id)
        children = [env.clone() for _ in operators]

        if robot_id == 0:
            # this is our player we are maximizing it
            cur_max = float('-inf')
            for child, op in zip(children, operators):
                child.apply_operator(robot_id, op)
                v = self.expectimax(child, 1, depth - 1)
                cur_max = max(cur_max, v)
            return cur_max
        else:
            # enemy player has a probability distribution over its moves
            total_probability = 0
            expected_value = 0
            move_probabilities = []

            # initiate the probabilities for each move
            for move in operators:
                # give east, pick up package a higher probability
                if move in ["move east", "pick up package"]:
                    probability = 2
                else:
                    probability = 1

                #append the probability to the list
                move_probabilities.append(probability)
                total_probability += probability

            # iterate over the moves and calculate the expected value, and normalize it
            for move, probability, child in zip(operators, move_probabilities, children):
                child.apply_operator(robot_id, move)

                # call the expectimax recursively
                evaluation = self.expectimax(child, 0, depth - 1)
                expected_value += (probability / total_probability) * evaluation
        return expected_value

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):

        # set the time limit for step
        self.time_limit = time_limit
        self.start_time = time.time()

        # set depth
        depth = 3

        # default value for step
        best_step_move = "move north"

        def best_step():
            nonlocal best_step_move
            # get the legal operators for the robot
            operators = env.get_legal_operators(agent_id)
            children = [env.clone() for _ in operators]

            for child, op in zip(children, operators):
                child.apply_operator(agent_id, op)

            # get the value of each child
            values = [self.expectimax(child, 1 - agent_id, depth) for child in children]
            # get the index of the child with the maximum value
            max_value = max(values)
            index_selected = values.index(max_value)
            best_step_move = operators[index_selected]

        try:
            func_timeout(time_limit - 0.1, best_step)
        except FunctionTimedOut:
            pass

        return best_step_move


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
