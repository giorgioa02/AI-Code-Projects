import random
from itertools import chain, combinations

class State(tuple):
	# A state is defined as a tuple (numbers, dice_summation)
	# Access with the following command:
	# numbers, dice_summation = state
	def __new__(self, numbers_left, dice_summation):
		return tuple.__new__(State, (frozenset(numbers_left), dice_summation))

class Environment:
	def __all_states_and_actions(self):
		all_numbers_left = [[], [1]]
		for i in range(2, self.total_numbers + 1):
			for curr in range(len(all_numbers_left)):
				curr_ = all_numbers_left[curr].copy()
				curr_.append(i)
				all_numbers_left.append(curr_)

		all_dice_summation = list(range(2, 12 + 1))

		states = []
		actions = {}
		
		for number_list in all_numbers_left:
			for dice in all_dice_summation:
				states.append(State(number_list, dice))
				actions[State(number_list, dice)] = []

		for numbers in all_numbers_left:
			all_combinations = chain.from_iterable(combinations(numbers, r) for r in range(len(numbers)+1))
			for combination in all_combinations:
				dice = self.calc_sum(combination)
				if dice>=2 and dice<=12:
					actions[State(numbers, dice)].append(combination)
		
		return states, actions


	def __init__(self):
		self.total_numbers = 9
		self.prob_dist = {i:0 for i in range(2, 12 + 1)}
		for i in range(1, 6 + 1):
			for j in range(1, 6 + 1):
				self.prob_dist[i+j] += 1/6 * 1/6
		self.all_states, self.all_states_actions = self.__all_states_and_actions()

	def available_actions(self, state):
		# Return a list of actions that is allowed in this case
		# Each action is a set of numbers.
		return self.all_states_actions[state]

	def all_transition_next(self, numbers_left, action_taken):
		# Return a list of all possible next steps with their probability.
		# Input: Current numbers and an action (a subset of previous numbers)
		# Each next step is represented in tuple (state, probability of the state)
		# State is a tuple itself - (numbers_left, dice_summation) 
		numbers_left = set(numbers_left)
		for it in action_taken:
			numbers_left.remove(it)
		return [(State(numbers_left, sum_), self.prob_dist[sum_]) for sum_ in self.prob_dist]

	def get_all_states(self):
		# Get a list of all states
		# Each state is a tuple - (numbers_left, dice_summation) 
		return self.all_states

	def calc_sum(self, numbers):
		# Calculate the summation of things in a list/set
		s = 0
		for i in numbers:
			s += i
		return s



class Agent:
	def __init__(self, env):
		self.env = env
		self.all_states = env.get_all_states()
		self.utilities = {state:0 for state in self.all_states}

	def giveup_reward(self, numbers_left):
		# The reward for choosing give up at this state
		c = self.env.total_numbers
		return c*(c+1)//2 - self.env.calc_sum(numbers_left)

	def value_iteration(self):
		max_change = 1e5
		while max_change >= 0.001:
			utilities_pre = self.utilities.copy() # Copy the utility, e.g. U_{t-1}
			max_change = 0 # Measure the maximum change in all states for this iteration - if smaller than 0.001 we stop.
			for state in self.all_states:
				# Complete this part with follwoing functions 
				# self.giveup_reward, self.env.available_actions, self.env.all_transition_next

				utility = utilities_pre[state] 			# Previous utility for the current state
				numbers_left, dice_summation = state	# Unpack the state tuple into its components
				possible_actions = self.env.available_actions(state)	# Get available actions for this state
				expected_utilities = []		# Initialize a list to store the expected utility for each action

				# If there are no available actions, assign the give-up reward as the utility
				if not possible_actions:
					# Assign the utility based on remaining numbers
					self.utilities[state] = self.giveup_reward(numbers_left)
				else:
					# Calculate the expected utility for each possible action in this state
					for action in possible_actions:
						# Get all possible next states and their transition probabilities after taking this action
						next_states = self.env.all_transition_next(numbers_left, action)
						# Bellman Equation: Compute the expected utility of this action (sum the utility of each next state, weighted by its transition probability)
						expected_u = sum(prob * utilities_pre[next_s] for next_s, prob in next_states)
						# Calculate the immediate reward for taking this action
						immediate_reward = self.env.calc_sum(action)
						# Add the total utility (immediate reward + future utility) for this action to the list
						expected_utilities.append(expected_u + immediate_reward)
					# Update the utility of the state to the best action utility
					self.utilities[state] = max(expected_utilities)
				# Calculate the difference between the previous and updated utility
				change = abs(utility - self.utilities[state])
				# Track the maximum change for convergence check
				max_change = max(max_change, change)
			
	def policy(self, state):
		possible_actions = self.env.available_actions(state)
		numbers_left, dice_summation = state
		# Initialize with give up directly
		max_utility = self.giveup_reward(numbers_left)
		best_action = []

		for action in possible_actions:
			# Finish this part - Find the best action with utility computed in value iteration
			...

		return best_action


if __name__=='__main__':
	env = Environment()
	# Try the following commands before coding
	#print(env.available_actions(State([1,2,3,4,5,6,7,8,9], 12)))
	#print(env.all_transition_next([1,2,3,4,5,6,7,8,9], [1,2]))


	agent = Agent(env)
	# Q1: Complete the Value iteration code here!
	agent.value_iteration()
	print('Utility of [1,2,3,4,5,6,7,8,9], 12: %.3f' % agent.utilities[State([1,2,3,4,5,6,7,8,9], 12)])
	print('Utility of [1,3,4,5,6,7,8,9], 12: %.3f' % agent.utilities[State([1,3,4,5,6,7,8,9], 12)])
	print('Utility of [1,3,5,6,7,8,9], 12: %.3f' % agent.utilities[State([1,3,5,6,7,8,9], 12)])
	
	# Q2: Complete policy function and run the code here!
	# print('Optimal action of [1,2,3,4,5,6,7,8,9], 12: %s' % str(agent.policy(State([1,2,3,4,5,6,7,8,9], 12))))
	# print('Optimal action of [1,3,4,5,6,7,8,9], 12: %s' % str(agent.policy(State([1,3,4,5,6,7,8,9], 12))))
	# print('Optimal action of [1,3,5,6,7,8,9], 12: %s' % str(agent.policy(State([1,3,5,6,7,8,9], 12))))