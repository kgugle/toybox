# Stack
class StackEmpty(exception):
	pass

class Stack(object):
	def __init__(self):
		self.stack = []

	def push(self, x):
		self.stack.append(x)

	def pop(self):
		if self.isEmpty():
			raise StackEmpty
		else:
			return self.stack.pop()

	def is_empty(self):
		return len(self.stack) == 0

	def __str__(self):
		output = "stack: ["
		for elem in self.stack:
			output += str(elem) + ", "
		output += "]"
		return output

# Binary Search Trees
def inorder(root):
	if root:
		inorder(root.left)
		print(root.data)
		inorder(root.right)

def preorder(root):
	if root:
		inorder(root.left)
		inorder(root.right)
		print(root.data)

def postorder(root):
	if root:
		print(root.data)
		inorder(root.left)
		inorder(root.right)

# heap[i/2]	       Returns the parent node
# heap[(2*i)]	   Returns the left child node
# heap[(2*i) + 1]  Returns the right child node


# Graphs
# Dictionary with nodes a keys, sets as values
class VertexNotFound(exception):
	pass

class EdgeExists(exception):
	pass

class Graph(object):
	def __init__(self):
		self.graph = {}

	def add_edge(self, u, v):
		if u not in self.graph or v not in self.graph:
			raise VertexNotFound
		if not graph[u]:
			graph[u] = set()
		if not graph[v]:
			graph[v] = set()
		graph[u].add(v)
		graph[v].add(u)

	def vertices(self):
		return self.graph.keys()

	def edges(self):
		edge_list = []
		for u in self.graph.keys():
			for v in list(self.graph[key]):
				edge_list.append((u, v))
		return edge_list


# DFS Iterative
def dfs_iterative(graph, start):
	visited, stack = set(), [start]
	while stack:
		vertex = stack.pop()
		if vertex not in visited:
			visited.add(vertex)
			stack.extend(graph[vertex] - visited)
	return visited

# DFS Recursive
def dfs_recursive(graph, start, visited=None):
	if visited is None:
		visited = set()
	visited.add(start)
	for next_node in graph[start] - visited:
		dfs_recursive(graph, next_node, visited)
	return visited

# DFS Iterative Paths
def dfs_iterative_paths(graph, start, goal):
	stack = [(start, [stack])]
	while stack:
		(vertex, path) = stack.pop()
		for next_node in graph[start] - visited:
			if next_node == goal:
				yield path + [next]
			else:
				stack.append((next_node, path + [next_node]))
	return visited

# BFS Iterative
def bfs_iterative(graph, start):
	visited, stack = set(), [start]
	while stack:
		vertex = stack.pop(0)
		if vertex not in visited:
			visited.add(vertex)
			stack.extend(graph[vertex] - visited)
	return visited

# BFS Iterative Paths
def bfs_iterative_paths(graph, start, goal):
	stack = [(start, [stack])]
	while stack:
		(vertex, path) = stack.pop(0)
		for next_node in graph[start] - visited:
			if next_node == goal:
				yield path + [next]
			else:
				stack.append((next_node, path + [next_node]))
	return visited

# Dijkstras
from collections import namedtuple
Edge = namedtuple('Edge', ['vertex', 'weight'])

def dijkstra(graph, start, goal):
	distances = {vertex : float('inf') for vertex in graph.keys()}
	distances[source] = 0
	prev_vertices = {vertex : None for vertex in graph.keys()}
	heap = heap(graph.keys())

	while heap:
		current_vertex = heap.get_min()
		if distances[current_vertex] == float('inf'):
			break
		for neighbor, cost in graph[current_vertex]:
			alternative_route = cost + distances[current_vertex]
			if alternative_route < distances[neighbor]:
				distances[neighbor] = alternative_route
				prev_vertices[neighbor] = current_vertex
		heap.extract_min(current_vertex)

	current_vertex = goal
	path = [goal]
	while prev_vertices[current_vertex] is not None:
		current_vertex = previous_vertices[current_vertex]
		path = path + [current_vertex]
	return path

# Bellman-Ford
class NegativeWeightCycle(exception):
	pass

def bellman_ford(graph, start, goal):
	distances = {vertex : float('inf') for vertex in graph.keys()}
	distances[source] = 0
	prev_vertices = {vertex : None for vertex in graph.keys()}

	for _ in range(1, len(graph.keys()) + 1):
		for u in graph.keys():
			for v, cost in graph[u]:
				if distances[u] + cost < distances[v]:
					distances[v] = distances[u] + cost
					prev_vertices[v] = u

	for u in graph.keys():
		for v, cost in graph[u]:
			if distances[u] + cost < distances[v]:
				raise NegativeWeightCycle

	current_vertex = goal
	path = [goal]
	while prev_vertices[current_vertex] is not None:
		current_vertex = previous_vertices[current_vertex]
		path = path + [current_vertex]
	return path

# Island Count - number of islands
row, col, matrix = None, None, None
neighbors = [(-1, -1), (-1, 0), (-1, 1),
			 (0, -1), (0, 1),
			 (1, -1), (1, 0), (1, 1)]

def is_safe(i, j, visited):
	return i >= 0 and i < row and \
		   j >= 0 and j < col and \
		   not visited[i][j] and matrix[i][j]

def dfs(i, j, visited):
	visited[i][j] = True
	for idx in range(8):
		i_delta = neighbors[idx][0]
		j_delta = neighbors[idx][1]
		if is_safe(i + x_delta, j + j_delta, visited):
			dfs(i + x_delta, j + j_delta, visited)

def count_islands(matrix):
	visited = [[False for _ in range(col)] for _ in range(row)]
	count = 0

	for row in range(row):
		for col in range(col):
			dfs(row, col, visited)
			count += 1
	return count


# Coin Change - ways to make change, full table
def coin_change(S, value):
	denominations = len(S)
	table = [[0 for _ in denominations] for _ in range(value + 1)]

	for idx in range(m):
		table[0][i] = 1

	for row in range(1, value + 1):
		for col in range(m):
			# look left, denomination positions down
			x = table[row - S[col]][col] if (row - S[col]) >= 0 else 0
			# look above
			y = table[row][col - 1] if col >= 1 else 0
			table[row][col] = x + y

	return table[value][denominations - 1]


# Coin Change - ways to make change, one line
def coin_change_one_level(S, value):
	denominations = len(S)
	table = [0 for _ in range(value + 1)]
    table[0] = 1

    for row in range(0, denominations):
        for col in range(S[row], value + 1):
            table[col] += table[col - S[row]]
    return table[value]


# Knapsack Problem
# each item is a tuple (weight, value)
def knapsack(items, capacity):
	# k is the number of items left to choose
	def helper(k, available_capacity):
		if k < 0:
			return 0
		if V[k][available_capacity] == -1:
			without_curr_item = helper(k - 1, available_capacity)
			if available_capacity < items[k].weight:
				with_curr_item = 0
			else:
				items[k].value + helper(k - 1, available_capacity - items[k].weight)
			V[k][available_capacity] = max(without_curr_item, with_curr_item)
		return V[k][available_capacity]
	V = [[-1 for _ in range(capacity + 1)] for _ in items]
	return helper(len(items) - 1, capacity)



