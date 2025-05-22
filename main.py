import heapq
import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import google.generativeai as genai
from contextlib import asynccontextmanager
import json
from loguru import logger

logger.add("app.log", rotation="500 MB", level="INFO")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up FastAPI application")
    yield
    logger.info("Shutting down FastAPI application")

app = FastAPI(lifespan=lifespan)

# CORS setup to allow frontend requests
origins = ["https://algoverse1.netlify.app"]
GEMINI_API_KEY = "AIzaSyBytaqUQhjGY8ufW8BLlBPoFtkjdHjHTBA"
genai.configure(api_key=GEMINI_API_KEY)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class AlgorithmRequest(BaseModel):
    category: str  # e.g., "Sorting"
    input_size: str  # e.g., "Small (n < 1000)"

class AlgorithmEntry(BaseModel):
    algorithm: str
    input_size: str
    time_complexity: str
    space_complexity: str
    execution_time_seconds: str  # Keeping as str to match your example (e.g., "10000s")

class Recommendation(BaseModel):
    recommendation: str
    insights: str
    performance_tips: List[str]
    trade_offs: List[str]

class BenchmarkResponse(BaseModel):
    algorithms: List[AlgorithmEntry]
    recommendations: List[Recommendation]

# Pydantic model for input validation
class SortRequest(BaseModel):
    array: List[int]

class SearchRequest(BaseModel):
    array: List[int]
    value: int

class RecursionRequest(BaseModel):
    n: int

class GraphRequest(BaseModel):
    adjacency_list: Dict[str, List[str]]
    start_node: str

class Edge(BaseModel):
    toNode: str
    weight: int

class DijkstraGraphRequest(BaseModel):
    adjacency_list: Dict[str, List[Edge]]
    start_node: str


class GraphStep(BaseModel):
    type: str
    node: str | None = None
    source: str | None = None
    target: str | None = None
    from_node: str | None = None
    distance: float | None = None

class PriorityQueue:
    def __init__(self):
        self.heap = []
        self.entry_finder = {}
        self.REMOVED = '<removed>'
        self.counter = 0

    def enqueue(self, item, priority):
        if item in self.entry_finder:
            self.remove(item)
        entry = [priority, self.counter, item]
        self.entry_finder[item] = entry
        heapq.heappush(self.heap, entry)
        self.counter += 1

    def remove(self, item):
        entry = self.entry_finder.pop(item, None)
        if entry:
            entry[-1] = self.REMOVED

    def dequeue(self):
        while self.heap:
            priority, count, item = heapq.heappop(self.heap)
            if item is not self.REMOVED:
                del self.entry_finder[item]
                return item
        raise KeyError('Pop from an empty priority queue')

    def is_empty(self):
        return len(self.entry_finder) == 0

class UnionFind:
    def __init__(self, elements):
        self.parent = {element: element for element in elements}
        self.rank = {element: 0 for element in elements}

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return
        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

    def connected(self, x, y):
        return self.find(x) == self.find(y)


@app.get("/")
async def root():
    return {"message": "Hello test"}

@app.post("/sort/bubble")
async def bubble_sort(request: SortRequest):
    arr = request.array
    if not arr or not all(isinstance(x, (int, float)) for x in arr):
        raise HTTPException(status_code=400, detail="Invalid input: must be a non-empty array of numbers")

    steps = []
    array = arr.copy()
    n = len(array)

    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            # Record comparison step
            steps.append({"type": "compare", "indices": [j, j + 1]})
            if array[j] > array[j + 1]:
                # Record swap step
                array[j], array[j + 1] = array[j + 1], array[j]
                steps.append({"type": "swap", "indices": [j, j + 1], "array": array.copy()})
                swapped = True
        if not swapped:
            break
        # Record sorted index
        steps.append({"type": "sorted", "index": n - i - 1})

    return {"steps": steps, "sortedArray": array}


@app.post("/sort/selection")
async def selection_sort(request: SortRequest):
    arr = request.array
    if not arr or not all(isinstance(x, (int, float)) for x in arr):
        raise HTTPException(status_code=400, detail="Invalid input: must be a non-empty array of numbers")

    steps = []
    array = arr.copy()
    n = len(array)

    for i in range(n):
        min_idx = i

        # Find the minimum element in the unsorted part of the array
        for j in range(i + 1, n):
            # Record comparison step
            steps.append({"type": "compare", "indices": [min_idx, j]})
            if array[j] < array[min_idx]:
                min_idx = j
                steps.append({"type": "minimum", "indices": [min_idx]})

        # Swap the found minimum element with the first element of the unsorted part
        if min_idx != i:
            array[i], array[min_idx] = array[min_idx], array[i]
            steps.append({"type": "swap", "indices": [i, min_idx], "array": array.copy()})

        # Record sorted index
        steps.append({"type": "sorted", "index": i})

    return {"steps": steps, "sortedArray": array}

@app.post("/sort/insertion")
async def insertion_sort(request: SortRequest):
    arr = request.array
    if not arr or not all(isinstance(x, (int, float)) for x in arr):
        raise HTTPException(status_code=400, detail="Invalid input: must be a non-empty array of numbers")

    steps = []
    array = arr.copy()
    n = len(array)

    # Mark the first element as sorted
    steps.append({"type": "sorted", "index": 0})

    # Traverse through 1 to len(array)
    for i in range(1, n):
        # Compare key with each element on the left until smaller element is found
        steps.append({"type": "selected", "indices": [i]})
        for j in range(i, 0, -1):
            # Record comparison step
            steps.append({"type": "compare", "indices": [j, j - 1]})

            if array[j] < array[j - 1]:
                # Swap adjacent elements
                array[j], array[j - 1] = array[j - 1], array[j]
                steps.append({"type": "swap", "indices": [j, j - 1], "array": array.copy()})
            else:
                break

        # Mark as sorted up to current index
        steps.append({"type": "sorted", "index": i})

    return {"steps": steps, "sortedArray": array}


@app.post("/sort/merge")
async def merge_sort(request: SortRequest):
    try:
        arr = request.array
        if not arr or not all(isinstance(x, (int, float)) for x in arr):
            raise HTTPException(status_code=400, detail="Invalid input: must be a non-empty array of numbers")

        steps = []
        array = arr.copy()

        def merge_sort_recursive(arr: List[float], left: int, right: int, depth: int = 0):
            if left < right:
                mid = (left + right) // 2

                # Step 1: Split for color-coded highlighting
                left_half = arr[left:mid + 1]
                right_half = arr[mid + 1:right + 1]
                steps.append({
                    "type": "split",
                    "left": left,
                    "right": right,
                    "mid": mid,
                    "left_half": left_half.copy(),
                    "right_half": right_half.copy(),
                    "depth": depth
                })

                # Step 2: Recursion to move bars down
                steps.append({
                    "type": "recurse",
                    "left": left,
                    "right": mid,
                    "depth": depth + 1
                })
                merge_sort_recursive(arr, left, mid, depth + 1)

                steps.append({
                    "type": "recurse",
                    "left": mid + 1,
                    "right": right,
                    "depth": depth + 1
                })
                merge_sort_recursive(arr, mid + 1, right, depth + 1)

                # Step 3: Backtrack to move bars up
                steps.append({
                    "type": "backtrack",
                    "left": left,
                    "right": right,
                    "depth": depth
                })

                # Step 4: Merge
                merge(arr, left, mid, right, depth)

        def merge(arr: List[float], left: int, mid: int, right: int, depth: int):
            left_half = arr[left:mid + 1].copy()
            right_half = arr[mid + 1:right + 1].copy()
            before_array = arr.copy()
            # Before merge
            steps.append({
                "type": "merge_before",
                "left": left,
                "right": right,
                "before_array": before_array,
                "depth": depth
            })

            i = j = 0
            k = left
            while i < len(left_half) and j < len(right_half):
                if left_half[i] <= right_half[j]:
                    arr[k] = left_half[i]
                    i += 1
                else:
                    arr[k] = right_half[j]
                    j += 1
                k += 1

            while i < len(left_half):
                arr[k] = left_half[i]
                i += 1
                k += 1

            while j < len(right_half):
                arr[k] = right_half[j]
                j += 1
                k += 1

            # After merge
            steps.append({
                "type": "merge_after",
                "left": left,
                "right": right,
                "before_array": before_array,
                "after_array": arr.copy(),
                "depth": depth
            })

            # Sorted segment
            steps.append({
                "type": "sorted",
                "left": left,
                "right": right,
                "depth": depth
            })

        merge_sort_recursive(array, 0, len(array) - 1)

        return {"steps": steps, "sortedArray": array}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/sort/quick")
async def quick_sort(request: SortRequest):
    try:
        arr = request.array
        if not arr or not all(isinstance(x, (int, float)) for x in arr):
            raise HTTPException(status_code=400, detail="Invalid input: must be a non-empty array of numbers")

        steps = []
        array = arr.copy()

        def quick_sort_recursive(arr: List[float], left: int, right: int):
            if left < right:
                # Step 1: Partition the array
                pivot_index = partition(arr, left, right)

                # Step 2: Recursively sort the left partition
                quick_sort_recursive(arr, left, pivot_index - 1)

                # Step 3: Recursively sort the right partition
                quick_sort_recursive(arr, pivot_index + 1, right)

                # After both recursive calls, the subarray from left to right is fully sorted
                steps.append({
                    "type": "sorted",
                    "indices": list(range(left, right + 1))
                })
            elif left == right:
                # Single element is sorted
                steps.append({
                    "type": "sorted",
                    "indices": [left]
                })

        def partition(arr: List[float], left: int, right: int) -> int:
            pivot = arr[right]
            # Highlight pivot
            steps.append({
                "type": "pivot",
                "pivot": right
            })

            i = left - 1  # Index of smaller element

            for j in range(left, right):
                # Highlight elements being compared, include pivot
                steps.append({
                    "type": "compare",
                    "left": j,
                    "right": right,
                    "pivot": right
                })

                if arr[j] <= pivot:
                    i += 1  # Increment index of smaller element
                    if i != j:
                        # Record swap, include pivot
                        steps.append({
                            "type": "swap",
                            "index1": i,
                            "index2": j,
                            "pivot": right
                        })
                        arr[i], arr[j] = arr[j], arr[i]

            # Place pivot in its correct position
            if i + 1 != right:
                steps.append({
                    "type": "swap",
                    "index1": i + 1,
                    "index2": right,
                    "pivot": right
                })
                arr[i + 1], arr[right] = arr[right], arr[i + 1]

            # Highlight the partition
            steps.append({
                "type": "partition",
                "left": left,
                "right": right,
                "pivot": i + 1
            })

            # Mark only the pivot as sorted
            steps.append({
                "type": "sorted",
                "indices": [i + 1]
            })

            return i + 1

        quick_sort_recursive(array, 0, len(array) - 1)

        return {"steps": steps, "sortedArray": array}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/search/linear")
async def insertion_sort(request: SearchRequest):
    arr = request.array
    value = request.value
    if not arr or not all(isinstance(x, (int, float)) for x in arr):
        raise HTTPException(status_code=400, detail="Invalid input: must be a non-empty array of numbers")

    steps = []
    n = len(arr)

    for i in range(0, n):
        steps.append({
            "type": "checking",
            "index": i
        })
        if arr[i] == value:
            steps.append({
                "type":"found",
                "index":i
            })

            return {"steps":steps}

    steps.append({
        "type": "not_found",
        "index": -1
    })

    return {"steps": steps}


@app.post("/search/binary")
async def binary_search(request: SearchRequest):
    arr = request.array
    value = request.value
    if not arr or not all(isinstance(x, (int, float)) for x in arr):
        raise HTTPException(status_code=400, detail="Invalid input: must be a non-empty array of numbers")

    # Check if array is sorted
    if not all(arr[i] <= arr[i + 1] for i in range(len(arr) - 1)):
        raise HTTPException(status_code=400, detail="Input array must be sorted")

    steps = []
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2
        steps.append({
            "type": "checking",
            "index": mid,
            "left": left,
            "right": right
        })
        # Rest of the code...

        if arr[mid] == value:
            steps.append({
                "type": "found",
                "index": mid
            })
            return {"steps": steps}
        elif arr[mid] < value:
            left = mid + 1
            steps.append({
                "type": "search_right",
                "left": left,
                "right": right
            })
        else:
            right = mid - 1
            steps.append({
                "type": "search_left",
                "left": left,
                "right": right
            })

    steps.append({
        "type": "not_found",
        "index": -1
    })

    return {"steps": steps}


@app.post("/graph/bfs", response_model=List[GraphStep])
async def bfs(request: GraphRequest):
    adjacency_list = request.adjacency_list
    start_node = request.start_node

    # Validate inputs
    if not isinstance(adjacency_list, dict):
        raise HTTPException(status_code=400, detail="Adjacency list must be a dictionary")
    if not isinstance(start_node, str) or not start_node:
        raise HTTPException(status_code=400, detail="Start node must be a non-empty string")
    if start_node not in adjacency_list:
        raise HTTPException(status_code=400, detail=f"Start node '{start_node}' not found in graph")

    # Validate nodes and edges
    for node, neighbors in adjacency_list.items():
        if not isinstance(node, str):
            raise HTTPException(status_code=400, detail="Node keys must be strings")
        if not isinstance(neighbors, list):
            raise HTTPException(status_code=400, detail=f"Neighbors of node '{node}' must be a list")
        for neighbor in neighbors:
            if not isinstance(neighbor, str):
                raise HTTPException(status_code=400, detail=f"Neighbor '{neighbor}' of node '{node}' must be a string")
            if neighbor not in adjacency_list:
                raise HTTPException(status_code=400, detail=f"Neighbor '{neighbor}' of node '{node}' not found in graph")

    # BFS algorithm
    steps = []
    visited = set()
    queue = [start_node]
    visited.add(start_node)

    steps.append({"type": "queue", "node": start_node})

    while queue:
        current = queue.pop(0)
        steps.append({"type": "dequeue", "node": current})

        neighbors = adjacency_list.get(current, [])
        for neighbor in neighbors:
            steps.append({"type": "explore", "source": current, "target": neighbor})
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                steps.append({"type": "visit", "node": neighbor, "from": current})
                steps.append({"type": "queue", "node": neighbor})
            else:
                steps.append({"type": "visited", "source": current, "target": neighbor})

        steps.append({"type": "finish", "node": current})

    return steps

@app.post("/graph/dfs", response_model=List[GraphStep])
async def dfs(request: GraphRequest):
    try:
        adjacency_list = request.adjacency_list
        start_node = request.start_node

        # Validate inputs
        if not isinstance(adjacency_list, dict):
            raise HTTPException(status_code=400, detail="Adjacency list must be a dictionary")
        if not isinstance(start_node, str) or not start_node.strip():
            raise HTTPException(status_code=400, detail="Start node must be a non-empty string")
        if start_node not in adjacency_list:
            raise HTTPException(status_code=400, detail=f"Start node '{start_node}' not found in graph")

        # Validate nodes and edges
        for node, neighbors in adjacency_list.items():
            if not isinstance(node, str) or not node.strip():
                raise HTTPException(status_code=400, detail="Node keys must be non-empty strings")
            if not isinstance(neighbors, list):
                raise HTTPException(status_code=400, detail=f"Neighbors of node '{node}' must be a list")
            for neighbor in neighbors:
                if not isinstance(neighbor, str) or not neighbor.strip():
                    raise HTTPException(status_code=400, detail=f"Neighbor of node '{node}' must be a non-empty string")
                if neighbor not in adjacency_list:
                    raise HTTPException(status_code=400, detail=f"Neighbor '{neighbor}' of node '{node}' not found in graph")

        # DFS algorithm
        steps = []
        visited = set()
        stack = [start_node]

        steps.append({"type": "queue", "node": start_node})

        while stack:
            current = stack.pop()
            if current not in visited:
                steps.append({"type": "dequeue", "node": current})
                visited.add(current)

                neighbors = adjacency_list.get(current, [])
                # Reverse neighbors to match frontend's stack behavior
                for neighbor in neighbors[::-1]:
                    steps.append({"type": "explore", "source": current, "target": neighbor})
                    if neighbor not in visited:
                        stack.append(neighbor)
                        steps.append({"type": "visit", "node": neighbor, "from": current})
                        steps.append({"type": "queue", "node": neighbor})
                    else:
                        steps.append({"type": "visited", "source": current, "target": neighbor})

                steps.append({"type": "finish", "node": current})

        return steps
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@app.post("/graph/dijkstra", response_model=List[GraphStep])
async def dijkstra(request: DijkstraGraphRequest):
    try:
        adjacency_list = request.adjacency_list
        start_node = request.start_node

        # Validate inputs
        if not isinstance(adjacency_list, dict):
            raise HTTPException(status_code=400, detail="Adjacency list must be a dictionary")
        if not isinstance(start_node, str) or not start_node.strip():
            raise HTTPException(status_code=400, detail="Start node must be a non-empty string")
        if start_node not in adjacency_list:
            raise HTTPException(status_code=400, detail=f"Start node '{start_node}' not found in graph")

        # Validate nodes and edges
        for node, edges in adjacency_list.items():
            if not isinstance(node, str) or not node.strip():
                raise HTTPException(status_code=400, detail="Node keys must be non-empty strings")
            if not isinstance(edges, list):
                raise HTTPException(status_code=400, detail=f"Edges of node '{node}' must be a list")
            for edge in edges:
                if not isinstance(edge.toNode, str) or not edge.toNode.strip():
                    raise HTTPException(status_code=400, detail=f"Target node of edge from '{node}' must be a non-empty string")
                if edge.toNode not in adjacency_list:
                    raise HTTPException(status_code=400, detail=f"Target node '{edge.toNode}' from '{node}' not found in graph")
                if not isinstance(edge.weight, int) or edge.weight <= 0:
                    raise HTTPException(status_code=400, detail=f"Weight of edge from '{node}' to '{edge.toNode}' must be a positive integer")

        # Dijkstra's algorithm
        steps = []
        distances = {node: float('inf') for node in adjacency_list}
        previous = {}
        pq = PriorityQueue()
        visited = set()

        distances[start_node] = 0
        pq.enqueue(start_node, 0)
        steps.append({"type": "queue", "node": start_node})

        while not pq.is_empty():
            current = pq.dequeue()
            if current in visited:
                continue

            steps.append({"type": "dequeue", "node": current})
            visited.add(current)

            for edge in adjacency_list.get(current, []):
                toNode = edge.toNode
                weight = edge.weight
                steps.append({"type": "explore", "source": current, "target": toNode})

                new_distance = distances[current] + weight
                if new_distance < distances[toNode]:
                    distances[toNode] = new_distance
                    previous[toNode] = current
                    if toNode not in visited:
                        pq.enqueue(toNode, new_distance)
                        steps.append({"type": "visit", "node": toNode, "distance": new_distance, "from": current})
                    else:
                        steps.append({"type": "distance", "node": toNode, "distance": new_distance})
                else:
                    steps.append({"type": "visited", "source": current, "target": toNode})

            steps.append({"type": "finish", "node": current, "distance": distances[current]})

        # Add shortest path steps
        for node in adjacency_list:
            if node != start_node and node in previous:
                current = node
                path = []
                while current in previous:
                    path.append((previous[current], current))
                    current = previous[current]
                for source, target in reversed(path):
                    steps.append({"type": "path", "source": source, "target": target})

        return steps
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.post("/graph/kruskal", response_model=List[GraphStep])
async def kruskal(request: DijkstraGraphRequest):
    try:
        adjacency_list = request.adjacency_list

        # Validate inputs
        if not isinstance(adjacency_list, dict):
            raise HTTPException(status_code=400, detail="Adjacency list must be a dictionary")
        if not adjacency_list:
            raise HTTPException(status_code=400, detail="Adjacency list cannot be empty")

        # Validate nodes and edges
        for node, edges in adjacency_list.items():
            if not isinstance(node, str) or not node.strip():
                raise HTTPException(status_code=400, detail="Node keys must be non-empty strings")
            if not isinstance(edges, list):
                raise HTTPException(status_code=400, detail=f"Edges of node '{node}' must be a list")
            for edge in edges:
                if not isinstance(edge.toNode, str) or not edge.toNode.strip():
                    raise HTTPException(status_code=400, detail=f"Target node of edge from '{node}' must be a non-empty string")
                if edge.toNode not in adjacency_list:
                    raise HTTPException(status_code=400, detail=f"Target node '{edge.toNode}' from '{node}' not found in graph")
                if not isinstance(edge.weight, int) or edge.weight <= 0:
                    raise HTTPException(status_code=400, detail=f"Weight of edge from '{node}' to '{edge.toNode}' must be a positive integer")

        # Kruskal's algorithm
        steps = []
        nodes = list(adjacency_list.keys())
        uf = UnionFind(nodes)

        # Collect unique edges (undirected)
        edges = []
        edge_map = set()
        for source, neighbors in adjacency_list.items():
            for edge in neighbors:
                key = tuple(sorted([source, edge.toNode]))
                if key not in edge_map:
                    edges.append({"source": source, "target": edge.toNode, "weight": edge.weight})
                    edge_map.add(key)

        # Sort edges by weight
        edges.sort(key=lambda x: x["weight"])

        # Process edges
        for edge in edges:
            source = edge["source"]
            target = edge["target"]

            # Step: Consider edge
            steps.append({"type": "consider", "source": source, "target": target})

            # Check if edge forms a cycle
            if not uf.connected(source, target):
                uf.union(source, target)
                steps.append({"type": "add", "source": source, "target": target})
            else:
                steps.append({"type": "reject", "source": source, "target": target})

        return steps
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.post("/recursion/factorial")
async def get_factorial_steps(request: RecursionRequest):
    n = request.n
    if n < 0 or n > 12:
        raise HTTPException(status_code=400, detail="Number must be between 0 and 12")

    steps = []

    def factorial_recursive(current):
        # Add entry step
        steps.append({
            "type": "entry",
            "n": current,
            "description": f"Enter factorial({current})"
        })

        if current == 0 or current == 1:
            # Base case
            steps.append({
                "type": "base",
                "n": current,
                "result": 1,
                "description": f"Base case: {current}! = 1"
            })
            return 1
        else:
            # Recursive case
            steps.append({
                "type": "recursive",
                "n": current,
                "nextCall": current - 1,
                "description": f"Call factorial({current - 1})"
            })

            sub_result = factorial_recursive(current - 1)

            # Return step
            steps.append({
                "type": "return",
                "n": current,
                "subValue": sub_result,
                "returnValue": current * sub_result,
                "description": f"Return {current} × {sub_result} = {current * sub_result}"
            })

            return current * sub_result

    factorial_recursive(n)

    return {"steps": steps}


class HanoiRequest(BaseModel):
    n: int


class HanoiStepsResponse(BaseModel):
    steps: list[dict]


@app.post("/recursion/hanoi", response_model=HanoiStepsResponse)
async def get_hanoi_steps(request: HanoiRequest):
    n = request.n
    if n < 1 or n > 5:
        raise HTTPException(status_code=400, detail="Number of disks must be between 1 and 5")

    steps = []

    def hanoi_recursive(n: int, source: str, auxiliary: str, destination: str):
        if n > 0:
            # Move n-1 disks from source to auxiliary via destination
            hanoi_recursive(n - 1, source, destination, auxiliary)

            # Move the nth disk from source to destination
            steps.append({
                "type": "move",
                "disk": n,
                "from": source,
                "to": destination,
                "description": f"Move disk {n} from peg {source} to peg {destination}"
            })

            # Move n-1 disks from auxiliary to destination via source
            hanoi_recursive(n - 1, auxiliary, source, destination)

    hanoi_recursive(n, 'A', 'B', 'C')

    return {"steps": steps}


def get_text_model():
    """Returns a text-only generative model."""
    try:
        return genai.GenerativeModel('gemini-2.0-flash')  # Updated model as requested
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise HTTPException(status_code=500, detail="Model initialization failed. Check if 'gemini-2.0-flash' is a valid model.")

@app.post("/compare")
async def compare(request: AlgorithmRequest,
    model: genai.GenerativeModel = Depends(get_text_model)
):
    """Compare algorithms for a given category and input size."""
    try:
        prompt = f"""
        You are an expert in algorithms and performance analysis. The user has selected category '{request.category}' and input size '{request.input_size}'. Focus exclusively on the selected category '{request.category}' and ignore all other categories. Compare only the algorithms specified for '{request.category}' as follows:

        - If '{request.category}' is 'Search Algorithms': Compare Linear Search and Binary Search.
        - If '{request.category}' is 'Sorting Algorithms': Compare Bubble Sort, Merge Sort, Selection Sort, Insertion Sort, Quick Sort, and Heap Sort.
        - If '{request.category}' is 'Graph Traversal': Compare Breadth-First Search (BFS), Depth-First Search (DFS), Dijkstra’s, and Kruskal.
        - If '{request.category}' is 'Recursion Algorithms': Compare Factorial Calculation and Tower of Hanoi Calculation.
        
        For the selected category '{request.category}' and input size '{request.input_size}', provide:
        1. A table comparing the specified algorithms with columns: algorithm, input_size, time_complexity, space_complexity, execution_time_seconds (as string with 's', e.g., '0.005s'). 
           - Estimate execution time realistically as follows:
             - Assume a CPU with 1 billion operations per second (1 operation ≈ 1 nanosecond).
             - Map '{request.input_size}' to a representative 'n':
               - 'Small (n < 1000)' → n = 500
               - 'Medium (1000 < n < 100000)' → n = 10,000
               - 'Large (n > 100000)' → n = 1,000,000
             - Use the algorithm's average-case time complexity (e.g., O(n), O(n log n)) with the mapped 'n' to calculate the number of operations.
             - Calculate base time: Number of operations / 1 billion seconds.
             - Add randomness: Adjust the base time by a random factor between -20% and +20% to mimic real-world variations (e.g., caching, system load). For example, if base time is 0.01s, final time could range from 0.008s to 0.012s.
             - Do not round the final time; keep full precision and append 's' (e.g., '0.0001329s').
        2. A recommendation with insights, performance_tips (list, max 3), and trade_offs (list, max 3). For the 'recommendation' field, return only the name of the recommended algorithm(s) (e.g., 'Quick Sort') without any prefix like 'Recommended Algorithm:'.
        
        Return ONLY this JSON format with no additional text outside the JSON:
        [
          [
            {{"algorithm": "Example", "input_size": "{request.input_size}", "time_complexity": "O(n)", "space_complexity": "O(1)", "execution_time_seconds": "0.005s"}},
            ...
          ],
          [
            {{"recommendation": "Example", "insights": "Explanation...", "performance_tips": ["Tip 1", "Tip 2", "Tip 3"], "trade_offs": ["Trade-off 1", "Trade-off 2", "Trade-off 3"]}}
          ]
        ]
        
        Specifically, format the recommendation, insights, performance tips, and trade-offs to match this style:
        Quick Sort
        Based on your input size and requirements, Quick Sort performs best with O(nlogn) average time complexity and relatively minimal space usage (O(logn) average). It showed superior performance in benchmarks for the given dataset size.
        Performance Tips:
        - Consider pivot selection strategy.
        - Implement tail-call optimization.
        - Use insertion sort for small subarrays.
        Trade-offs:
        - Not a stable sorting algorithm.
        - Worst-case O(n2) time complexity is possible.
        - Performance highly depends on pivot choice.
        
        Do not include algorithms or data for any category other than '{request.category}'. Limit your response strictly to the selected category and input size.
        """
        logger.info(f"Sending prompt to Gemini: {request.category}, {request.input_size}")

        response = model.generate_content(prompt)
        response_text = response.text.strip()
        logger.info(f"Raw Gemini response (full): {response_text}")

        if not response_text:
            logger.error("Gemini returned an empty response")
            raise HTTPException(status_code=500, detail="Error: Gemini returned an empty response")

        # Parse the response
        try:
            # Load the entire response as a list containing two arrays
            data = json.loads(response_text)
            if not isinstance(data, list) or len(data) != 2:
                logger.error(f"Invalid response format: {response_text[:100]}...")
                raise ValueError("Response must contain exactly two arrays")

            # Parse the algorithms table
            algorithms_data = data[0]
            algorithms = [AlgorithmEntry(**entry) for entry in algorithms_data]

            # Parse the recommendations
            recommendations_data = data[1]
            recommendations = [Recommendation(**entry) for entry in recommendations_data]

        except json.JSONDecodeError as e:
            # Attempt to extract JSON if it's embedded in text
            import re
            json_match = re.search(r'\[\s*\[.*?\]\s*,\s*\[.*?\]\s*\]', response_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(0)
                logger.info(f"Extracted JSON portion: {json_text[:200]}...")
                try:
                    data = json.loads(json_text)
                    if not isinstance(data, list) or len(data) != 2:
                        logger.error(f"Extracted JSON invalid: {json_text[:100]}...")
                        raise ValueError("Extracted JSON must contain exactly two arrays")
                    algorithms_data = data[0]
                    recommendations_data = data[1]
                    algorithms = [AlgorithmEntry(**entry) for entry in algorithms_data]
                    recommendations = [Recommendation(**entry) for entry in recommendations_data]
                except json.JSONDecodeError as inner_e:
                    logger.error(f"Failed to parse extracted JSON: {inner_e}, Extracted: {json_text[:100]}...")
                    raise HTTPException(status_code=500,
                                        detail=f"Error: Failed to parse extracted AI response: {str(inner_e)}")
            else:
                logger.error(f"No valid JSON found in response: {response_text[:100]}...")
                raise HTTPException(status_code=500, detail=f"Error: Failed to parse AI response: {str(e)}")

        return BenchmarkResponse(
            algorithms=algorithms,
            recommendations=recommendations
        )

    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# Run the server
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)