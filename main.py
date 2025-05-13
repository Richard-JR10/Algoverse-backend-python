import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
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
origins = ["http://localhost:5174","http://localhost:5173","https://algoverse1.netlify.app/"]
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

            # Mark pivot as sorted
            steps.append({
                "type": "sorted",
                "index": i + 1
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