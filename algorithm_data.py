"""
Algorithm Knowledge Base Creator
Creates structured algorithm data for the learning assistant
"""

import json
import os

def create_algorithm_knowledge_base():
    """
    Creates comprehensive algorithm knowledge base with explanations,
    code examples, and practice problems
    """
    
    algorithms = {
        "sorting": {
            "quicksort": {
                "name": "Quicksort",
                "category": "sorting",
                "difficulty": "intermediate",
                "time_complexity": {
                    "best": "O(n log n)",
                    "average": "O(n log n)", 
                    "worst": "O(n²)"
                },
                "space_complexity": "O(log n)",
                "description": "Quicksort is a divide-and-conquer algorithm that picks a pivot element and partitions the array around it, then recursively sorts the subarrays.",
                "how_it_works": [
                    "Choose a pivot element from the array",
                    "Partition the array so elements smaller than pivot go left, larger go right",
                    "Recursively apply quicksort to left and right subarrays",
                    "Base case: arrays with 0 or 1 element are already sorted"
                ],
                "pseudocode": """
function quicksort(arr, low, high):
    if low < high:
        pivot = partition(arr, low, high)
        quicksort(arr, low, pivot - 1)
        quicksort(arr, pivot + 1, high)

function partition(arr, low, high):
    pivot = arr[high]
    i = low - 1
    for j = low to high - 1:
        if arr[j] <= pivot:
            i = i + 1
            swap arr[i] with arr[j]
    swap arr[i + 1] with arr[high]
    return i + 1
                """,
                "implementations": {
                    "python": '''
def quicksort(arr, low=0, high=None):
    """
    Sorts array using quicksort algorithm
    Args: arr - list to sort, low - start index, high - end index
    """
    if high is None:
        high = len(arr) - 1
    
    if low < high:
        # Partition and get pivot index
        pivot = partition(arr, low, high)
        
        # Recursively sort left and right subarrays
        quicksort(arr, low, pivot - 1)
        quicksort(arr, pivot + 1, high)

def partition(arr, low, high):
    """Partitions array around pivot element"""
    pivot = arr[high]  # Choose last element as pivot
    i = low - 1        # Index of smaller element
    
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

# Usage example:
arr = [64, 34, 25, 12, 22, 11, 90]
quicksort(arr)
print(f"Sorted array: {arr}")
                    ''',
                    "java": '''
public class QuickSort {
    public static void quicksort(int[] arr, int low, int high) {
        if (low < high) {
            int pivot = partition(arr, low, high);
            quicksort(arr, low, pivot - 1);
            quicksort(arr, pivot + 1, high);
        }
    }
    
    private static int partition(int[] arr, int low, int high) {
        int pivot = arr[high];
        int i = low - 1;
        
        for (int j = low; j < high; j++) {
            if (arr[j] <= pivot) {
                i++;
                swap(arr, i, j);
            }
        }
        swap(arr, i + 1, high);
        return i + 1;
    }
    
    private static void swap(int[] arr, int i, int j) {
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
}
                    '''
                },
                "advantages": [
                    "Average case O(n log n) performance",
                    "In-place sorting (low memory usage)",
                    "Cache-efficient due to good locality",
                    "Widely used in practice"
                ],
                "disadvantages": [
                    "Worst case O(n²) performance",
                    "Not stable (doesn't preserve relative order)",
                    "Performance depends on pivot selection"
                ],
                "related_algorithms": ["mergesort", "heapsort", "bubble_sort"],
                "practice_problems": [
                    "Implement quicksort with random pivot selection",
                    "Modify quicksort to handle duplicate elements efficiently",
                    "Implement 3-way partitioning for quicksort"
                ]
            },
            
            "mergesort": {
                "name": "Merge Sort",
                "category": "sorting",
                "difficulty": "intermediate",
                "time_complexity": {
                    "best": "O(n log n)",
                    "average": "O(n log n)",
                    "worst": "O(n log n)"
                },
                "space_complexity": "O(n)",
                "description": "Merge sort is a stable, divide-and-conquer sorting algorithm that divides the array into halves, sorts them recursively, and merges the sorted halves.",
                "how_it_works": [
                    "Divide the array into two halves",
                    "Recursively sort both halves",
                    "Merge the two sorted halves back together",
                    "Base case: arrays with 0 or 1 element are already sorted"
                ],
                "implementations": {
                    "python": '''
def mergesort(arr):
    """
    Sorts array using merge sort algorithm
    Args: arr - list to sort
    Returns: sorted list
    """
    if len(arr) <= 1:
        return arr
    
    # Divide array into two halves
    mid = len(arr) // 2
    left = mergesort(arr[:mid])
    right = mergesort(arr[mid:])
    
    # Merge sorted halves
    return merge(left, right)

def merge(left, right):
    """Merges two sorted arrays into one sorted array"""
    result = []
    i = j = 0
    
    # Compare elements and merge in sorted order
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    # Add remaining elements
    result.extend(left[i:])
    result.extend(right[j:])
    
    return result
                    '''
                }
            }
        },
        
        "searching": {
            "binary_search": {
                "name": "Binary Search",
                "category": "searching", 
                "difficulty": "beginner",
                "time_complexity": {
                    "best": "O(1)",
                    "average": "O(log n)",
                    "worst": "O(log n)"
                },
                "space_complexity": "O(1)",
                "description": "Binary search finds a target value in a sorted array by repeatedly dividing the search space in half.",
                "how_it_works": [
                    "Start with the middle element of sorted array",
                    "If target equals middle element, return index",
                    "If target is less than middle, search left half",
                    "If target is greater than middle, search right half",
                    "Repeat until found or search space is empty"
                ],
                "implementations": {
                    "python": '''
def binary_search(arr, target):
    """
    Searches for target in sorted array using binary search
    Args: arr - sorted list, target - value to find
    Returns: index of target or -1 if not found
    """
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1  # Target not found

# Usage example:
arr = [2, 3, 4, 10, 40]
target = 10
result = binary_search(arr, target)
print(f"Element found at index: {result}")
                    '''
                }
            }
        },
        
        "data_structures": {
            "binary_tree": {
                "name": "Binary Tree",
                "category": "data_structures",
                "difficulty": "intermediate", 
                "description": "A binary tree is a hierarchical data structure where each node has at most two children, referred to as left and right child.",
                "operations": {
                    "insertion": "O(log n) average, O(n) worst",
                    "deletion": "O(log n) average, O(n) worst", 
                    "search": "O(log n) average, O(n) worst"
                },
                "implementations": {
                    "python": '''
class TreeNode:
    """Node class for binary tree"""
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class BinaryTree:
    """Binary Tree implementation"""
    def __init__(self):
        self.root = None
    
    def insert(self, val):
        """Insert value into binary search tree"""
        if not self.root:
            self.root = TreeNode(val)
        else:
            self._insert_recursive(self.root, val)
    
    def _insert_recursive(self, node, val):
        """Helper method for recursive insertion"""
        if val < node.val:
            if not node.left:
                node.left = TreeNode(val)
            else:
                self._insert_recursive(node.left, val)
        else:
            if not node.right:
                node.right = TreeNode(val)
            else:
                self._insert_recursive(node.right, val)
    
    def inorder_traversal(self, node=None, result=None):
        """Inorder traversal: left -> root -> right"""
        if result is None:
            result = []
        if node is None:
            node = self.root
            
        if node:
            self.inorder_traversal(node.left, result)
            result.append(node.val)
            self.inorder_traversal(node.right, result)
            
        return result
                    '''
                }
            }
        }
    }
    
    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Save algorithms data
    with open('data/algorithms.json', 'w') as f:
        json.dump(algorithms, f, indent=4)
    
    # Create practice problems database
    practice_problems = {
        "sorting": [
            {
                "id": 1,
                "difficulty": "easy",
                "problem": "Sort the array [64, 34, 25, 12, 22, 11, 90] using quicksort. Show each step of the partitioning process.",
                "hint": "Choose the last element as pivot and show how elements are rearranged",
                "topic": "quicksort"
            },
            {
                "id": 2, 
                "difficulty": "medium",
                "problem": "Implement a stable version of quicksort. What modifications are needed?",
                "hint": "Consider how to maintain relative order of equal elements",
                "topic": "quicksort"
            },
            {
                "id": 3,
                "difficulty": "easy", 
                "problem": "Trace through merge sort on array [38, 27, 43, 3, 9, 82, 10]. Show the divide and merge steps.",
                "hint": "Draw the recursion tree showing how array is split and merged",
                "topic": "mergesort"
            }
        ],
        "searching": [
            {
                "id": 4,
                "difficulty": "easy",
                "problem": "Use binary search to find element 22 in sorted array [1, 3, 5, 7, 9, 11, 14, 16, 22, 25, 30]. Show each comparison.",
                "hint": "Start from middle element and show how search space is halved",
                "topic": "binary_search"
            }
        ],
        "data_structures": [
            {
                "id": 5,
                "difficulty": "medium",
                "problem": "Insert the values [5, 3, 7, 2, 4, 6, 8] into a binary search tree. Draw the final tree structure.",
                "hint": "Start with empty tree and insert one value at a time following BST property",
                "topic": "binary_tree"
            }
        ]
    }
    
    # Save practice problems
    with open('data/problems.json', 'w') as f:
        json.dump(practice_problems, f, indent=4)
    
    print("Algorithm knowledge base created successfully!")
    print("Files created:")
    print("- data/algorithms.json")
    print("- data/problems.json")

if __name__ == "__main__":
    create_algorithm_knowledge_base()