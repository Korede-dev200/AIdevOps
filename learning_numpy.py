import numpy as np

# CREATING ARRAYS
# Creating a ndarray using the array function (We are passing a list) This is also the same as a 1D Array
arr = np.array([1, 2, 3, 4, 5])

print(arr)
print(type(arr))
# Print a free line to seperate your results in the terminal
print("\n")

# Use a tuple to create a NumPy array
arr = np.array((1, 2, 3, 4, 5))

print(arr)
# Print a free line to seperate your results in the terminal
print("\n")

# 0-D arrays, or Scalars, are the elements in an array. Each value in an array is a 0-D array.
# Create a 0-D array with value 42
arr = np.array(42)

print(arr)
# Print a free line to seperate your results in the terminal
print("\n")

# An array that has 1-D arrays as its elements is called a 2-D array.
# These are often used to represent matrix or 2nd order tensors.
# Create a 2-D array containing two arrays with the values 1,2,3 and 4,5,6
arr = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

print(arr)
# Print a free line to seperate your results in the terminal
print("\n")

# An array that has 2-D arrays (matrices) as its elements is called 3-D array.
# These are often used to represent a 3rd order tensor
# Create a 3-D array with two 2-D arrays, both containing two arrays with the values 1,2,3 and 4,5,6
arr = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])

print(arr)
# Print a free line to seperate your results in the terminal
print("\n")

# NumPy Arrays provides the ndim attribute that returns an integer that tells us how many dimensions the array have.
# Check how many dimensions the arrays have
a = np.array(42)
b =  np.array([1, 2, 3, 4, 5])
c = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
d = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])

print(a.ndim)
print(b.ndim)
print(c.ndim)
print(d.ndim)
# Print a free line to seperate your results in the terminal
print("\n")

# Higher Dimensional Arrays
# An array can have any number of dimensions
# When the array is created, you can define the number of dimensions by using the ndmin argument
# Create an array with 5 dimensions and verify that it has 5 dimensions
arr = np.array([1, 2, 3, 4], ndmin=5)

print(arr)
print("Number of dimensions: ", arr.ndim)
# Print a free line to seperate your results in the terminal
print("\n") 






# NUMPY ARRAY INDEXING
# Access Array Elements
# Array indexing is the same as accessing an array element
# You can access an array element by referring to its index number
# The indexes in NumPy arrays start with 0, meaning that the first element has index 0, and the second has index 1 etc
# Get the first element from the following array
arr = np.array([1, 2, 3, 4])

print(arr[1])
# Print a free line to seperate your results in the terminal
print("\n") 

# Get third and fourth elements from the following array and add them
arr = np.array([1, 2, 3, 4])

print(arr[1] + arr[3])
# Print a free line to seperate your results in the terminal
print("\n") 

# Access 2-D Arrays
# To access elements from 2-D arrays we can use comma separated integers representing the dimension and the index of the element
# Think of 2-D arrays like a table with rows and columns, where the dimension represents the row and the index represents the column
# Access the element on the first row, second column
arr = np.array([
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10]
])

print("2nd element on the first row: ", arr[0, 1])
# Print a free line to seperate your results in the terminal
print("\n")

# Access the element on the 2nd row, 5th column
arr = np.array([
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10]
])

print("5th element on the 2nd row: ", arr[1, 4])
# Print a free line to seperate your results in the terminal
print("\n")

# Access 3-D Arrays
# To access elements from 3-D arrays we can use comma separated integers representing the dimensions and the index of the element
# Access the third element of the second array of the first array
arr = np.array([[
    [1, 2, 3],
    [4, 5, 6]],
    [
        [7, 8, 9],
        [10, 11, 12]
]])

print(arr[0, 1, 2])
# Print a free line to seperate your results in the terminal
print("\n")

# Negative Indexing
# Use negative indexing to access an array from the end
# Print the last element from the 2nd dim
arr = np.array([
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10]
])

print("Last element from second dim:", arr[1, -1])
# Print a free line to seperate your results in the terminal
print("\n")




# NUMPY ARRAY SLICING
# Slicing in python means taking elements from one given index to another given index.
# We pass slice instead of index like this: [start:end].
# We can also define the step, like this: [start:end:step].
# If we don't pass start its considered 0
# If we don't pass end its considered length of array in that dimension
# If we don't pass step its considered 1
# Slice elements from index 1 to index 5 from the following array
arr = np.array([1, 2, 3, 4, 5, 6, 7])

print(arr[1:5])
# Note: The result includes the start index, but excludes the end index
# Print a free line to seperate your results in the terminal
print("\n")

# Slice elements from index 4 to the end of the array
arr = np.array([1, 2, 3, 4, 5, 6, 7])

print(arr[4:])
# Print a free line to seperate your results in the terminal
print("\n")

# Slice elements from the beginning to index 4 (not included)
arr = np.array([1, 2, 3, 4, 5, 6, 7])

print(arr[:4])
# Print a free line to seperate your results in the terminal
print("\n")

# Negative Slicing
# Use the minus operator to refer to an index from the end
# Slice from the index 3 from the end to index 1 from the end
arr = np.array([1, 2, 3, 4, 5, 6, 7])

print(arr[-3:-1])
# Print a free line to seperate your results in the terminal
print("\n")

# STEP
# Use the step value to determine the step of the slicing
# Return every other element from index 1 to index 5
arr = np.array([1, 2, 3, 4, 5, 6, 7])

print(arr[1:5:2])
# Print a free line to seperate your results in the terminal
print("\n")

# Return every other element from the entire array
arr = np.array([1, 2, 3, 4, 5, 6, 7])

print(arr[::2])
# Print a free line to seperate your results in the terminal
print("\n")

# Slicing 2-D Arrays
# From the second element, slice elements from index 1 to index 4 (not included
arr = np.array([
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10]
])

print(arr[1, 1:4])
# Print a free line to seperate your results in the terminal
print("\n")

# From both elements, return index 2
arr = np.array([
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10]
])

print(arr[0:2, 1])
# Print a free line to seperate your results in the terminal
print("\n")

# From both elements, slice index 1 to index 4 (not included), this will return a 2-D array
arr = np.array([
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10]
])

print(arr[0:2, 1:4])
# Print a free line to seperate your results in the terminal
print("\n")