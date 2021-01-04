#!/usr/bin/env python
# coding: utf-8

# # **Assignment1 For Numpy**(piaic145383)

# Difficulty Level **Beginner**

# 1. Import the numpy package under the name np

# In[1]:


import numpy as np


# 2. Create a null vector of size 10 

# In[2]:


np.zeros([10,10])


# 3. Create a vector with values ranging from 10 to 49

# In[5]:


a=np.arange(10,50)


# 4. Find the shape of previous array in question 3

# In[6]:


np.shape(a)


# 5. Print the type of the previous array in question 3

# In[8]:


a.dtype


# 6. Print the numpy version and the configuration
# 

# In[11]:


# print(np.__version__)
np.version.version


# 7. Print the dimension of the array in question 3
# 

# In[12]:


np.ndim(a)


# 8. Create a boolean array with all the True values

# In[53]:


np.ones((5,5),dtype=bool)


# 9. Create a two dimensional array
# 
# 
# 

# In[40]:


# b=np.arange(10).reshape(2,5)
np.array([(1,2,3),(4,5,6)])


# 10. Create a three dimensional array
# 
# 

# In[51]:


c=np.arange(20).reshape(2,5,2)
# np.ndim(c)
c


# Difficulty Level **Easy**

# 11. Reverse a vector (first element becomes last)

# In[73]:


a=np.array([1,2,3,4,5])
np.flip(a,0)


# 12. Create a null vector of size 10 but the fifth value which is 1 

# In[68]:


c=np.zeros(11)
c[5] =1
c


# 13. Create a 3x3 identity matrix

# In[61]:


np.identity(3)


# 14. arr = np.array([1, 2, 3, 4, 5]) 
# 
# ---
# 
#  Convert the data type of the given array from int to float 

# In[84]:


x = np.array([1,2,3,4,5])

x.dtype


# In[86]:


y=x.astype('float32')
y


# 15. arr1 =          np.array([[1., 2., 3.],
# 
#                     [4., 5., 6.]])  
#                       
#     arr2 = np.array([[0., 4., 1.],
#      
#                    [7., 2., 12.]])
# 
# ---
# 
# 
# Multiply arr1 with arr2
# 

# In[3]:


arr1 = np.array([[1., 2., 3.],

            [4., 5., 6.]])
arr2 = np.array([[0., 4., 1.],

           [7., 2., 12.]])
arr1*arr2


# 16. arr1 = np.array([[1., 2., 3.],
#                     [4., 5., 6.]]) 
#                     
#     arr2 = np.array([[0., 4., 1.], 
#                     [7., 2., 12.]])
# 
# 
# ---
# 
# Make an array by comparing both the arrays provided above

# In[6]:


compare=arr1 == arr2
compare


# 17. Extract all odd numbers from arr with values(0-9)

# In[13]:


arr = np.arange(1,10,2)
arr


# 18. Replace all odd numbers to -1 from previous array

# In[15]:


arr = -1
arr


# 19. arr = np.arange(10)
# 
# 
# ---
# 
# Replace the values of indexes 5,6,7 and 8 to **12**

# In[28]:


arr = np.arange(10)
arr[[5,6,7,8]] = 12
arr


# 20. Create a 2d array with 1 on the border and 0 inside

# In[39]:


# np.array([(1,1,1),(0,0,0),(1,1,1)])
np.array([[1,1,1],
           [0,0,0],
           [1,1,1]
                  ])


# Difficulty Level **Medium**

# 21. arr2d = np.array([[1, 2, 3],
# 
#                     [4, 5, 6], 
# 
#                     [7, 8, 9]])
# 
# ---
# 
# Replace the value 5 to 12

# In[43]:


arr2d = np.array([[1, 2, 3],

            [4, 5, 6], 

            [7, 8, 9]])
arr2d[1][1] = 12
arr2d


# 22. arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
# 
# ---
# Convert all the values of 1st array to 64
# 

# In[54]:



arr3d = np.array([[[1, 2, 3], 
                   [4, 5, 6]], 
                  [[7, 8, 9], 
                   [10, 11, 12]]])
arr3d[0] = 64
arr3d


# 23. Make a 2-Dimensional array with values 0-9 and slice out the first 1st 1-D array from it

# In[69]:


c= np.arange(10).reshape(2,5)
c[0]


# 24. Make a 2-Dimensional array with values 0-9 and slice out the 2nd value from 2nd 1-D array from it

# In[70]:


c


# In[73]:


c[1][1]


# 25. Make a 2-Dimensional array with values 0-9 and slice out the third column but only the first two rows

# In[74]:


c


# In[87]:


# pending


# 26. Create a 10x10 array with random values and find the minimum and maximum values

# In[91]:


f= np.random.rand(10,10)
f


# In[92]:


np.min(f)


# In[93]:


np.max(f)


# 27. a = np.array([1,2,3,2,3,4,3,4,5,6]) b = np.array([7,2,10,2,7,4,9,4,9,8])
# ---
# Find the common items between a and b
# 

# In[98]:


a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])
np.intersect1d(a,b)


# 28. a = np.array([1,2,3,2,3,4,3,4,5,6])
# b = np.array([7,2,10,2,7,4,9,4,9,8])
# 
# ---
# Find the positions where elements of a and b match
# 
# 

# In[99]:


a = np.array([1,2,3,2,3,4,3,4,5,6]) 
b = np.array([7,2,10,2,7,4,9,4,9,8])


# In[102]:


# pending


# 29.  names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])  data = np.random.randn(7, 4)
# 
# ---
# Find all the values from array **data** where the values from array **names** are not equal to **Will**
# 

# In[109]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
data = np.random.randn(7, 4)


# In[110]:


# pending


# 30. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) data = np.random.randn(7, 4)
# 
# ---
# Find all the values from array **data** where the values from array **names** are not equal to **Will** and **Joe**
# 
# 

# In[ ]:


# pending


# Difficulty Level **Hard**

# 31. Create a 2D array of shape 5x3 to contain decimal numbers between 1 and 15.

# In[167]:


intg=np.arange(1,16).reshape(5,3)
intg


# In[169]:


decimal = intg.astype('float32')
decimal


# 32. Create an array of shape (2, 2, 4) with decimal numbers between 1 to 16.

# In[150]:


d=np.arange(1,17).reshape(2,2,4)
g = d.astype('float32')
g


# 33. Swap axes of the array you created in Question 32

# In[156]:


d=np.arange(1,17).reshape(2,2,4)
np.swapaxes(d,axis1=1, axis2=0)


# 34. Create an array of size 10, and find the square root of every element in the array, if the values less than 0.5, replace them with 0

# In[133]:


arr = np.arange(10)
np.sqrt(arr)


# 35. Create two random arrays of range 12 and make an array with the maximum values between each element of the two arrays

# In[8]:


arr1 = np.random.randn(12)
arr2 = np.random.randn(12)
maxArray = np.maximum(arr1,arr2)
maxArray


# 36. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
# 
# ---
# Find the unique names and sort them out!
# 

# In[127]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
a= np.unique(names)
np.sort(a)


# 37. a = np.array([1,2,3,4,5])
# b = np.array([5,6,7,8,9])
# 
# ---
# From array a remove all items present in array b
# 
# 

# In[177]:


a = np.array([1,2,3,4,5])
b = np.array([5,6,7,8,9])
index = [4]
a = np.delete(a, index)
a


# 38.  Following is the input NumPy array delete column two and insert following new column in its place.
# 
# ---
# sampleArray = numpy.array([[34,43,73],[82,22,12],[53,94,66]]) 
# 
# 
# ---
# 
# newColumn = numpy.array([[10,10,10]])
# 

# In[190]:


sampleArray = np.array([[34,43,73],[82,22,12],[53,94,66]])
newColumn = np.array([[10,10,10]])
delet= np.delete(sampleArray,2,1)
delet


# In[194]:


# np.append(delet,newColumn,axis=1)    
# trying but comes error


# 39. x = np.array([[1., 2., 3.], [4., 5., 6.]]) y = np.array([[6., 23.], [-1, 7], [8, 9]])
# 
# 
# ---
# Find the dot product of the above two matrix
# 

# In[116]:


x = np.array([[1., 2., 3.], [4., 5., 6.]])
y = np.array([[6., 23.], [-1, 7], [8, 9]])

np.dot(x,y)


# 40. Generate a matrix of 20 random values and find its cumulative sum

# In[113]:


a = np.random.randn(20)
np.cumsum(a)


# In[ ]:




