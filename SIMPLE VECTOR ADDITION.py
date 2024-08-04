import tensorflow as tf
scalar = tf.constant(7)
vector = tf.constant([10, 10])
print("Vector dimensions:", vector.ndim)
matrix = tf.constant([[1, 2], [3, 4]])
print("Matrix dimensions:", matrix.ndim)
matrix1 = tf.constant([[2, 4], [6, 8]])
result = matrix + matrix1
print("Addition of two matrices:")
print(result)
