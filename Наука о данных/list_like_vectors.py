# summarize two vectors

def vector_add(v, w):
    return [v_i + w_i for v_i, w_i in zip(v, w)]

# v subtract w

def vector_substract(v, w):
    return [v_i - w_i for v_i, w_i in zip(v, w)]

# [1] = v[1] + w[1]...+z[1]
# [2] = v[2] + w[3]...+z[4]
# ...
# [n] = v[n] + w[n]...+z[n]
# return [[1], [2], ..., [n]]

def vector_sum(vectors):
    #return reduce(vector_add, vectors)

    result = vectors[0]

    for vector in vectors[1:]:
        result = vector_add(result, vector)
    return result

# vector and scalar multiplying

def scalar_multiply(c, v):
    return [c * v_i for v_i in v]

# component-wise average of a list of vectors of the same size

def vector_mean(vectors):
    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))

# scalar product of vectors

def dot(v, w):
    return sum(v_i * w_i for v_i, w_i in zip(v, w))

# sum of squares

def sum_of_squares(vector):
    return dot(vector, vector)

# scalar vector length

def magnitude(vector):
    return math.sqrt(sum_of_squares(vector))

# squared distance between vectors

def squared_distance(v, w):
    return sum_of_squares(vector_substract(v, w))

# distance

def distance(v, w):
    return math.sqrt(squared_distance(v, w))
    #return magnitude(vector_substract(v, w))
