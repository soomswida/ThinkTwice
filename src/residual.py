import numpy as np
import numpy.linalg as LA

# Let's implement the residual distribution 
# from (Yaniv Leviathan et al., 2023).
# p'(x) = norm(max(0, p(x) - q(x)))


# Declare functions for this
def acceptedDist(p_dist, q_dist):
    accepted = np.zeros(len(p_dist))

    # min(p(x), q(x))
    for x in range(len(p_dist)):
        accepted[x] = min(p_dist[x], q_dist[x])

    return accepted

def residualDist(p_dist, q_dist):
    residual = np.zeros(len(p_dist))

    # max(0, p(x)- q(x))
    for x in range(len(p_dist)):
        residual[x] = p_dist[x] - q_dist[x]
        residual[x] = max(0, residual[x])
    
    # norm(...)
    norm = LA.norm(residual)
    residual = residual / norm    

    return residual

def getFinalDist(p_dist, q_dist):
    accepted = acceptedDist(p_dist, q_dist)
    residual = residualDist(p_dist, q_dist)
    
    return np.add(accepted, residual) 


# Declare two vectors for sake of the experiment
p = np.array([0.6, 0.3, 0.1])
q = np.array([0.5, 0.45, 0.05])

print(f"p={p}")
print(f"q={q}")

print("The result is:")
print(getFinalDist(p, q))



