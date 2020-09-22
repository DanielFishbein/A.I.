import random
import numpy as np

class pixel_node:
    def __init__(self,value,weight):
        self.value = value
        self.weight = weight


class hidden_col_1_node:
    def __init__(self,value,weight,b):
        self.value = value
        self.weight = weight
        self.bias = b


class hidden_col_2_node:
    def __init__(self,value,weight,b):
        self.value = value
        self.weight = weight
        self.bias = b


class output_node:
    def __init__(self,value,b):
        self.value = value
        self.bias = b




pixel_node.value = random.random()
a1 = pixel_node.value
pixel_node.weight = random.random()
a2 = pixel_node.weight

hidden_col_1_node.weight = random.random()
b1 = hidden_col_1_node.weight
hidden_col_1_node.bias = random.random()
z_1 = pixel_node.weight * pixel_node.value + hidden_col_1_node.bias
hidden_col_1_node.value = np.arctan(z_1)

hidden_col_2_node.weight = random.random()
hidden_col_2_node.bias = random.random()
#z_2 = hidden_col_1_node.weight * hidden_col_1_node.value + hidden_col_2_node.bias
z_2 = random.random()
hidden_col_2_node.value = np.arctan(z_2)

output_node.bias = random.random()
z_3 = hidden_col_2_node.weight * hidden_col_2_node.value + output_node.bias
output_node.value = np.arctan(z_3)

def arctan_prime(x):
    z = 1/(1 + x**2)
    return z




answer = 1
cost = (output_node.value - answer)**2
while abs(cost) > 0.000001:

    output_node.value = np.arctan(hidden_col_2_node.weight * hidden_col_2_node.value + output_node.bias)
    cost = (output_node.value - answer)**2


    
    cost_d_weight_1 = hidden_col_2_node.value*arctan_prime(z_3)*2*(output_node.value - answer)
    cost_d_bias_1 = arctan_prime(z_3)*2*(output_node.value - answer)

    hidden_col_2_node.weight = hidden_col_2_node.weight - cost_d_weight_1
    output_node.bias = output_node.bias - cost_d_bias_1



    cost_d_weight_2 = hidden_col_1_node.value*arctan_prime(z_2)*2*(hidden_col_2_node.value - output_node.value)
    cost_d_bias_2 = arctan_prime(z_2)*2*(hidden_col_2_node.value - output_node.value)

    hidden_col_1_node.weight = hidden_col_1_node.weight - cost_d_weight_2
    hidden_col_2_node.bias = hidden_col_2_node.bias - cost_d_bias_2



    cost_d_weight_3 = pixel_node.value*arctan_prime(z_1)*2*(hidden_col_1_node.value - hidden_col_2_node.value)
    cost_d_bias_3 = arctan_prime(z_1)*2*(hidden_col_1_node.value - hidden_col_2_node.value)

    pixel_node.weight = pixel_node.weight - cost_d_weight_3
    hidden_col_1_node.bias = hidden_col_1_node.bias - cost_d_bias_3



    print(cost)
