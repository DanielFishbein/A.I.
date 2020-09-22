

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
