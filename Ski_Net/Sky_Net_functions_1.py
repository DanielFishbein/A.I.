import numpy as np
import random
import math
import gzip
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import sys


from Sky_Net_classes_1 import pixel_node
from Sky_Net_classes_1 import hidden_col_1_node
from Sky_Net_classes_1 import hidden_col_2_node
from Sky_Net_classes_1 import output_node


'''
** arctan function to squish values down to between
** 0 and 1.
** There was no reason for this specific choice other
** than it quishes number down to between 0 and 1
** Returns a number between 0 and 1
'''

def arctan_prime(z):
    return 1/((1 + z**2))



#This function initializes all the class with random values
#returns a list of arrays containing all the nodes in order of [input, hidden_1, hidden_2, output]
def initials_classes(pixels):
    num_nodes_pixel = 784
    num_nodes_hidden_1 = 16
    num_nodes_hidden_2 = 16
    num_nodes_output = 10
    '''
    ***************  LAYER 1  **********************
    ** Here I create a pixel node for each pixel value.
    ** I then put each pixel value in its corisponding
    ** pixel node.
    ** Each pixel also gets an array of 16 weights
    ** corisponding to each of the 16 nodes in
    ** the next layer.
    ** I then put all the pixel nodes into an array
    '''
    array_pixel_node = np.array([])
    for i in pixels:
        weight = np.random.random(num_nodes_hidden_1)
        new_node = pixel_node(i,weight)
        array_pixel_node = np.append(array_pixel_node, new_node)


    '''
    ***************  LAYER 2  **********************
    ** Here I create a 16 hidden nodes.
    ** I then give each hidden node a random "value", bias
    ** and a random array of 16 weight for each
    ** hidden node in the next layer.
    ** I then put all these hidden nodes into an array.
    '''
    array_1_hidden_node = np.array([])
    for i in range(0,num_nodes_hidden_1):
        value = np.random.random()
        bias = np.random.random()
        weight = np.random.random(num_nodes_hidden_2)
        new_hid_node = hidden_col_1_node(value,weight,bias)
        array_1_hidden_node = np.append(array_1_hidden_node, new_hid_node)


    '''
    ***************  LAYER 3  **********************
    ** Here I create 16 hidden nodes again.
    ** I then give each hidden node a random "value", bias
    ** and a random array of 10 weight for each
    ** output node in the next layer.
    ** I then put all these hidden nodes into an array.
    '''
    array_2_hidden_node = np.array([])
    for i in range(0,num_nodes_hidden_2):
        value = np.random.random()
        bias = np.random.random()
        weight = np.random.random(num_nodes_output)
        new_hid_node = hidden_col_2_node(value,weight,bias)
        array_2_hidden_node = np.append(array_2_hidden_node, new_hid_node)


    '''
    ***************  LAYER 4  **********************
    ** Here I create a 10 output nodes.
    ** I then give each output node a random "value" and
    ** put all these output nodes into an array.
    '''
    array_output_node = np.array([])
    for i in range(0,num_nodes_output):
        value = np.random.random()
        bias = np.random.random()
        new_output_node = output_node(value,bias)
        array_output_node = np.append(array_output_node, new_output_node)

    return array_pixel_node, array_1_hidden_node, array_2_hidden_node, array_output_node




'''
** This function is given a list of node arrays and goes through
** each of the layers, starting with array_1_hidden_node, and computes
** the weighted sum of each node.  After each new value is
** calculated it is updated in the specified node
** Returns a number that the computer thinks was inputed
** based on the largest value in the output nodes
'''
def guess_number(node_arrays):

    array_pixel_node = node_arrays[0]         # array_pixel_node
    array_1_hidden_node = node_arrays[1]      # array_1_hidden_node
    array_2_hidden_node = node_arrays[2]      # array_2_hidden_node
    array_output_node = node_arrays[3]        # array_output_node

    num_nodes_pixel = 784
    num_nodes_hidden_1 = 16
    num_nodes_hidden_2 = 16
    num_nodes_output = 10

    '''
    ***************  LAYER 2  **********************
    ** I pass the weighted_sum_of_node() function the indicy
    ** that I am on, the node that is being calculated, and
    ** the previous array of nodes.
    ** In this case, it is the array of pixels.
    ** The weighted sum is then fed into the arctan()
    ** function to return a value between 0 and 1.
    ** The node that is bing worked on then has its value
    ** updated.
    ** This prossess is repeated for each of the other node arrays
    '''

    for i in range(0,num_nodes_hidden_1):
        x = weighted_sum_of_node(i, array_1_hidden_node[i], array_pixel_node)
        array_1_hidden_node[i].value = np.arctan(x)


    '''***************  LAYER 3  **********************'''
    for i in range(0,num_nodes_hidden_2):
        x = weighted_sum_of_node(i, array_2_hidden_node[i], array_1_hidden_node)
        array_2_hidden_node[i].value = np.arctan(x)


    '''***************  LAYER 4  **********************'''
    #This one also adds all the calculated values to an array
    list_values = np.array([])
    for i in range(0,num_nodes_output):
        x = weighted_sum_of_node(i, array_output_node[i], array_2_hidden_node)
        arctan = np.arctan(x)
        array_output_node[i].value = arctan
        list_values = np.append(list_values, arctan)
    #    list.append(array_output_node[i].value)
    return list_values                                                          ##Array of values of each guess.  Think of this as how sure the computer is on each number




'''
** This function opens and gathers all the emnist data.
** I don't know how it extracts and converts the data.
** But I do know that it put it in a useable format, that
** I can use.  Check the URL if you want to see where the
** code comes from.  The function returns an array of lists.
** The outer array is the collection of all the images and
** labels.  The inner list is a pair of image and label
** (in the order of [image,label]).
** Images are arrays of digets. Labels are an array of 1 diget
'''
#https://stackoverflow.com/questions/40427435/extract-images-from-idx3-ubyte-file-or-gzip-via-python
def get_training_data(num_pics,size):
    num_examples = num_pics                                            #number of examples to import
    image_size = size                                                 #image size (X by X)
    num_images = num_pics                                               #number of labeled images to import
    dumb_list = [0,0]                                               #initializing dummy list
    array_image_exam = []                                 #initalizing array for images and labels

    f1 = gzip.open('train-images-idx3-ubyte.gz','r')                #opening images in read mode
    f2 = gzip.open('train-labels-idx1-ubyte.gz','r')                #opening labels in read mode

    f1.read(16)                                                     #????
    buf = f1.read(image_size * image_size * num_images)             #????
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)    #????
    data = data.reshape(num_images, image_size, image_size, 1)      #????

    f2.read(8)                                                      #????
    for i in range(0,num_examples):                                 #loop to steap through each example
        buf = f2.read(1)                                            #????
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)#????

        image = (np.asarray(data[i])/255).flatten()                 #fitting data to an fattened array
        #image = np.asarray(data[i]).squeeze()                      #to view graph, uncomment this and comment the .flatten() line above

        dumb_list = np.array([image,labels])                                         #putting image in first place of dumb list
        array_image_exam.append(dumb_list)   #adding the paired list of image and label list to an array
    return(array_image_exam)                                        #return an array with all 60,000 pairs in it



def get_training_data_prime(num_pics,size):

    num_examples = num_pics                                            #number of examples to import
    image_size = size                                                 #image size (X by X)
    num_images = num_pics                                               #number of labeled images to import
    dumb_list = [0,0]                                               #initializing dummy list
    array_image_exam = []                                 #initalizing array for images and labels

    f1 = gzip.open('train-images-idx3-ubyte.gz','r')                #opening images in read mode
    f2 = gzip.open('train-labels-idx1-ubyte.gz','r')                #opening labels in read mode

    f1.read(16)                                                     #????
    buf = f1.read(image_size * image_size * num_images)             #????
    data = np.frombuffer(buf, dtype=np.uint8)#.astype(np.float32)    #????
    data = data.reshape(num_images, image_size, image_size, 1)      #????

    f2.read(8)                                                      #????
    for i in range(0,num_examples):                                 #loop to steap through each example
        buf = f2.read(1)                                            #????
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)#????

        #image = (np.asarray(data[i])/255).flatten()                 #fitting data to an fattened array
        image = np.asarray(data[i]).squeeze()                      #to view graph, uncomment this and comment the .flatten() line above

        dumb_list = np.array([image,labels])                                         #putting image in first place of dumb list
        array_image_exam.append(dumb_list)   #adding the paired list of image and label list to an array
    return(array_image_exam)                                        #return an array with all 60,000 pairs in it



'''
** This function takes in the node number, the node,
** and the array of nodes of hte previous layer.
** It extracts the bias of the node it is looking at,
** and then goes through the array of nodes in the
** previous layer to get the value of the other nodes
** as well as the corispoing weight for this node.
** The weights and biases are multiplied together and
** then added to a growing sum.
** The final sum is then added with the bias and returned.
** Returns a Decimal of some value
'''
def weighted_sum_of_node(node_num, node, previous_layer_array):
    b = node.bias
    sum = 0
    for i in range(0,previous_layer_array.size):
        a = previous_layer_array[i].value
        w = previous_layer_array[i].weight[node_num]
        aw = a*w
        sum = sum + aw
    weighted_sum = sum + b
    return weighted_sum




'''
This function is given a list of 4
arrays of classes, unpacts the data
within each class and writes it down.
Returns: nothing
'''

def save(node_arrays):

    array_pixel_node = node_arrays[0]                           #breaking up the list of arrays
    array_1_hidden_node = node_arrays[1]
    array_2_hidden_node = node_arrays[2]
    array_output_node = node_arrays[3]

    writer = open("Sky_Net_data.txt",'w')                       #opens a file in write mode
    try:                                                        #syntax for if someting doesnt work


        #write pixel weights
        writer.write('pixel weights:'+ '\n')                    #titel the section   ('\n' means new line)
        for num_pixel_node in array_pixel_node:                 #step through each pixel node
            txt_weight = num_pixel_node.weight[:]               #create a copy array of the node's weights
            txt_weight_str = str(txt_weight.tolist())           #turns that copy into a list and then into a string
            writer.write(txt_weight_str + '\n')                 #witng down the stringed list in the file
        writer.write('\n')                                      #give a break line in the file


        #write hidden_col_1 value,weight,bias
        writer.write('hidden_col_1 value,weight,bias:'+ '\n')   #title the section
        for num_hidden_col_1_node in array_1_hidden_node:       #step through each hidden col 1 node
            txt_value = str(num_hidden_col_1_node.value)        #get the value of the node and then turn that Decimal into a string
            writer.write(txt_value + '\n')                      #writng down the stringed Decimal in the file

            txt_weight = num_hidden_col_1_node.weight[:]        #create a copy array of the node's weights
            txt_weight_str = str(txt_weight.tolist())           #turns that copy into a list and then into a string
            writer.write(txt_weight_str + '\n')                 #witng down the stringed list in the file

            txt_bias = str(num_hidden_col_1_node.bias)          #get the bias of the node and then turn that Decimal into a string
            writer.write(txt_bias + '\n')                       #writng down the stringed Decimal in the file
        writer.write('\n')                                      #give a break line in the file


        #write hidden_col_2 value,weight,bias:
        writer.write('hidden_col_2 value,weight,bias:'+ '\n')    #repete prossess of hidden col 1
        for num_hidden_col_2_node in array_2_hidden_node:
            txt_value = str(num_hidden_col_2_node.value)
            writer.write(txt_value + '\n')

            txt_weight = num_hidden_col_2_node.weight[:]
            txt_weight_str = str(txt_weight.tolist())
            writer.write(txt_weight_str + '\n')

            txt_bias = str(num_hidden_col_2_node.bias)
            writer.write(txt_bias + '\n')
        writer.write('\n')


        #write output value,bias:
        writer.write('output_node value,bias:'+ '\n')
        for num_output_node in array_output_node:               #step through each output node
            txt_value = str(num_output_node.value)              #get the value of the node and then turn that Decimal into a string
            writer.write(txt_value + '\n')                      #writng down the stringed Decimal in the file

            txt_value = str(num_output_node.bias)               #get the bias of the node and then turn that Decimal into a string
            writer.write(txt_value + '\n')                      #writng down the stringed Decimal in the file

    finally:                                                    #syntax for if something went wronge
        writer.close()                                          #clsoe the file
    return






'''
This function is given a list of 4
arrays of classes, and packs the data
from the file within each class.
Returns: nothing
'''
def load(node_arrays):
    array_pixel_node = node_arrays[0]                           #unpackign the list
    array_1_hidden_node = node_arrays[1]
    array_2_hidden_node = node_arrays[2]
    array_output_node = node_arrays[3]

    reader = open("Sky_Net_data.txt",'r')                #open the file in read mode
    try:                                                        #syntax for if someting when wrong in opeing the file
        #read pixel weights
        line_1 = reader.readline().strip('\n')                  #reading the first line and stripping the '\n' off it
        for num_pixel_node in array_pixel_node:                 #for loop to step through each pixel node
            line_weight = reader.readline().strip('\n')         #reading the line in the file and stripping the '\n' off it
            num_pixel_node.weight = np.array(eval(line_weight)) #converting the string of the list into a useable list and then converting that list into a np.array() and then packing it into its respective node weight
        line_end = reader.readline().strip('\n')                #reading the space between the sections

        #read hidden_col_1
        line_1 = reader.readline().strip('\n')                  #reading the first line and stripping the '\n' off it
        for num_hidden_col_1_node in array_1_hidden_node:       #for loop to step through each hidden 1 node
            line_value = reader.readline().strip('\n')          #reading the line in the file and stripping the '\n' off it
            num_hidden_col_1_node.value = float(line_value)     #convertingthe string into a float and pakcing that float into its respective class value

            line_weight = reader.readline().strip('\n')         #reading the line in the file and stripping the '\n' off it
            num_hidden_col_1_node.weight = np.array(eval(line_weight))  #converting the string of the list into a useable list and then converting that list into a np.array() and then packing it into its respective node weight

            line_bias = reader.readline().strip('\n')           #reading the line in the file and stripping the '\n' off it
            num_hidden_col_1_node.bias = float(line_bias)       #convertingthe string into a float and pakcing that float into its respective class bias
        line_end = reader.readline().strip('\n')                #reading the space between the sections


        #read hidden_col_2
        line_1 = reader.readline().strip('\n')                 #repete prossess of hidden col 1
        for num_hidden_col_2_node in array_2_hidden_node:
            line_value = reader.readline().strip('\n')
            num_hidden_col_2_node.value = float(line_value)

            line_weight = reader.readline().strip('\n')
            num_hidden_col_2_node.weight = np.array(eval(line_weight))

            line_bias = reader.readline().strip('\n')
            num_hidden_col_2_node.bias = float(line_bias)
        line_end = reader.readline().strip('\n')


        #read output
        line_1 = reader.readline().strip('\n')                #reading the first line and stripping the '\n' off it
        for num_output_node in array_output_node:             #step through each output node
            line_value = reader.readline().strip('\n')        #reading the line in the file and stripping the '\n' off it
            num_output_node.value = float(line_value)         #convertingthe string into a float and pakcing that float into its respective class value

            line_bias = reader.readline().strip('\n')         #reading the line in the file and stripping the '\n' off it
            num_output_node.bias = float(line_bias)           #convertingthe string into a float and pakcing that float into its respective class bias

    finally:                                                  #syntax for if someting when wrong in opeing the file
        reader.close()                                        #close the file
    return



def back_prop(node_arrays,error_array):
    array_pixel_node = node_arrays[0]                           #unpackign the list
    array_1_hidden_node = node_arrays[1]
    array_2_hidden_node = node_arrays[2]
    array_output_node = node_arrays[3]

    #dc/dw = a_k * arctan_prime(z_j) * 2(a_j - y_j)
    #dc/dw = a_k * arctan_prime(z_j) * dc/dw

    #dc/db = arctan_prime(z_j) * 2(a_j - y_j)
    #dc/db = arctan_prime(z_j) * dc/db


    ####################### 1 ###########
    output = []
    for j in range(0,len(array_output_node)):
        output.append(array_output_node[j].value)
    error = output - error_array

    f_prime = []
    for j in range(0, len(array_output_node)):       #current layer
        z_3 = 0
        for k in range(0, len(array_2_hidden_node)): #Back layer
            z_3 = z_3 + array_2_hidden_node[k].weight[j] * array_2_hidden_node[k].value
        z_3 = z_3 + array_output_node[j].bias
        f_prime.append(arctan_prime(z_3))
    delta = np.multiply(error, f_prime)

    for j in range(0,len(array_2_hidden_node)):
        weight = np.array(array_2_hidden_node[j].weight)

        weight = weight.transpose()
        final = weight*delta
        array_2_hidden_node[j].weight = array_2_hidden_node[j].weight - final

    for j in range(0, len(array_output_node)):
        array_output_node[j].bias = array_output_node[j].bias - delta[j]



    ####################### 2 ###########
    output = []
    for j in range(0,len(array_2_hidden_node)):
        output.append(array_2_hidden_node[j].value)
    error = output


    f_prime = []
    for j in range(0, len(array_2_hidden_node)):       #current layer
        z_3 = 0
        for k in range(0, len(array_1_hidden_node)): #Back layer
            z_3 = z_3 + array_1_hidden_node[k].weight[j] * array_1_hidden_node[k].value
        z_3 = z_3 + array_2_hidden_node[j].bias
        f_prime.append(arctan_prime(z_3))
    delta = np.multiply(error, f_prime)

    for j in range(0,len(array_1_hidden_node)):
        weight = np.array(array_1_hidden_node[j].weight)

        weight = weight.transpose()
        final = weight*delta
        array_1_hidden_node[j].weight = array_1_hidden_node[j].weight - final

    for j in range(0, len(array_2_hidden_node)):
        array_2_hidden_node[j].bias = array_2_hidden_node[j].bias - delta[j]



    ####################### 3 ###########

    output = []
    for j in range(0,len(array_1_hidden_node)):
        output.append(array_1_hidden_node[j].value)
    error = output

    f_prime = []
    for j in range(0, len(array_1_hidden_node)):       #current layer
        z_3 = 0
        for k in range(0, len(array_pixel_node)): #Back layer
            z_3 = z_3 + array_pixel_node[k].weight[j] * array_pixel_node[k].value
        z_3 = z_3 + array_1_hidden_node[j].bias
        f_prime.append(arctan_prime(z_3))
    delta = np.multiply(error, f_prime)

    for j in range(0,len(array_pixel_node)):
        weight = np.array(array_pixel_node[j].weight)

        weight = weight.transpose()
        final = weight*delta
        array_pixel_node[j].weight = array_pixel_node[j].weight - final

    for j in range(0, len(array_1_hidden_node)):
        array_1_hidden_node[j].bias = array_1_hidden_node[j].bias - delta[j]


    '''
    for j in range(0, len(array_output_node)):   #current Layer
        z_3 = 0
        a_k = 0
        for k in range(0, len(array_2_hidden_node)): #Back layer
            z_3 = z_3 + array_2_hidden_node[k].weight[j] * array_2_hidden_node[k].value                 #z_3 = sum(W_j * a_k)
        z_3 = z_3 + array_output_node[j].bias                                                           #z_3 = sum(w_j * a_k) + b_j
        for k in range(0, len(array_2_hidden_node)): #Back layer
            a_k = array_2_hidden_node[k].value
            cost_d_weight_1 = a_k*arctan_prime(z_3)*2*(array_output_node[j].value - error_array[j])     #dc/dw = a_k * arctan_prime(z_3) * 2 *(a_j - y_j)
            array_2_hidden_node[k].weight[j] = array_2_hidden_node[k].weight[j] - cost_d_weight_1       #w_ij = w_ij - dc/dw

        cost_d_bias_1 = arctan_prime(z_3)*2*(array_output_node[j].value - error_array[j])               #dc/db =  arctan_prime(z_3) * 2 *(a_j - y_j)

        array_output_node[j].bias = array_output_node[j].bias - cost_d_bias_1                           #b_j = b_j - dc/db
    '''




    '''
    for k in range(0, len(array_1_hidden_node)):       #back layer
        for j in range(0, len(array_2_hidden_node)):   #currentlayer
            z_2 = array_1_hidden_node[k].weight[j] * array_1_hidden_node[k].value + array_2_hidden_node[i].bias
            cost_d_weight_2 = array_1_hidden_node[k].value*arctan_prime(z_2)*2*array_2_hidden_node[j].value - array_output_node[j].value
            cost_d_bias_2 = arctan_prime(z_2)*2*array_2_hidden_node[j].value - array_output_node[j].value

            array_1_hidden_node[k].weight[j] = array_1_hidden_node[k].weight[j] - cost_d_weight_2
            array_2_hidden_node[j].bias = array_2_hidden_node[j].bias - cost_d_bias_2



    for k in range(0, len(array_pixel_node)):            #back layer
        for j in range(0, len(array_1_hidden_node)):     #currentlayer
            for i in range(0, len(array_2_hidden_node)): #front layer
                a = array_1_hidden_node[j].value - array_2_hidden_node[i].value
                sum = sum + a
            sum = sum/len(array_2_hidden_node)
            z_1 = array_pixel_node[k].weight[j] * array_pixel_node[k].value + array_2_hidden_node[i].bias
            cost_d_weight_1 = array_pixel_node[k].value*arctan_prime(z_1)*2*sum
            cost_d_bias_1 = arctan_prime(z_1)*2*sum

            array_pixel_node[k].weight[j] = array_pixel_node[k].weight[j] - cost_d_weight_1
            array_1_hidden_node[j].bias = array_1_hidden_node[j].bias - cost_d_bias_1
    '''
    return None
