import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import math
import sys
import random
import math
import gzip
from random import shuffle
import os

import Sky_Net_classes_1
from Sky_Net_classes_1 import pixel_node
from Sky_Net_classes_1 import hidden_col_1_node
from Sky_Net_classes_1 import hidden_col_2_node
from Sky_Net_classes_1 import output_node

import Sky_Net_functions_1
from Sky_Net_functions_1 import initials_classes
from Sky_Net_functions_1 import guess_number
from Sky_Net_functions_1 import load
from Sky_Net_functions_1 import save
from Sky_Net_functions_1 import get_training_data
from Sky_Net_functions_1 import get_training_data_prime
from Sky_Net_functions_1 import back_prop

'''
cwd = os.getcwd()  # Get the current working directory (cwd)
files = os.listdir(cwd)  # Get all the files in that directory
print("Files in '%s': %s" % (cwd, files))
'''


'''
*********Instructions************
** IF YOU WANT TO TRAIN A NEW NETWORK:
** 1.) Set new_net to:  True
** 2.) Run script once
** 3.) Set new_net to:  False
** 4.) Run script
**
** OTHERWISE LEAVE THIS ALONE
'''
new_net = False
human_net = False
test_net = False








'''
** Opens an image file and converts it
** to gray scale (255 = white, 0 = black)/
** The program then resizes the image to
** 27 pixels by 27 pixels.
** Then the program converts the image into
** an np.array() and converts the grayscale to
** be between 0 and 1 and inverts white and black.
** I then flatten the array of dots
'''
if human_net == True:
    image = Image.open('test_image.png').convert('L')
    new_image = image.resize((28,28))
    dots = np.array(new_image)                      #these are 28 lists ranging from [0 -- 27]
    dots = 1 - dots/255
    pixels = dots.flatten()

    total_num_image = 1
    subset = 1
    throughset = 1

else:
    if test_net == False:
        total_num_image = 60000
        subset = 1
        throughset = int(total_num_image/subset)
        image_label = get_training_data(total_num_image,28)
        pixels = image_label[0][0]
    else:
        total_num_image = 3
        subset = 10
        throughset = int(total_num_image/subset)
        image_label = get_training_data(total_num_image,28)
        image_label_prime = get_training_data_prime(total_num_image,28)
        pixels = image_label[0][0]


'''
*********DONT USE AFTER TRAINING************
** If new_net is True than I run the function initials_classes_init.
** This function is given a flattened array of
** values and initializes all the neccessary classes
** with random values with the exception of the pixels.
** I then save those arrays using the save() function
** and end the program.
'''
if new_net == True:
    node_arrays = initials_classes(pixels)
    save(node_arrays)
    sys.exit()


'''
** Otherwise I call the function initials_classes
** function and then load the previously saved arrays
** using the load function.
'''
node_arrays = initials_classes(pixels)
load(node_arrays)


if human_net == True:
    '''
    ** I give the function, guess_number(), the node
    ** arrays, and it computes the computers guess.
    '''
    list_values = guess_number(node_arrays)
    guess = np.argmax(list_values)
    print(list_values)
    print("guess", guess)
    fig = plt.figure()
    ax2 = fig.add_subplot(1,1,1)
    plt.title("Gray_image")
    ax2.imshow(dots, interpolation='nearest', cmap=cm.Greys_r)
    plt.show()
    sys.exit()



d_avg_cost = np.array([0,0,0,0,0,0,0,0,0,0])
pre_d_avg_cost = np.array([0,0,0,0,0,0,0,0,0,0])
lap = 1
'''************* PROGRAM STARTS HERE *************'''
if human_net != True:
    if test_net != True:
        shuffle(image_label)

#vairable
count = 0


y_axis_plot = []
y_axis_derivitive = []
accuracy = []
uncurtanty = []

for outer_set in range(0,throughset):
    if outer_set != 0:
        print(outer_set + 1, "of", throughset, "sets.", "Lap", lap)

    avg_cost = np.array([0,0,0,0,0,0,0,0,0,0])
    plot_avg_cost = 0
    total = 0
    correct = 0
    for inner_set in range(0,subset):

        ########################## bug ############################################################
        # node_arrays does not update for new images for batch size greater than 1 #
        for i in range(0,len(node_arrays[0])):                              #updates picture
            node_arrays[0][i].value = float(image_label[count][0][i])

        '''

        for i in range(0,len(node_arrays[0])):
            running_total = running_total + 1
            if image_label[count][0][i] == image_label[count+1][0][i]:
                tally = tally + 1
        print(tally, running_total)
        '''

        correct_answer = np.array([0,0,0,0,0,0,0,0,0,0])                    #updates answer
        correct_answer[image_label[count][1]] = 1

        #node_array is changing
        list_values = guess_number(node_arrays)                             #guesses the answer

        '''
        running_total = 0
        tally = 0
        if inner_set == 0:
            old = list_values
        if inner_set == 1:
            new = list_values
            for i in range(0,len(old)):
                running_total = running_total + 1
                if old[i] == new[i]:
                    tally = tally + 1
            print(running_total, tally)
        '''

        count = count + 1

        #calculateing the accuracy
        answer = np.argmax(correct_answer)
        guess = np.argmax(list_values)

        #print(guess, list_values, "\n")

        if guess == answer:
            #print("I am right!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            correct = correct + 1
        #print(answer, guess)
        total = total + 1
        accuracy.append(correct / total)



        #calcualting uncertainty of NN
        uncurtanty.append(sum(abs(list_values)))
        abc = sum(abs(list_values))




        avg_cost = avg_cost + (list_values - correct_answer)**2               #finds the total wrongness
        plot_avg_cost =  plot_avg_cost + (list_values - correct_answer)**2#########


    pre_d_avg_cost = (1/subset) *avg_cost                                   #finds multiple of average wrongness
    if test_net == True:
        guess = np.argmax(list_values)
        print("list_values",list_values)
        print("guess", guess)
        print( "answer", image_label[count][1])
        print("answer_prime",image_label_prime[count][1])
        fig = plt.figure()
        ax2 = fig.add_subplot(1,1,1)
        plt.title("What PC sees")
        ax2.imshow(image_label_prime[count][0], interpolation='nearest', cmap=cm.Greys_r)
        plt.show()
        count = count + 1
        fun = False
        #sys.exit()
        continue


    plot_avg_cost = plot_avg_cost/subset###############
    print("The derivitive of the sumed average cost of set", outer_set + 1, "is", sum(pre_d_avg_cost) ,sum(pre_d_avg_cost - d_avg_cost))
    print("uncertainty is: ", abc)
    d_avg_cost = pre_d_avg_cost

    y_axis_plot.append(sum(plot_avg_cost))      ###############
    y_axis_derivitive.append(sum(d_avg_cost))   ###############


    if  outer_set >  200:
        #if np.abs(sum(pre_d_avg_cost)) < abs(1):
            break

    back_prop(node_arrays,correct_answer)








save(node_arrays)

fig = plt.figure()
ax1 = fig.add_subplot(811)
ax1.set_title("sum(derivitive avg_cost) vs steps")
plt.xlabel("steps (1 step = 1 example)")
plt.ylabel("sum(derivitive avg_cost)")
ax1.plot(y_axis_derivitive)

ax2 = fig.add_subplot(813)
ax2.set_title("loss vs steps")
plt.xlabel("steps (1 step = 1 example)")
plt.ylabel("sum(avg_cost)")
ax2.plot(y_axis_plot)

#accuacy graph
ax1 = fig.add_subplot(815)
ax1.set_title("accuracy vs steps")
plt.xlabel("steps (1 step = 1 example)")
plt.ylabel("accuracy")
ax1.plot(accuracy)

#unaccuacy graph
ax1 = fig.add_subplot(817)
ax1.set_title("uncurtainty")
plt.xlabel("steps (1 step = 1 example)")
plt.ylabel("uncertainty")
ax1.plot(uncurtanty)
plt.show()
