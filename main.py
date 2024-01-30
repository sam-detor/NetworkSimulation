import matplotlib.pyplot as plt
from network import Network
from datetime import datetime
import os
import random

def main():
    random.seed(10)

    #  network parameters 
    numResources = 4 
    numPairs = 12
    numIterations = 100
    externalTrafficProbability = 0
    numPairsPerResource = 3 # 0 means random resource assignment, otherwise it is the number of pairs per resource

    # initialize network simulation 
    myNetwork = Network(numResources, numPairs, externalTrafficProbability, numPairsPerResource)
    myNetwork.printNetworkState()
    data = myNetwork.simulate(numIterations)
    print(data)

    # save plots 
    timestamp_str = datetime.now().strftime("%m-%d_%H-%M-%S")
    folder_path =f"folder_{timestamp_str}/"

    # create the new folder
    try:
        os.makedirs(folder_path)
        print(f"New folder '{folder_path}' created successfully.")
    except OSError as e:
        print(f"Error creating folder: {e}")




    ################################################
    # Plot 1: All Pairs 
    ################################################
        
    #plt.subplot(3, 1, 1)  # 3 rows, 1 column, plot 1
    for i in range(len(data[1])):
        myLabel1 = "Pair" + str(i)+ "Throuput"
        myLabel2 = "Pair" + str(i) + "Limit"
        plt.plot(data[0], data[1][i], label=myLabel1) # this is fuckign breaking 
        plt.plot(data[0], data[2][i], label=myLabel2)
    plt.title('Pairs')
    plt.legend()

    plt.savefig(folder_path + "Pairs.png")
    plt.clf()
    



    #############################################
    # Plot 2: Destinations (1 plot for each dest)
    #############################################
    
    #plt.subplot(3, 1, 3)  # 3 rows, 1 column, plot 3
    for i in range(len(data[3])):
        plt.clf()
        myLabel1 = "Dest" + str(i) + "Throuput"
        myLabel2 = "Dest" + str(i) + "Limit"
        myLabel3 = "Dest" + str(i) + "Opt"
        myLabel4 = "Dest" + str(i) + "Target"
        plt.plot(data[0], data[3][i], label=myLabel1)
        plt.plot(data[0], data[4][i], label=myLabel2)
        plt.plot(data[0], data[5][i], label=myLabel3)
        plt.plot(data[0], data[6][i], label=myLabel4)
        plt.title("Destination" + str(i))
        plt.legend()
        plt.savefig(folder_path + "Destination" + str(i) + ".png")

    plt.clf()

    # how can i print out the goal setpoint  


    ################################################
    # Plot 3: All Resources 
    ################################################
        
    #plt.subplot(3, 1, 2)  # 3 rows, 1 column, plot 2
    print("Data 3:", data[7])
    print("Data 3 len:", data[7])
    for i in range(len(data[7])):
        myLabel1 = "Resource" + str(i) + "Throuput"
        myLabel2 = "Resource" + str(i) + "Limit"
        plt.plot(data[0], data[7][i], label=myLabel1)
        plt.plot(data[0], data[8][i], label=myLabel2)
    plt.title('Resources')
    plt.legend()
    plt.savefig(folder_path + "Resources.png")


    ################################################
    # Plot 4: Pairs on Each Resource 
    ################################################
    destinations = myNetwork.getDestinations()
    pairData = data[1]

    fig_size = (10, 6)

    for dest, pairs in destinations.items():
        plt.figure(figsize=fig_size)
        
        for pair in pairs:
            plt.plot(data[0], pairData[pair.getId()], label="Pair" + str(pair.getId()))

        plt.title("Pairs on Destination " + str(dest._hashVal))
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.savefig(folder_path + "Destination" + str(dest._hashVal) + "_Pairs_Plot.png", bbox_inches='tight')


if __name__ == "__main__":
    main()