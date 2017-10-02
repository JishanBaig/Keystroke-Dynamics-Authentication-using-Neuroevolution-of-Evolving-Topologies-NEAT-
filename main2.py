
""" 2-input XOR example """
from __future__ import print_function

from neat import nn, population, statistics, visualize

# Network inputs and expected outputs.
from pip._vendor.distlib.compat import raw_input

xor_inputs = [[0, 0], [0, 1], [1, 0],[1,1]]
xor_outputs = [0, 1, 1,0]
x = [1]

def eval_fitness(genomes):
    for g in genomes:
        net = nn.create_feed_forward_phenotype(g)

        sum_square_error = 0.0
        for inputs, expected in zip(xor_inputs, xor_outputs):
            # Serial activation propagates the inputs through the entire network.
            output = net.serial_activate(inputs)
            #print type(output)
            sum_square_error += (float(output[0]) - float(expected)) ** 2

        # When the output matches expected for all inputs, fitness will reach
        # its maximum value of 1.0.
        g.fitness = 1 - sum_square_error


pop = population.Population('xor2_config.txt')
while len(x)!=0:
    pop.run(eval_fitness, 300)

    print('Number of evaluations: {0}'.format(pop.total_evaluations))

    # Display the most fit genome.
    winner = pop.statistics.best_genome()
    print('\nBest genome:\n{!s}'.format(winner))
    s=('\nBest genome:\n{!s}'.format(winner))
    out=open("bestnn.txt","w")
    out.write(str(s))
    out.close()

    # Verify network output against training data.
    print('\nOutput:')
    winner_net = nn.create_feed_forward_phenotype(winner)
    print("For the Training data : ")
    for inputs, expected in zip(xor_inputs, xor_outputs):
        #print (inputs)
        output = winner_net.serial_activate(inputs)
        print("expected {0:1.5f} got {1:1.5f}".format(expected, output[0]))

        # Testing of the test data
        print("Testing of test data starts from here : ")
        Test_data=[[0,1]]
        net = nn.create_feed_forward_phenotype(winner)
    for input in Test_data:
        output = net.serial_activate(input)
        print ("for "+ str(input) +" the predicted output is : " + str(output))
    print ("Testing ends here.")
    #print (output)
    #print("hiiiiiiiiiiiiiiiiii")
    #print(output)
    #Training according to the test output
    y = str(raw_input('Enter the actual output : '))
    x=y.split()
    for k in x:
        k=int(k)
    for i in Test_data:
        xor_inputs.append(i)
    for j in x:
        xor_outputs.append(j)


# Visualize the winner network and plot/log statistics.
visualize.plot_stats(pop.statistics)
visualize.plot_species(pop.statistics)
visualize.draw_net(winner, view=True, filename="xor2-all.gv")
visualize.draw_net(winner, view=True, filename="xor2-enabled.gv", show_disabled=False)
visualize.draw_net(winner, view=True, filename="xor2-enabled-pruned.gv", show_disabled=False, prune_unused=True)
statistics.save_stats(pop.statistics)
statistics.save_species_count(pop.statistics)
statistics.save_species_fitness(pop.statistics)