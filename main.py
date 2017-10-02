""" 2-input XOR example """
from __future__ import print_function

from neat import nn, population, statistics, visualize

divisor = 100000
fp=open("train10.txt")
inputs=[]
outputs=[]
for i in fp:
    temp = []
    i.rstrip()
    j=i.split()
    outputs.append(int(j[-1]))
    j.pop()
    for k in j:
        temp.append(int(int(k)/int(divisor)))
    inputs.append(temp)
fp.close()
print (inputs)
print (outputs)


# Network inputs and expected outputs.
xor_inputs = inputs
xor_outputs = outputs


def eval_fitness(genomes):
    for g in genomes:
        net = nn.create_feed_forward_phenotype(g)

        sum_square_error = 0.0
        for inputs, expected in zip(xor_inputs, xor_outputs):
            # Serial activation propagates the inputs through the entire network.
            output = net.serial_activate(inputs)
            sum_square_error += (output[0] - expected) ** 2

        # When the output matches expected for all inputs, fitness will reach
        # its maximum value of 1.0.
        g.fitness = 1 - sum_square_error


pop = population.Population('xor2_config.txt')

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

fp_test=open("test10.txt")
test_inputs=[]
for i in fp_test:
    temp = []
    i.rstrip()
    j=i.split()
    for k in j:
        temp.append(int(int(k)/int(divisor)))
    test_inputs.append(temp)

print (test_inputs)
fp_test.close()
Test_data=test_inputs
net = nn.create_feed_forward_phenotype(winner)
fp_out=open("neat_output.txt",'w')
for input in Test_data:
    output = net.serial_activate(input)
    fp_out.writelines(str(output)+"\n")
    print ("for "+ str(input) +" the predicted output is : " + str(output))
fp_out.close()
print ("Testing ends here.")
#print (output)
#print("hiiiiiiiiiiiiiiiiii")
#print(output)



# Visualize the winner network and plot/log statistics.
visualize.plot_stats(pop.statistics)
visualize.plot_species(pop.statistics)
visualize.draw_net(winner, view=True, filename="xor2-all.gv")
visualize.draw_net(winner, view=True, filename="xor2-enabled.gv", show_disabled=False)
visualize.draw_net(winner, view=True, filename="xor2-enabled-pruned.gv", show_disabled=False, prune_unused=True)
statistics.save_stats(pop.statistics)
statistics.save_species_count(pop.statistics)
statistics.save_species_fitness(pop.statistics)