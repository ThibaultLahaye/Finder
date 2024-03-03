import pandas as pd
import matplotlib.pyplot as plt
import time
import os

from r0713047 import r0713047

def safe_open_w(path):
    ''' Open "path" for writing, creating any parent directories as needed.
    '''
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, 'w')

# Class to report basic results of an evolutionary algorithm
class Reporter:

	def __init__(self, filename):
		self.allowedTime = 300
		self.numIterations = 0
		self.filename = filename + ".csv"
		self.delimiter = ','
		self.startTime = time.time()
		self.writingTime = 0

        # Create file if the path does not exist yet
        # this returns direcoty does not exist error
		# outFile = open(self.filename, "w") 
		outFile = safe_open_w(self.filename)
		outFile.write("# Student number: " + filename + "\n")
		outFile.write("# Iteration, Elapsed time, Mean value, Best value, Cycle\n")
		outFile.close()





	# Append the reported mean objective value, best objective value, and the best tour
	# to the reporting file. 
	#
	# Returns the time that is left in seconds as a floating-point number.
	def report(self, meanObjective, bestObjective, bestSolution):
		if (time.time() - self.startTime < self.allowedTime + self.writingTime):
			start = time.time()
			
			outFile = open(self.filename, "a")
			outFile.write(str(self.numIterations) + self.delimiter)
			outFile.write(str(start - self.startTime - self.writingTime) + self.delimiter)
			outFile.write(str(meanObjective) + self.delimiter)
			outFile.write(str(bestObjective) + self.delimiter)
			for i in range(bestSolution.size):
				outFile.write(str(bestSolution[i]) + self.delimiter)
			outFile.write('\n')
			outFile.close()

			self.numIterations += 1
			self.writingTime += time.time() - start
		return (self.allowedTime + self.writingTime) - (time.time() - self.startTime)

if __name__ == "__main__":
    problem = r0713047()

    problem_size = 50

    for run_id in range(5):
        # Run the algorithm
        # store the results of the algorithm in seperate csv files
        destination_path = f"results/{problem_size}/run_{run_id}"

        # Create a reporter object
        reporter = Reporter(destination_path)

        # Run the algorithm
        pop_mean_fitness, pop_best_fitness, _ = problem.optimize(f"tours/tour{problem_size}.csv")

        # Store the population mean and best fitness, by calling the report function of the reporter object
        reporter.report(pop_mean_fitness, pop_best_fitness, [])  
        
