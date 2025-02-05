import pandas as pd
import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt.plots import plot_convergence
import matplotlib.pyplot as plt
from gplearn.genetic import SymbolicRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from skopt import Optimizer
from joblib import Parallel, delayed
import time
import os
from datetime import datetime
import csv

no_iterations = 40
no_samples_per_iter = 25
n_initial_points = 200
file_name = "./Nd_rp700"

parsimony_coefficient = 0.005


print(f"file name: {file_name}")

data = pd.read_excel("./Nd_rp700.xlsx",sheet_name=0)
data = data.fillna(0)
x = data.values[:,1:-1]
y = data.values[:,-1]

feature_names = data.columns.values[1:-1]

# magic code to let skopt working with modern version of numpy
np.int = int

# Adjust the space definition to match your problem's specific needs
space = [
    # Integer(5000, 10000, name='population_size'),
    Integer(50, 100, name='generations'),
    Integer(10, 1000, name='tournament_size'),
    # sum should be less than 1
    Real(0.1, 0.7, name='p_crossover'),
    Real(0.01, 0.1, name='p_subtree_mutation'),
    Real(0.01, 0.1, name='p_hoist_mutation'),
    Real(0.01, 0.1, name='p_point_mutation'),
    # end of sum
    Real(0.01, 0.1, name='p_point_replace'),
]

# Define fixed parameters
fixed_params = {
    'population_size': 10000,
    'feature_names': feature_names,
    'random_state': 0,
    'metric': 'rmse',
    'parsimony_coefficient': parsimony_coefficient,
    'max_samples': 1,
    'stopping_criteria': 0.3,
    'function_set': ('add', 'sub', 'mul', 'div', 'neg'),
    'const_range': (-9.0, 9.0),
}

# Objective function adjustment to return a tuple
def objective_function(params):
    all_params = {**params, **fixed_params}  
    est_gp = SymbolicRegressor(**all_params)
    est_gp.fit(x, y.reshape(-1))
    y_pred = est_gp.predict(x)
    rmse = mean_squared_error(y, y_pred, squared=False)
    return rmse, est_gp._program

opt = Optimizer(dimensions=space, base_estimator="GP", n_initial_points=n_initial_points, random_state=0, n_jobs=-1)

# Placeholder for storing scores to display progress
scores_history = []
best_scores = []
models_all = []

##################
current_time_formatted = datetime.now().strftime("%m-%d %H:%M:%S")
print(f"Timestamp: {current_time_formatted}")
# Initialize the optimizer with the defined space
print(f"Starting initialization with no_iterations: {no_iterations}, no_samples_per_iter: {no_samples_per_iter}, n_initial_points: {n_initial_points}")
start_time = time.time()  # Start time of the initialize

suggested_points = opt.ask(n_points=n_initial_points)  # Points for parallel evaluation

# Evaluate in parallel
results = Parallel(n_jobs=no_samples_per_iter)(
    delayed(objective_function)(dict(zip([dim.name for dim in space], point))) for point in suggested_points
)
scores, models = zip(*results)  # This unpacks the list of tuples

opt.tell(suggested_points, scores)

# Update scores history
scores_history.extend(scores)
models_all.extend(models)

# Update best score so far
current_best_score = min(scores_history)
best_scores.append(current_best_score)

end_time = time.time()  # End time of the initialize
elapsed_time = end_time - start_time  # Calculate elapsed time for the initialize
print(f"Finished initialization, time taken: {elapsed_time:.2f} seconds")
##################

# Perform optimization with dynamic updates
for n in range(no_iterations):  # Total number of iterations
    print(f"Starting iteration {n+1}/{no_iterations}")
    current_time_formatted = datetime.now().strftime("%m-%d %H:%M:%S")
    print(f"Timestamp: {current_time_formatted}")
    start_time = time.time()  # Start time of the iteration

    suggested_points = opt.ask(n_points=no_samples_per_iter)  # Points for parallel evaluation

    # Evaluate in parallel
    results = Parallel(n_jobs=no_samples_per_iter)(
        delayed(objective_function)(dict(zip([dim.name for dim in space], point))) for point in suggested_points
    )
    scores, models = zip(*results) 

    opt.tell(suggested_points, scores)

    # Update scores history
    scores_history.extend(scores)
    models_all.extend(models)

    # Update best score so far
    current_best_score = min(scores_history)
    best_scores.append(current_best_score)

    # Timing
    end_time = time.time()  # End time of the iteration
    elapsed_time = end_time - start_time  # Calculate elapsed time for the iteration

    # Print the best so far and the time taken
    print(f"Iteration {n+1}/{no_iterations}: Best RMSE so far: {current_best_score:.4f}, Time taken: {elapsed_time:.2f} seconds")


# Save the final plot
plt.figure(figsize=(10, 5))
plt.plot(best_scores, 'o-', color='red', label='Best RMSE So Far')
plt.title('Optimization Progress')
plt.xlabel('Iteration')
plt.ylabel('RMSE')
plt.legend()
plt.savefig(f'{file_name}.png')
plt.close()
print(f"Saved plot to {file_name}.png")
# Initialize a list to store the output results
output_results = []
# Prepare data for CSV
csv_data = [("Rank", "RMSE", "Length", "Model Equation")]

# Identify the top 20 results based on RMSE scores
top_indices = np.argsort(opt.yi)

# Train and evaluate SymbolicRegressor for each top parameter set
for rank, index in enumerate(top_indices, start=1):
    model = models_all[index]
    rmse = scores_history[index]
    output_results.append(f"Rank {rank}, RMSE: {rmse:.4f}, Length: {model.length_}, Model Equation: {model}")
    # Prepare data row for CSV
    csv_data.append((rank, f"{rmse:.4f}", model.length_, str(model)))
    
    if rank < 20:
        print((f"Rank {rank}, RMSE: {rmse:.4f}, Length: {model.length_}, Model Equation: {model}"))

# Save the output results to a text file
with open(f'{file_name}.txt', 'w') as file:
    for result in output_results:
        file.write(result + "\n")

print(f"Saved results to {file_name}.txt")

# Save the results to a CSV file
with open(f'{file_name}.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(csv_data)

print(f"Saved results to {file_name}.csv")
