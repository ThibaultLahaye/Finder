import pandas as pd
import matplotlib.pyplot as plt

heuristics = dict()
heuristics[50] = 27723
heuristics[100] = 90851
heuristics[200] = 39745
heuristics[500] = 157034
heuristics[750] = 197541
heuristics[1000] = 195848

def percentage_from_heuristic(value, heuristic):
    percentage = (heuristic / value - 1) * 100
    if percentage < 0:
        return -(percentage + 100), 'diminishment', 'red'
    else:
        return percentage, 'improvement', 'green'

def load_and_process_csv(file_path):
    # Load CSV file into a DataFrame
    df = pd.read_csv(file_path, comment='#')

    iteration = df.iloc[:, 0]
    elapsed_time = df.iloc[:, 1]
    mean_values = df.iloc[:, 2]
    best_values = df.iloc[:, 3]

    num_cities = len(df.iloc[4, 4:-1].values)

    return iteration, elapsed_time, mean_values, best_values, num_cities

def plot_elapsed_time(iteration, elapsed_time, mean_values, best_values, heuristic, percentage_from_heuristic, best_mean_value, best_value, log_scale=False):
    plt.figure(figsize=(10, 6))
    plt.plot(elapsed_time, mean_values, label='Mean Value')
    plt.plot(elapsed_time, best_values, label='Best Value')
    plt.axhline(y=heuristic, color='r', linestyle='--', label='Heuristic')
    plt.xlabel('Elapsed Time')
    plt.ylabel('Values')
    plt.yscale('log') if log_scale else None  # Set y-axis to logarithmic scale if log_scale is True
    plt.title('Elapsed Time vs Mean and Best Values')

    # Add improvement_as_percentage as text annotation
    percentage, improvement_text, color = percentage_from_heuristic(best_value, heuristic)
    plt.annotate(f'{improvement_text}: {percentage:.2f}%', xy=(0.05, 0.85), xycoords='axes fraction', fontsize=12, fontweight='bold', color=color)

    # Add best_mean_value as text annotation
    plt.annotate(f'Best Mean Value: {best_mean_value:.2f}', xy=(0.05, 0.83), xycoords='axes fraction', fontsize=12, fontweight='bold', color='blue')

    # Add best_value as text annotation
    plt.annotate(f'Best Value: {best_value:.2f}', xy=(0.05, 0.81), xycoords='axes fraction', fontsize=12, fontweight='bold', color='purple')

    plt.legend()
    plt.grid(True)
    plt.show()

def plot_iteration(iteration, mean_values, best_values, heuristic, percentage_from_heuristic, best_mean_value, best_value, log_scale=False):
    plt.figure(figsize=(10, 6))
    plt.plot(iteration, mean_values, label='Mean Value')
    plt.plot(iteration, best_values, label='Best Value')
    plt.axhline(y=heuristic, color='r', linestyle='--', label='Heuristic')
    plt.xlabel('Iteration')
    plt.ylabel('Values')
    plt.yscale('log') if log_scale else None  # Set y-axis to logarithmic scale if log_scale is True
    plt.title('Iteration vs Mean and Best Values')

    # Add improvement_as_percentage as text annotation
    percentage, improvement_text, color = percentage_from_heuristic(best_value, heuristic)
    plt.annotate(f'{improvement_text}: {percentage:.2f}%', xy=(0.05, 0.85), xycoords='axes fraction', fontsize=12, fontweight='bold', color=color)

    # Add best_mean_value as text annotation
    plt.annotate(f'Best Mean Value: {best_mean_value:.2f}', xy=(0.05, 0.83), xycoords='axes fraction', fontsize=12, fontweight='bold', color='blue')

    # Add best_value as text annotation
    plt.annotate(f'Best Value: {best_value:.2f}', xy=(0.05, 0.81), xycoords='axes fraction', fontsize=12, fontweight='bold', color='purple')

    plt.legend()
    plt.grid(True)
    plt.show()

def create_plots(iteration, elapsed_time, mean_values, best_values, num_cities, log_scale=False, show_heuristic=False):
    heuristic = heuristics[num_cities]
    best_value = best_values.iloc[-1]
    best_mean_value = mean_values.iloc[-1]

    plot_elapsed_time(iteration, elapsed_time, mean_values, best_values, heuristic, percentage_from_heuristic, best_mean_value, best_value, log_scale)
    plot_iteration(iteration, mean_values, best_values, heuristic, percentage_from_heuristic, best_mean_value, best_value, log_scale)

if __name__ == "__main__":
    file_path = './r0713047.csv'

    iteration, elapsed_time, mean_value, best_value, num_cities = load_and_process_csv(file_path)
    create_plots(iteration, elapsed_time, mean_value, best_value, num_cities, log_scale=False, show_heuristic=False)
