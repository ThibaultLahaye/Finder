import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__": 
    dfs = []
    for i in range(1, 60):
        dfs.append(pd.read_csv(f'results/50/run_{i}.csv', comment='#'))

    # For each dataframe in dfs, get the best and mean values 
    # and append them to the best_values and mean_values lists
    # For each dataframe in dfs, only the best and mean vlues of the last iteration are used
    best_values = []
    mean_values = []
    for df in dfs:
        best_values.append(df.iloc[-1, 3].tolist())
        mean_values.append(df.iloc[-1, 2].tolist())
    
    print(best_values)

    # create a histogram of the best values
    plt.figure(figsize=(10, 6))
    plt.hist(best_values, bins=7)
    plt.xlabel('Best Values')
    plt.ylabel('Frequency')
    plt.title('Histogram of Best Values')
    plt.show()

    # create a histogram of the mean values
    plt.figure(figsize=(10, 6))
    plt.hist(mean_values, bins=7)
    plt.xlabel('Mean Values')
    plt.ylabel('Frequency')
    plt.title('Histogram of Mean Values')
    plt.show()
