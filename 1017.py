import pandas as pd

centers_df = pd.read_excel('data_final_center.xlsx')

# Extract coordinates from DataFrame and convert to numpy array
centers = centers_df.iloc[:, 1:4].to_numpy()

# Print the centers to check if they are loaded correctly
print(centers)
print(centers[:, 0])
