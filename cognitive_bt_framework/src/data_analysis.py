import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV data into a DataFrame
# df = pd.read_csv('/home/liam/dev/cognitive_bt_framework/cognitive_bt_framework/data/run_data_09-01-2024-16-52-12.csv')
# df = pd.read_csv('/home/liam/dev/cognitive_bt_framework/cognitive_bt_framework/data/run_data_08-31-2024-12-41-11.csv')
df = pd.read_csv('/home/liam/dev/cognitive_bt_framework/cognitive_bt_framework/data/run_data_ablated_09-14-2024-18-42-14.csv')

# Convert the Success column to boolean
df['Success'] = df['Success'].astype('bool')

# Drop rows where ExecTime is NaN
df = df.dropna(subset=['ExecTime'])

# Basic Statistics
print("Basic Statistics:")
print(df.describe())

# Success rate by goal
success_rate_by_goal = df.groupby('Goal')['Success'].mean()
print("\nSuccess Rate by Goal:")
print(success_rate_by_goal)

# Average execution time by goal
avg_exec_time_by_goal = df.groupby('Goal')['ExecTime'].mean()
print("\nAverage Execution Time by Goal:")
print(avg_exec_time_by_goal)

# Success rate by scene
success_rate_by_scene = df.groupby('Scene')['Success'].mean()
print("\nSuccess Rate by Scene:")
print(success_rate_by_scene)

# Average execution time by scene
avg_exec_time_by_scene = df.groupby('Scene')['ExecTime'].mean()
print("\nAverage Execution Time by Scene:")
print(avg_exec_time_by_scene)

# Plot success rate by goal
plt.figure(figsize=(10, 6))
sns.barplot(x=success_rate_by_goal.index, y=success_rate_by_goal.values)
plt.title('Success Rate by Goal')
plt.xlabel('Goal')
plt.ylabel('Success Rate')
plt.show()

# Plot average execution time by goal
plt.figure(figsize=(10, 6))
sns.barplot(x=avg_exec_time_by_goal.index, y=avg_exec_time_by_goal.values)
plt.title('Average Execution Time by Goal')
plt.xlabel('Goal')
plt.ylabel('Average Execution Time (seconds)')
plt.show()

# Plot success rate by scene
plt.figure(figsize=(10, 6))
sns.barplot(x=success_rate_by_scene.index, y=success_rate_by_scene.values)
plt.title('Success Rate by Scene')
plt.xlabel('Scene')
plt.ylabel('Success Rate')
plt.show()

# Plot average execution time by scene
plt.figure(figsize=(10, 6))
sns.barplot(x=avg_exec_time_by_scene.index, y=avg_exec_time_by_scene.values)
plt.title('Average Execution Time by Scene')
plt.xlabel('Scene')
plt.ylabel('Average Execution Time (seconds)')
plt.show()

# Distribution of execution times
plt.figure(figsize=(10, 6))
sns.histplot(df['ExecTime'], kde=True)
plt.title('Distribution of Execution Times')
plt.xlabel('Execution Time (seconds)')
plt.ylabel('Frequency')
plt.show()

# Boxplot of execution times by goal
plt.figure(figsize=(10, 6))
sns.boxplot(x='Goal', y='ExecTime', data=df)
plt.title('Execution Time by Goal')
plt.xlabel('Goal')
plt.ylabel('Execution Time (seconds)')
plt.show()

# Scatter plot of execution time vs. trial, colored by success
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Trial', y='ExecTime', hue='Success', data=df)
plt.title('Execution Time vs. Trial, Colored by Success')
plt.xlabel('Trial')
plt.ylabel('Execution Time (seconds)')
plt.show()
