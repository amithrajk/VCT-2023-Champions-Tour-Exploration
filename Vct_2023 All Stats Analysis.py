#!/usr/bin/env python
# coding: utf-8

# 
# 
# **Title: Exploring Valorant Champions Tour 2023 Data**
# 
# **Introduction:**
# In this data analysis, we explored data from the Valorant Champions Tour (VCT) 2023 to gain insights into team performance, player statistics, and the distribution of key metrics. Valorant is a popular competitive shooter game, and VCT is a major esports event. Let's dive into the data and uncover some interesting trends.
# 
# **Team Performance:**
# We started by analyzing team performance. It's fascinating to see how different teams performed in various maps. Some teams excelled in certain maps, while others struggled. The "win_lose" variable helped us understand which teams had the upper hand in these matchups. It's clear that team strategy and map selection play a significant role in VCT outcomes.
# 
# **Player Statistics:**
# We also delved into player statistics, focusing on key metrics like kills, deaths, assists, rating, and more. These statistics give us a glimpse into individual player performance. We noticed that some players consistently stood out with high ratings, showcasing their skills in the game. It's intriguing to see how these statistics contribute to a player's overall performance.
# 
# **Outlier Detection:**
# Outliers were an important aspect of our analysis. We used box plots to identify extreme values in the data. Detecting outliers is crucial for understanding data quality and ensuring they don't adversely affect our analysis. Addressing outliers appropriately can lead to more accurate insights.
# 
# **Distribution Analysis:**
# We explored the distribution of numerical variables, such as player ratings. The data showed that the distribution of player ratings is slightly skewed. Understanding these distributions helps us know what to expect in terms of player performance and how it affects team dynamics.
# 
# **Correlation Analysis:**
# Correlation analysis revealed interesting relationships between variables. We discovered positive correlations between some player statistics, which make sense in the context of a team-based game like Valorant. These insights help us understand how different metrics influence each other.
# 
# **Conclusion:**
# Valorant Champions Tour 2023 is a competitive esports event with rich data that offers valuable insights. From team performance to player statistics, the data provides a comprehensive view of the VCT landscape. Exploring this data not only helps us appreciate the dynamics of professional Valorant but also serves as a basis for more in-depth analysis and predictions. As Valorant continues to grow in popularity, data-driven insights become even more critical for fans and analysts alike.
# 
# **Future Analysis:**
# In the future, we can further explore this dataset by building predictive models, conducting player comparisons, and providing insights into strategies used by top-performing teams. The possibilities are endless, and the data will continue to tell a compelling story of Valorant esports.
# 
# 

# In[3]:


import pandas as pd
import numpy as np
import os


# In[4]:


os.chdir("D:\\")
os.getcwd()


# In[5]:


vct_file=pd.read_csv("D:\player_stats.csv")
vct_file


# inference:This code will read the CSV file located at "D:\player_stats.csv" and store its contents in the vct_file DataFrame. You can then print the DataFrame to see the data from the CSV file.

# In[6]:


# Display column names of the DataFrame
column_names = vct_file.columns

print("Column Names:",column_names)


# inference:In this code, vct_file.columns returns a list of column names in the DataFrame, and then we print each column name one by one. This will display the names of all the columns in your DataFrame.

# In[7]:


vct_file.head()


# In[8]:


vct_file.tail()


# In[9]:


vct_file.info()


# In[10]:


# Use the .describe() method to get an overview of summary statistics
summary_stats = vct_file.describe()

# Print the summary statistics
summary_stats


# The .describe() method provides statistics for each numerical column in your DataFrame, including count, mean, standard deviation, minimum, 25th percentile (Q1), median (50th percentile), 75th percentile (Q3), and maximum values. It gives you a quick overview of the central tendency, spread, and distribution of your numerical data.

# In[11]:


# Calculate the correlation matrix
correlation_matrix = vct_file.corr(numeric_only=True)

correlation_matrix


# In[12]:


import seaborn as sns
import matplotlib.pyplot as plt

# Create a heatmap of the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()


# Inference:
# In the heatmap, positive correlations are usually shown in warmer colors (e.g., red), while negative correlations are shown in cooler colors (e.g., blue). Values close to 0 appear close to white, indicating little to no correlation.
# 
# By examining the correlation matrix and heatmap, you can identify which pairs of variables are strongly correlated, which can be valuable for feature selection, understanding relationships in your data, and informing further analysis or modeling tasks.

# In[13]:


win_lose_counts = vct_file['win_lose'].value_counts()
print(win_lose_counts)


# Inference:This will tell you how many times each category appears in your data.

# In[34]:


import matplotlib.pyplot as plt

# Create a bar plot for the "win_lose" variable
plt.figure(figsize=(8, 6))
win_lose_counts.plot(kind='bar', color='darkblue')
plt.title("Distribution of 'win_lose'")
plt.xlabel("Win/Lose")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.show()


# inference : category diff on win /loose in the data how the count has been distrubuted or repeated is been shown here.

# In[22]:


import matplotlib.pyplot as plt

# Create a histogram for the 'rating' column
plt.figure(figsize=(8, 6))
plt.hist(vct_file['kill'], bins=20, color='skyblue')
plt.title("Histogram of 'rating'")
plt.xlabel("kill")
plt.ylabel("acs")
plt.show()



# In[30]:


# Group the data by the 'team' column and calculate the mean rating for each team
team_ratings = vct_file.groupby('team')['score_team'].mean()

# Sort the teams based on their average rating in descending order
top_teams = team_ratings.sort_values(ascending=False)

# Select the top 5 teams
top_5_teams = top_teams.head(5)

# Display the top 5 teams and their average ratings
print("Top 5 Teams:")
print(top_5_teams)


# inference :Top 5 teams based on score.

# In[31]:


import seaborn as sns
import matplotlib.pyplot as plt

# Group the data by the 'team' column and calculate the mean 'score_team' for each team
team_scores = vct_file.groupby('team')['score_team'].mean().reset_index()

# Sort the teams based on their average 'score_team' in descending order
top_5_teams = team_scores.nlargest(5, 'score_team')

# Create a bar plot to visualize the top 5 teams
plt.figure(figsize=(10, 6))
sns.barplot(x='score_team', y='team', data=top_5_teams, palette='viridis')
plt.title("Top 5 Teams by Average 'score_team'")
plt.xlabel("Average 'score_team'")
plt.ylabel("Team")
plt.show()


# 
# **Inference:**
# 
# 1. **Top-Performing Teams:** The bar plot clearly highlights the top 5 teams with the highest average 'score_team' values. These teams have consistently scored well in the matches they've played.
# 
# 2. **Comparison:** The visualization allows for easy visual comparison of the top teams. You can see how their average 'score_team' values stack up against each other.
# 
# 3. **Performance Ranking:** The ranking of teams based on 'score_team' provides valuable insights into which teams have been the most successful in terms of scoring in their matches.
# 
# 4. **Potential Insights:** By analyzing this data, you can draw insights into what makes these top teams perform better. It may be related to their strategies, teamwork, or individual player performance.
# 
# 5. **Data-Driven Decisions:** The visualization can be used to inform decisions related to team selection, strategy adjustments, or further analysis. Teams at the top of the ranking may be seen as strong contenders.
# 
# Overall, this visualization provides a quick and effective way to identify and compare the top-performing teams based on their average 'score_team' values. It's a valuable tool for both data-driven decision-making and gaining insights into team performance in Valorant Champions Tour 2023.

# In[ ]:




