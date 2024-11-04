# Kickstarter-ML-Models

## The Client
The hypothetical client is focused on understanding the factors influencing the success or failure of Kickstarter projects through predictive modeling and clustering analysis.

## The Challenge
The challenge is to accurately predict the success of Kickstarter projects using various features, while also identifying distinct patterns in project characteristics that can inform creators and backers. Traditional models like decision trees and logistic regression were found inadequate due to their limitations in capturing complex relationships within the dataset.

## The Approach
1. Classification Model:
  - Model Selection: A Gradient Boosting Algorithm (GBA) was chosen for its high accuracy in predicting project outcomes, outperforming decision trees, logistic regression, and random forests. GBA's adaptive learning and ability to focus on misclassified instances make it well-suited for this task.
  - Feature Selection: The dataset was refined to include only relevant features, retaining those like project goals and descriptions while removing irrelevant or repetitive ones. Hyper-parameter tuning was performed using GridSearchCV to enhance model accuracy.
2. Clustering Model:
  - Model Selection: K-means clustering was utilized for its efficiency in partitioning data into specified clusters. The optimal number of clusters (k) was determined using the silhouette method.
  - Insights: The clustering analysis revealed six distinct groups of projects, highlighting patterns in goals and amounts pledged. For instance, projects with lower goals tended to receive higher pledges, suggesting strategic advantages for creators.

## The Results
- The GBA model provided an effective classification of project outcomes, with key insights into important features influencing success.
- The K-means clustering identified meaningful patterns across projects, allowing for strategic guidance for creators and informed decision-making for backers. Despite similar success rates across clusters, the analysis offers valuable insights into funding strategies, enhancing the understanding of Kickstarter's crowdfunding dynamics.
