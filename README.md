# Catology
AI project for recognizing cat breeds from a brief description, and natural language.

Project created by Negoita Mihai, Radu Roxana & Zara Mihnea

## Technologies used:
- **Pandas**, **Numpy** for calculations and data manipulation
- **Scikit-learn** for training models and data preprocessing
- **Matplotlib** for generating graphs for data analysis
- [ **Optional** ]: **Keras** for neural networks, if we decide to use neural networks as well

## Final objective 
An application that receives text from the user, and generates:
- a natural language description of a cat breed
- or a description of a breed
- or compares two cat breeds
## Application architecture
Each Container is a Python process. The Python processes in the diagram are controlled and used by a main script that serves as a communication environment between processes

![AI_Project drawio (4)](https://github.com/user-attachments/assets/2fb0fcdc-d9d7-49b2-8f9d-f74dab362e54)

### Dataset Data manipulation / analysis / preprocessing / Features Engineering steps(also lab exercises)
1. Create program that tells if there are duplicates or null values
2. Handle duplicates
3. Handle null values
4. Print the number of instances for each class
6. Find out how many distinct values there are for each attribute and print number of values and frequency for each value, for each class
7. Generate graphs using matplotlib to visualize the dataset and find correlations between attributes
8. Create preprocessing pipeline(scaling, one-hot encoding, feature engineering, and whatever else needed
9. Consider Features Engineering ideas like combining attributes, removing attributes, adding new attributes.
10. Augment the dataset with new instances using methods like SMOTE or undersampling or oversampling, fix the dataset if unbalanced
11. Consider removing outliers
12. Consider Combining this dataset with other datasets

### Preprocessing pipeline for preprocessor container
1. Scale input data
2. One hot encode categorical attributes
3. Remove irrelevant attributes
4. Add features engineering steps to pipeline if using

### Train the model(s)
1. Create a test set from the dataset that models never see during training
2. Train multiple classifiers on the dataset and use the best ones
3. Always validate models properly when training, using cross-validation or other methods
4. Tune model hyperparameters using Grid Search and Randomizer Search
5. Combine the best classifiers using hard or soft voting
6. Save the model(s) on the disk for later usage
