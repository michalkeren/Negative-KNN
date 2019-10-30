# Preview
this program's goal is to assist in the research of professor Shai Avidan, in the theory that "Negative KNN" might improve the performance of the classic classification algorithm- KNN, under some conditions, and also compare it to the "ENN & KNN" algorithm. 
and also compare its accuracy to "ENN & KNN" algorithm.

In Negative KNN, a selected amount of neighbors [NEG_K], are not included in the decision making of the classification process.
Meaning for each item in the testing set, NEG_K neighbors from every class, that are closest to the item, are ignored during the prediction process.

# Usage 
The program returns a calculation for either a data file, for instance "iris.data", or data in a form of two customized gaussians.

In order to select the type of data and modify the variables ,scroll to the bottom of the .py file.
After activating one of the main functions, run the program. 
The program will print the accuracy results for each test in the following form:

---------------------------------


_tested algorithm 


TrainingSet Accuracy: ___    (TrainingSet is also the testing set in that case)


TestSet Accuracy: ___


---------------------------------
