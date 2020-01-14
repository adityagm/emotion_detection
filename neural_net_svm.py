from sklearn import svm

class NeuralNetSvm():

    def __init__(self):
##        self.gamma = 0.01
##        self. C = 70
        global clf
        # initialise the classifier
        clf = svm.LinearSVC()
        self.x = 0
        self.y = 0
        
    def fit(self, data, labels):
##        print(len(data))
        
        self.x, self.y = data[:], labels[:]
        clf.fit(self.x, self.y)

    def predict(self, landmarks):
        prediction = clf.predict(landmarks)
        return prediction
    
