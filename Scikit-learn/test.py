from sklearn import tree

feature = [[178,1], [155,0], [177,1], [165,0], [169,1], [160,0]]
label = ['male', 'female', 'male', 'female', 'male', 'female']

clf = tree.DecisionTreeClassifier()
clf = clf.fit(feature, label)

print(clf.predict([[173,0]]))