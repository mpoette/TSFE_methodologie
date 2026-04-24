from sklearn.metrics import classification_report
import numpy as np
def utilisationCLF(clf, X_train, y_train, X_test, y_test, le, typeclf = "Random Forest",  printer = False):
    #class_weight = balanced pour donner autant d'importance aux sains minoritaires qu'aux dysfonctionnels majoritaires...
    X = np.concatenate([X_train,X_test])
    y_encoded = np.concatenate([y_train,y_test])
    clf.fit(X_train, y_train)
    y_pred_nb_train = clf.predict(X_train)
    y_pred_nb_test = clf.predict(X_test)
    print(f"{typeclf} :")
    
    train_score=clf.score(X_train,y_train)
    
    test_score=clf.score(X_test,y_test)
    print(f"Le score sur les données de test est {test_score}")
    print(classification_report(y_test, y_pred_nb_test, target_names=["Left Dys", "Right Dys", "Healthy"], zero_division=0))
    print("test réussi avec un score de : ", test_score)
    return test_score