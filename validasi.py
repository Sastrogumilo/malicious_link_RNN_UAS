from sklearn import model_selection

class Cross_Validasi:
    X_test = None
    X_train = None
    target_train = None
    target_test = None
    def __init__(self, X, target):
        self.X = X
        self.target = target

    def value(self):

        Cross_Validasi.X_train, \
            Cross_Validasi.X_test, \
                Cross_Validasi.target_train, \
                    Cross_Validasi.target_test = \
            model_selection.train_test_split(self.X, self.target, test_size=0.25, random_state=33)

        return Cross_Validasi.X_train, Cross_Validasi.X_test, \
            Cross_Validasi.target_test, Cross_Validasi.target_train