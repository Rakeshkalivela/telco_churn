from trainer import TelcoChurn

def main():

    tel = TelcoChurn("../data/WA_Fn-UseC_-Telco-Customer-Churn.csv", random_state=10)

    X_train, X_test, y_train, y_test  = tel.preprocess()

    neg, pos = (tel.y_train == 0).sum(), (tel.y_train == 1).sum()
    scale = neg / pos

    tel.XGBmodel(scale_pos_weight=scale)

    tel.model_fit()

    tel.evaluate()

if __name__ == "__main__":
    main()
