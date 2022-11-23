def polynomial_regression_by_any_degree(degree:int, x:list, y:list) -> list:
    """
    This function performs a regression of any degree.

    Requirements
    --
        import numpy as np\n
        from sklearn.linear_model import LinearRegression\n
        from sklearn.preprocessing import PolynomialFeatures\n
        from sklearn.metrics import r2_score

    Parameters
    --
        degree: regression degree.
        x: numpy array.
        y: numpy array.

    Returns
    --
        list: [r2_score:int, predict_y:list, coef:list, intercept:int]
    """
    model = LinearRegression()
    pf = PolynomialFeatures(degree)

    x_fit = pf.fit_transform(x.reshape(-1, 1))

    model.fit(x_fit, y)

    predict = model.predict(x_fit)

    r2 = r2_score(y, predict)

    return [r2, predict, model.coef_, model.intercept_]
