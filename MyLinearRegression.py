import numpy as np
import random
import sklearn
from sklearn.datasets.samples_generator import make_regression
import pylab
from scipy import stats




def linearRegression(alpha,x,y,ep = 0.0001,max_iteration =1000):
    converged =False
    iteration = 0
    numberOfSample = x.shape[0]

    #initialize thera
    theta0 = np.random.random(x.shape[0])
    theta1 = np.random.random(x.shape[0])
    J = sum([(theta0 + theta1 * x[i] - y[i]) ** 2 for i in range(numberOfSample)])


    while not converged:
        # grad0 = 1.0 / m * sum([(t0 + t1 * x[i] - y[i]) for i in range(m)])
        # grad1 = 1.0 / m * sum([(t0 + t1 * x[i] - y[i]) * x[i] for i in range(m)])

        d0= 1/numberOfSample * sum([(theta0+theta1*x[i])-y[i] for i in range(numberOfSample)])
        d1= 1/numberOfSample * sum([(theta0+theta1*x[i])-y[i]*x[i] for i in range(numberOfSample)])

        t0 = theta0-alpha*d0
        t1= theta1-alpha*d1

        theta0 = t0
        theta1 =t1

        e = sum([(d0 + d1 * x[i] - y[i]) ** 2 for i in range(numberOfSample)])

        if abs(J.all() - e.all()) <= ep :
            print (  'Converged, iterations: ', iter, '!!!')
            converged = True


        J= e;
        iteration+=1

        if iteration>max_iteration:
            print("Maximum Iteration")
            break;

    return d0,d1;



if __name__== "__main__":
    x, y = make_regression(n_samples=1000, n_features=1, n_informative=1,
                           random_state=0, noise=355)
    print('x.shape = %s y.shape = %s' % (x.shape, y.shape))

    alpha = 0.01
    ep = 0.01
    max_iteration = 1000
    theta0, theta1 = linearRegression(alpha, x, y, ep, 1000)
    print(theta0)
    print(theta1)
    # slope, intercept, r_value, p_value, slope_std_error = stats.linregress(x[:, 0], y)
    # print(('intercept = %s slope = %s') % (intercept, slope))

    # plot
    for i in range(x.shape[0]):
        y_predict = theta0 + theta1 * x

    pylab.plot(x, y, 'o')
    pylab.plot(x, -y_predict, 'k-')
    pylab.show()
    print("Done!")
