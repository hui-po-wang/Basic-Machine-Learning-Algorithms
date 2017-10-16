import sys
import math
import scipy.special

def combination(n, r):
    fac = math.factorial
    return  fac(n) / (fac(r) * fac(n-r))

def beta_dis_density(theta, a, b):
    gamma = math.gamma
    return (gamma(a + b) / (gamma(a) * gamma(b))) * (theta ** (a - 1)) * ((1 - theta) ** (b - 1))

def binomial_dis_density(theta, N, k):
    return combination(N, k) * (theta ** k) * ((1 - theta) ** (N - k))

def compute_beta_prob(a, b, maximum, minimum):
    return scipy.special.betainc(a, b, maximum) - scipy.special.betainc(a, b, minimum)

if __name__ == '__main__':
    arguments = sys.argv
    fname = arguments[1]
    a = float(arguments[2])
    b = float(arguments[3])

    trials = []
    with open(fname, 'r') as f:
        for line in f.readlines():
            trials.append(line.strip())
    #print (trials)
    max_theta = -1e9
    min_theta = 1e9
    for i, t in enumerate(trials):
        N = len(t)
        k = len([head for head in t if head=='1'])
        theta = k / N
        #print("a", a, "b", b, "N", N, "k", k, "theta", theta)
        posterior_a = a + k
        posterior_b = b + N - k
        max_theta = max(theta, max_theta)
        min_theta = min(theta, min_theta)
        print ("Line [%d/%d]: Binomail likelihood: %f, theta: %.4f, Beta prior density: %f, beta posterior density: %f, probability piror: %.4f, posterior: %.4f, beta mean prior: %.4f, posterior: %.4f" 
          % (i+1,
            len(trials),
            binomial_dis_density(theta, N, k),
            theta,
            beta_dis_density(theta, a, b), 
            beta_dis_density(theta, posterior_a, posterior_b),
            compute_beta_prob(a, b, theta, 0), 
            compute_beta_prob(posterior_a, posterior_b, theta, 0),
            a/(a+b),
            posterior_a/(posterior_a+posterior_b)))

        a = posterior_a
        b = posterior_b
