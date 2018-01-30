def indicator(function_array_to_be_indicated, its_domain, barrier):
    """the indicator influences the function argument, not value. So here it iterates through x-domain and cuts any
    values of function with an argument less than H"""
    indicated = []
    for index in range(len(its_domain)):
        if its_domain[index] > barrier:
            indicated.append(function_array_to_be_indicated[index])
        else:
            indicated.append(0)
    return indicated

def G(S, K):
    """the payoff function of put option. Nothing to do with barrier"""
    return max(K-S, 0)
