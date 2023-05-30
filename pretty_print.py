"""
Generate a mathematical expression of the symbolic regression network (AKA EQL network) using SymPy. This expression
can be used to pretty-print the expression (including human-readable text, LaTeX, etc.). SymPy also allows algebraic
manipulation of the expression.
The main function is network(...)
There are several filtering functions to simplify expressions, although these are not always needed if the weight matrix
is already pruned.
"""
import pdb

import sympy as sp
import functions as functions


def apply_activation(W, funcs, n_double=0):
    """Given an (n, m) matrix W and (m) vector of funcs, apply funcs to W.

    Arguments:
        W:  (n, m) matrix
        funcs: list of activation functions (SymPy functions)
        n_double:   Number of activation functions that take in 2 inputs

    Returns:
        SymPy matrix with 1 column that represents the output of applying the activation functions.
    """
    W = sp.Matrix(W)
    if n_double == 0:
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                W[i, j] = funcs[j](W[i, j])
    else:
        W_new = W.copy()
        out_size = len(funcs)
        for i in range(W.shape[0]):
            in_j = 0
            out_j = 0
            while out_j < out_size - n_double:
                W_new[i, out_j] = funcs[out_j](W[i, in_j])
                in_j += 1
                out_j += 1
            while out_j < out_size:
                W_new[i, out_j] = funcs[out_j](W[i, in_j], W[i, in_j + 1])
                in_j += 2
                out_j += 1
        for i in range(n_double):
            W_new.col_del(-1)
        W = W_new
    return W


def sym_pp(W_list, funcs, var_names, threshold=0.01, n_double=None, add_bias=False, biases=None):
    """Pretty print the hidden layers (not the last layer) of the symbolic regression network

    Arguments:
        W_list: list of weight matrices for the hidden layers
        funcs:  dict of lambda functions using sympy. has the same size as W_list[i][j, :]
        var_names: list of strings for names of variables
        threshold: threshold for filtering expression. set to 0 for no filtering.
        n_double: list Number of activation functions that take in 2 inputs

    Returns:
        Simplified sympy expression.
    """
    vars = []
    for var in var_names:
        if isinstance(var, str):
            vars.append(sp.Symbol(var))
        else:
            vars.append(var)
    try:
        expr = sp.Matrix(vars).T

        if add_bias and biases is not None:
            assert len(W_list) == len(biases), "The number of biases must be equal to the number of weights."
            for i, (W, b) in enumerate(zip(W_list, biases)):
                W = filter_mat(sp.Matrix(W), threshold=threshold)
                b = filter_mat(sp.Matrix(b), threshold=threshold)
                expr = expr * W + b
                expr = apply_activation(expr, funcs[i + 1], n_double=n_double[i])

        else:
            for i, W in enumerate(W_list):
                W = filter_mat(sp.Matrix(W), threshold=threshold)  # Pruning
                expr = expr * W
                expr = apply_activation(expr, funcs[i + 1], n_double=n_double[i])
    except:
        pdb.set_trace()
    # expr = expr * W_list[-1]
    return expr


def last_pp(eq, W, add_bias=False, biases=None):
    """Pretty print the last layer."""
    if add_bias and biases is not None:
        return eq * filter_mat(sp.Matrix(W)) + filter_mat(sp.Matrix(biases))
    else:
        return eq * filter_mat(sp.Matrix(W))


def network(weights, funcs, var_names, threshold=0.01, add_bias=False, biases=None):
    """Pretty print the entire symbolic regression network.

    Arguments:
        weights: list of weight matrices for the entire network
        funcs:  dict of lambda functions using sympy. has the same size as W_list[i][j, :]
        var_names: list of strings for names of variables
        threshold: threshold for filtering expression. set to 0 for no filtering.

    Returns:
        Simplified sympy expression."""
    n_double = [functions.count_double(funcs_per_layer) for funcs_per_layer in funcs.values()]
    # translate operators to sympy operators
    sp_funcs = {}
    for key, value in funcs.items():
        sp_value = [func.sp for func in value]
        sp_funcs.update({key: sp_value})

    if add_bias and biases is not None:
        assert len(weights) == len(biases), "The number of biases must be equal to the number of weights - 1."
        expr = sym_pp(weights[:-1], sp_funcs, var_names, threshold=threshold, n_double=n_double, add_bias=add_bias, biases=biases[:-1])
        expr = last_pp(expr, weights[-1], add_bias=add_bias, biases=biases[-1])
    else:
        expr = sym_pp(weights[:-1], sp_funcs, var_names, threshold=threshold, n_double=n_double, add_bias=add_bias)
        expr = last_pp(expr, weights[-1], add_bias=add_bias)

    try:
        expr = expr[0, 0]
        return expr
    except Exception as e:
        print("An exception occurred:", e)



def filter_mat(mat, threshold=0.01):
    """Remove elements of a matrix below a threshold."""
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if abs(mat[i, j]) < threshold:
                mat[i, j] = 0
    return mat


def filter_expr(expr, threshold=0.01):
    """Remove additive terms with coefficient below threshold
    TODO: Make more robust. This does not work in all cases."""
    expr_new = sp.Integer(0)
    for arg in expr.args:
        if arg.is_constant() and abs(arg) > threshold:  # hack way to check if it's a number
            expr_new = expr_new + arg
        elif not arg.is_constant() and abs(arg.args[0]) > threshold:
            expr_new = expr_new + arg
    return expr_new


def filter_expr2(expr, threshold=0.01):
    """Sets all constants under threshold to 0
    TODO: Test"""
    for a in sp.preorder_traversal(expr):
        if isinstance(a, sp.Float) and a < threshold:
            expr = expr.subs(a, 0)
    return expr
