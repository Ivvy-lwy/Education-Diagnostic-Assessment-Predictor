from utils import *

import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.

    for i in range(len(data["user_id"])):
        u = data["user_id"][i]
        q = data["question_id"][i]
        c = data["is_correct"][i]

        log_lklihood += c * (theta[u] - beta[q]) - np.log(1 + np.exp(theta[u] - beta[q]))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta, c_matrix, in_data_matrix):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    t = theta.copy()
    b = beta.copy()
    diff = np.multiply(c_matrix - sigmoid(np.subtract.outer(t[:, 0], b[:, 0])), in_data_matrix)
    theta += lr * np.dot(diff, np.ones((1774, 1)))
    beta += lr * np.dot((-1 * diff.T), np.ones((542, 1)))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations, c_matrix, in_data_matrix):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    theta = np.full((542, 1), 0.5)
    beta = np.full((1774, 1), 0.5)

    val_acc_lst = []
    neg_lld_train = []
    neg_lld_val = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta[:, 0], beta=beta[:, 0])
        neg_lld_train.append(neg_lld)
        neg_lld_val.append(neg_log_likelihood(val_data, theta=theta[:, 0], beta=beta[:, 0]))
        score = evaluate(data=val_data, theta=theta[:, 0], beta=beta[:, 0])
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta, c_matrix, in_data_matrix)
    plt.figure(1)
    plt.plot([i + 1 for i in range(iterations)], neg_lld_train)
    plt.plot([i + 1 for i in range(iterations)], neg_lld_val)
    plt.legend(["Training", "Validation"])
    plt.xlabel("Iterations")
    plt.ylabel("Negative Log-Likelihood")
    plt.show()
    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    c_matrix = np.zeros((542, 1774))  # is_correct matrix
    in_data_matrix = np.zeros((542, 1774))  # if observation is in data

    users = train_data["user_id"]
    questions = train_data["question_id"]
    is_correct = train_data["is_correct"]
    for i in range(len(users)):
        u = users[i]
        q = questions[i]
        c = is_correct[i]

        c_matrix[u, q] = c
        in_data_matrix[u, q] = 1

    num_iterations = 50
    lr = 0.1
    theta, beta, val_acc_lst = irt(train_data, val_data, lr, num_iterations, c_matrix, in_data_matrix)

    fig1 = plt.figure()
    ax = fig1.add_axes([0, 0, 1, 1])
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Accuracy")
    plt.plot([i + 1 for i in range(num_iterations)], val_acc_lst, "-g", label="validation")
    plt.legend(loc="upper right")
    plt.show()

    max_i = np.argmax(np.array(val_acc_lst))
    print(
        "The iteration value with the highest validation accuracy is " + str(max_i + 1) + " with an accuracy of " + str(
            val_acc_lst[max_i]))
    print("The test accuracy is " + str(evaluate(test_data, theta, beta)))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()