from utils import *

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def split_data_premium(data):
    premium = {
        "user_id": [],
        "question_id": [],
        "is_correct": []
    }
    student_meta = load_meta_csv("../data")
    ids = []
    for i in range(len(student_meta["user_id"])):
        premium_pupil = student_meta["premium_pupil"][i]
        user_id = student_meta["user_id"][i]
        if premium_pupil == 1:
            ids.append(user_id)

    for idx, uid in enumerate(data["user_id"]):
        if uid in ids:
            premium["user_id"].append(uid)
            premium["question_id"].append(data["question_id"][idx])
            premium['is_correct'].append(data["is_correct"][idx])
    return premium


def neg_log_likelihood(data, theta, beta, lamb):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param lamb:
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
    for idx, uid in enumerate(data["user_id"]):
        c = data["is_correct"][idx]
        qid = data["question_id"][idx]
        x = lamb[qid] * theta[uid] - lamb[qid] * beta[qid]
        log_lklihood += c * x - np.log(1 + np.exp(x))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta, lamb):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param lamb:
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
    theta_der = np.zeros((len(theta), 1))
    beta_der = np.zeros((len(beta), 1))
    lamb_der = np.ones((len(lamb), 1))

    for idx, uid in enumerate(data["user_id"]):
        c = data["is_correct"][idx]
        qid = data["question_id"][idx]
        x = lamb[qid] * theta[uid] - lamb[qid] * beta[qid]
        theta_der[uid] -= c * lamb[qid] - lamb[qid] * sigmoid(x)
        beta_der[qid] -= -c * lamb[qid] + lamb[qid] * sigmoid(x)
        lamb_der[qid] -= c * theta[uid] - c * beta[qid] - (theta[uid] - beta[qid]) * sigmoid(x)
    theta -= lr * theta_der
    beta -= lr * beta_der
    lamb -= lr * lamb_der

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta, lamb


def irt(data, val_data, lr, iterations):
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
    theta = np.zeros((542, 1))
    beta = np.zeros((1774, 1))
    lamb = np.ones((1774, 1))

    val_acc_lst = []
    train_neg_lld = []
    val_neg_lld = []

    for i in range(iterations):
        train_neg_lld.append(neg_log_likelihood(data, theta=theta, beta=beta, lamb=lamb))
        val_neg_lld.append(neg_log_likelihood(val_data, theta=theta, beta=beta, lamb=lamb))
        score = evaluate(data=val_data, theta=theta, beta=beta, lamb=lamb)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(train_neg_lld, score))
        theta, beta, lamb = update_theta_beta(data, lr, theta, beta, lamb)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, lamb, train_neg_lld, val_neg_lld


def evaluate(data, theta, beta, lamb):
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
        x = (lamb[q] * theta[u] - lamb[q] * beta[q]).sum()
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
    lr = 0.01
    iterations = 50
    premium_train = split_data_premium(train_data)
    premium_val = split_data_premium(val_data)
    premium_test = split_data_premium(test_data)

    theta, beta, lamb, train_lld, val_lld = irt(premium_train, premium_val, lr, iterations)

    # plt.plot(train_lld, label="train")
    # plt.plot(val_lld, label="validation")
    # plt.xlabel('Iterations')
    # plt.ylabel('Negative Log Likelihood')
    # plt.legend()
    # plt.show()

    val_score = evaluate(premium_val, theta, beta, lamb)
    test_score = evaluate(premium_test, theta, beta, lamb)
    print("Validation accuracy is {}, test accuracy is {}.".format(val_score, test_score))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    # selected_questions = [1, 10, 100]
    # theta = theta.reshape((542,))
    # theta.sort()
    # for idx, q in enumerate(selected_questions):
    #     x = lamb[q] * (theta - beta[q])
    #     plt.plot(theta, sigmoid(x), label=f"j{idx + 1}={q}")
    # plt.xlabel('Theta')
    # plt.ylabel('Probability')
    # plt.legend()
    # plt.show()

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
