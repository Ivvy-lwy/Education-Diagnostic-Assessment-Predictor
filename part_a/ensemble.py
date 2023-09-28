# TODO: complete this file.
import numpy as np
from utils import *
from part_a.item_response import *


def bootstrap(data, num):
    new_data = []
    size = len(data["question_id"])

    for n in range(num):
        temp = np.random.choice(size, size, replace=True)
        new_dict = {
            "user_id": [],
            "question_id": [],
            "is_correct": [],
        }
        for i in temp:
            new_dict['user_id'].append(data["user_id"][i])
            new_dict['question_id'].append(data["question_id"][i])
            new_dict['is_correct'].append(data["is_correct"][i])
        new_data.append(new_dict)

    return new_data


def bagging_predictions(data, theta, beta, num):
    prob = np.zeros((num, len(data[0]["question_id"])))
    for n in range(num):
        for i, q in enumerate(data[n]["question_id"]):
            u = data[n]["user_id"][i]
            x = (theta[n][u] - beta[n][q]).sum()
            p_a = sigmoid(x)
            prob[n][i] = p_a
    return prob.mean(axis=0)


def evaluate(data, prob):
    pred = np.where(prob >= 0.5, 1, 0)
    return np.sum((data["is_correct"] == np.array(pred))) / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    sample_num = 3

    # Bootstrap the training set.
    sample_data = bootstrap(train_data, sample_num)

    # Train item_response model
    lr = 0.01
    iterations = 50
    theta_list = []
    beta_list = []

    for i in range(sample_num):
        theta, beta, train_lld, val_lld = irt(sample_data[i], val_data, lr, iterations)
        theta_list.append(theta)
        beta_list.append(beta)

    train_prob = bagging_predictions(sample_data, theta_list, beta_list, sample_num)
    val_prob = bagging_predictions([val_data for i in range(sample_num)], theta_list, beta_list, sample_num)
    test_prob = bagging_predictions([test_data for i in range(sample_num)], theta_list, beta_list, sample_num)

    train_scores = []
    val_scores = []
    test_scores = []

    for n in range(sample_num):
        train_scores.append(evaluate(sample_data[n], train_prob))
        val_scores.append(evaluate(val_data, val_prob))
        test_scores.append(evaluate(test_data, test_prob))

    train_avg_score = np.array(train_scores).mean(axis=0)
    val_avg_score = np.array(val_scores).mean(axis=0)
    test_avg_score = np.array(test_scores).mean(axis=0)
    print("Train average accuracy is {}.".format(train_avg_score))
    print("Validation average accuracy is {}.".format(val_avg_score))
    print("Test average accuracy is {}.".format(test_avg_score))


if __name__ == "__main__":
    main()
