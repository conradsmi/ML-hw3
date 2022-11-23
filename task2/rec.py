import numpy as np

import matplotlib.pyplot as plt

from surprise import Dataset, Reader, accuracy, similarities
from surprise.prediction_algorithms.matrix_factorization import SVD
from surprise.prediction_algorithms.knns import KNNBasic
from surprise.model_selection import KFold

_SEED = 0

def train(models, metrics, runs, data):
    run_accuracies = np.zeros((len(models), runs, len(metrics)))
    for i, model in enumerate(models):
        print(f'\nRunning model {i+1} out of {len(models)+1}')
        kf = KFold(n_splits=runs, shuffle=True, random_state=_SEED)
        for j, (train, test) in enumerate(kf.split(data)):
            model.fit(train)
            predictions = model.test(test)
            for k, a in enumerate(metrics):
                run_accuracies[i,j,k] = a(predictions, verbose=True)

    average_accuracies = np.mean(run_accuracies, axis=1)
    print(average_accuracies)
    return average_accuracies

# reference:
# https://www.pythoncharts.com/matplotlib/grouped-bar-charts-matplotlib/
def plot(accuracies, metrics_names, model_names, width=0.3):
    fig, ax = plt.subplots()
    x = np.arange(len(accuracies))
    for i, m in enumerate(metrics_names):
        b = ax.bar(x + i*width, accuracies[:, i], width=width, label=m)
    for b in ax.patches:
        value = b.get_height()
        x_text = b.get_x() + b.get_width()
        y_text = b.get_y() + value
        ax.text(x_text, y_text, f'{round(value, 3)}')
        
    ax.set_xticks(x + (len(metrics_names)*width) / 2)
    ax.set_xticklabels(model_names)
    ax.yaxis.grid(True, color='#AAAAAA')
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.legend(loc='lower right')
    fig.tight_layout()
    plt.show()

    # b1 = ax.bar(x, accuracies[:, 0], width=width, label=metrics[0].__name__)
    # b2 = ax.bar(x + width, accuracies[:, 1], width=width, label=metrics[1].__name__)
    

if __name__ == '__main__':
    reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
    data = Dataset.load_from_file('data/ratings_small.csv', reader=reader)
    
    # part C and D
    runs = 5
    PMF_model = SVD(biased=False, random_state=_SEED, verbose=False)
    user_model = KNNBasic(sim_options={"user_based": True}, random_state=_SEED, verbose=False)
    item_model = KNNBasic(sim_options={"user_based": False}, random_state=_SEED, verbose=False)

    metrics = [accuracy.rmse, accuracy.mae]
    models = [PMF_model, user_model, item_model]
    '''acc = train(models, metrics, runs, data)'''

    # part E
    sim_measures = [similarities.cosine, similarities.msd, similarities.pearson]

    user_models = [KNNBasic(sim_options={"name": s.__name__, "user_based": True}, random_state=_SEED) for s in sim_measures]
    user_model_names = [f'user_{s.__name__}' for s in sim_measures]
    item_models = [KNNBasic(sim_options={"name": s.__name__, "user_based": False}, random_state=_SEED) for s in sim_measures]
    item_model_names = [f'item_{s.__name__}' for s in sim_measures]

    models = user_models + item_models
    metrics_names = [a.__name__ for a in metrics]
    model_names = user_model_names + item_model_names 

    '''acc = train(models, metrics, runs, data)
    plot(acc, metrics_names, model_names)'''

    # part F and G
    
    user_k_search_models = [KNNBasic(k=k, sim_options={"name": 'msd', "user_based": True}, verbose=False) for k in range(5, 100, 5)]
    user_acc = train(user_k_search_models, [accuracy.rmse], runs, data)
    plot(user_acc, [accuracy.rmse], [str(k) for k in range(5, 100, 5)], width=0.2)

    item_k_search_models = [KNNBasic(k=k, sim_options={"name": 'msd', "user_based": False}, verbose=False) for k in range(5, 100, 5)]
    item_acc = train(item_k_search_models, [accuracy.rmse], runs, data)
    plot(item_acc, [accuracy.rmse], [str(k) for k in range(5, 100, 5)], width=0.2)

    best_k_user = np.argmin(user_acc)
    best_k_item = np.argmin(item_acc)
    print(f'Best k of user model is {5*(best_k_user+1)} & RMSE is {user_acc[best_k_user,0]}')
    print(f'Best k of item model is {5*(best_k_item+1)} & RMSE is {item_acc[best_k_item,0]}')

    