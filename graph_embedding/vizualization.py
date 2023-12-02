import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import ast
import pandas as pd
import seaborn as sns
import demjson


def retrieve_best_solution(system_path):
    suffix = '_std'
    df = pd.read_csv(system_path+'/results.csv', sep=';')
    df.rename(columns={'chm_value': 'chm_values'}, inplace=True)
    df.rename(columns={'chd_value': 'chd_values'}, inplace=True)
    df['chm_value'] = df['chm_values'].apply(lambda x: float(x.split(',')[0].split('(')[1]))
    df['chd_value'] = df['chd_values'].apply(lambda x: float(x.split(',')[0].split('(')[1]))
    numerical_columns = ['smq_value', 'cmq_value', 'chd_value', 'chm_value']

    std_columns = [col+suffix for col in numerical_columns]
    scaler = StandardScaler() # MinMaxScaler()
    df[std_columns] = scaler.fit_transform(df[numerical_columns])
    sns.set_palette("Set2")

    plt.figure(figsize=(10, 6))

    for col in numerical_columns:
        plt.plot(df['nb_clusters'], df[col+suffix], marker='o', label=col.split('_')[0].upper())

    plt.xlabel('Number of clusters')
    plt.ylabel('Standardized metrics')
    plt.title(f'Standardized metrics evolution with the number of clusters for {system_path.split("/")[-1]} project')
    plt.legend()
    plt.savefig(f'{system_path}/figure.png')
    # plt.show()


    # Find the best cluster
    df['criteria'] = df[std_columns].sum(axis=1)
    df_sorted = df.sort_values(by='criteria', ascending=False)
    best_cluster = demjson.decode(df_sorted.iloc[0]['community_dict'])
    print("This is the best clustering solution:")
    # for cluster, elements in best_cluster.items():
    #     print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> CLUSTER {cluster} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    #     for e in elements:
    #         print(f'    > {e} <')
    import csv
    with open(system_path+"/_bestCLuster.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        for cluster, elements in best_cluster.items():
            writer.writerow(elements)
    print('The best solution is saved saved to '+system_path+"/_bestCLuster.csv")



# system_path = './data/POS2'
# retrieve_best_solution(system_path)