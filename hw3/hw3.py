import numpy as np
from scipy.stats import rankdata


# Given the difference between the difference in features between the dictionary object of people and the restaurant  features, it's necessary to eliminate the "indian food", "mexican food", and "hipster points" features(not entirely sure what that last one even means).


# Transform the user data into a matrix ( M_people). Keep track of column and row IDs. 

M_people = np.random.randint(10, size=(10,6))

names = ['Raisel', 'Gerlach','Jonathan','Kai','Telmo','Melantha','Charleen','Aria','Milo','Aral']


# Transform the restaurant data into a matrix (M_resturants) using the same column index.

M_restaurants = np.random.randint(10, size=(8,6))

rest_names = ['Moon Wok','Sun of a Bun','Drink & Dive','Lets Ketchup','Nacho Daddy','Sea You Soon','Unphogettable','Turnip the Beat']
features = ['distance','novelty','cost','avg_rating','cuisine','vegetarian']


# Informally describe what a linear combination is  and how it will relate to our restaurant matrix.


# Choose a person and compute(using a linear combination) the top restaurant for them.  What does each entry in the resulting vector represent? 
def find_best_per(person, people_names, people_matrix, restaurants, restaurant_matrix):
    if isinstance(person, str): 
        name=person
        person=names.index(person)
    else:
        name=names[person]
    #print(np.dot(restaurant_matrix, people_matrix[person]))
    place = np.argmax(np.dot(restaurant_matrix,people_matrix[person]))
    best_res=restaurants[place]
    print(f'Best restaurant for {name} is {best_res}')



def total_matrix(mat_rest, mat_people):
    M_user_x_rest=np.array([])
    for i in range(len(mat_people)):
        if i == 0:
            M_user_x_rest = np.dot(mat_rest,mat_people[i])
        else:
            mat = np.dot(mat_rest, mat_people[i])
            M_user_x_rest = np.vstack((M_user_x_rest, mat))

    print(M_user_x_rest)
    return M_user_x_rest


def total_rest():
   for i in range(len(names)):
        find_best_per(names[i],names, M_people, rest_names, M_restaurants)

def a_ij():
    M_user_x_rest=np.array([])
    for i in range(len(M_restaurants)):
        if i == 0:
            M_user_x_rest = np.dot(M_people, M_restaurants[i])
        else:
            mat = np.dot(M_people, M_restaurants[i])
            M_user_x_rest = np.vstack((M_user_x_rest, mat))

    print(M_user_x_rest)



def best_rest_overall(matrix, restaurants):
    best_list = matrix.sum(axis=0)
    print(best_list)
    ind = np.argmax(best_list)
    best_restaurant = restaurants[ind]
    print(f'The best overall restaurant is {best_restaurant}')


def ranked_matrix(matrix, restaurants):
    ranked_mat = np.array([])
    for i in range(len(matrix)):
        if i ==0:
            ranked_mat = (rankdata(matrix[i]-1).astype(int))
        else:
            rank = (rankdata(matrix[i]) - 1).astype(int)
            ranked_mat = np.vstack((ranked_mat, rank))
    print(ranked_mat)
    best_rest_overall(ranked_mat, restaurants)

def main():
    print(f'Linear Combination calculating top restaurant for each person:')
    total_rest()
    print('\n')
    print(f'Stack of dot product for all people')
    totes = total_matrix(M_restaurants, M_people)
    #print('\n')
    #print(f'Same stack, but switched axes')
    #a_ij()
    print('\n')
    best_rest_overall(totes, rest_names)
    print('\n')
    print(f'Matrix of Ranked Restaurants (Higher is better)')
    ranked_matrix(totes, rest_names)
    print('\n')
    M=np.random.randint(10,size=(10,5))
    X=np.random.randint(10,size=(8,5))
    totes2 = total_matrix(X,M)
    best_rest_overall(totes2, rest_names)

if __name__=="__main__":
    main()

