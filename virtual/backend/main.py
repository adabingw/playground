import tensorflow as tf 
import os 

from regression import sunk
from transformer_nn import runner

def init_path(): 
    module = input('module to run: ')
    print("running", module, "...")
    if module == 'regression': 
        print('\n')
        print('welcome to sunk! this module predicts your likelihood of surviving the titanic given your class, gender, and age.')
        print('\n')
        run = True 
        while run: 
            pclass = input('enter your class (1 is the best, 3 is the worst): ')
            age = input('enter age: ') 
            gender = input('enter m for male, f for female: ') 
            if not age.isdigit(): 
                print('age must be a digit')
            elif int(age) < 0 or int(age) > 100: 
                print('please enter a valid age')
            elif not 1 <= int(pclass) <= 3: 
                print('please enter a valid class') 
            elif gender != 'm' and gender != 'f': 
                print('please enter a valid gender')
            else: 
                gender = 1 if gender == 'm' else 0 
                prediction = sunk.pred(pclass, gender, age)
                print("your chances of surviving the titanic is ", prediction) 
                run = False 
                
    elif module == 'transformer': 
        translator = runner.new_translator() 
        sentence = input('enter something to translate to english from portuguese: ')
        runner.translate(translator=translator, sentence=sentence, ground_truth="no ground truth")
        print('trans')
    else: 
        print('cannot find module', module)

if __name__ == "__main__":
    # print(os.environ['PYTHONPATH'])
    init_path()