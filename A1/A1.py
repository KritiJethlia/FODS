import random
import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt
import os.path, imageio

def dataset_generation(N, m):
    '''Dataset is generated by considering a particular value of MLE(m) and no of trials(N) '''
    heads = int(np.ceil(N*m)) 
    tails = N- heads
    dataset = ['H']*heads
    dataset += ['T']*tails
    random.shuffle(dataset)
    return dataset,heads,tails 
    

#Assuming a=2 and b=3 for beta distribution
def prior_dist(a, b ,m) :
    '''Prior distribution is considered is beta distribution without normalization factor'''
    return  pow(m,a-1)* pow(1-m,b-1)  #Beta function


def normalization(a,b): 
    '''normalization factor '''
    return gamma(a+b)/(gamma(a)*gamma(b))


def bernoulli(m,x) :
    '''bernoulli distribution formula'''
    return pow(m,x)*pow(1-m,1-x)

# we know the posterior will be bournoulli with a=a+head b=b+tail
def posterior(a,b, heads, tails):
    '''Calculating posterior for Lisa'''
    x = np.linspace(0,1,100)
    y = prior_dist(a,b,x)* pow(bernoulli(x,1),heads)* pow(bernoulli(x,0),tails)  # posterior ~=prior*likelihood
    plt.plot(x, normalization(a+heads,b+tails)*y, 'r--') #plotting 
    plt.xlabel("mu")
    plt.show()
    plt.savefig("Lisa"+'.png',dpi=90)
    print("Lisa's posterior estimate of getting head is " , (a+heads)/(a+b+heads+tails)) 


def create_gif(filenames, duration): 
    '''Function to create GIF out of images'''
    img = [] 
    for filename in filenames: 
    	img.append(imageio.imread(filename)) 
    output_file = 'Bob_dist_mle_0.5.gif'
    imageio.mimsave(output_file, img, duration=duration) 


def step_posterior(a,b,dataset):
    '''Calculating posterior distribution step by step for Bob'''
    x = np.linspace(0, 1, 100)
    trial = len(dataset)
    prior = prior_dist(a,b,x)
    # fig, ax = plt.subplots()
    # plt.plot(x,normalization(a,b)*prior,'r') 
    # plt.xlabel("mu")
    # plt.title("Initial distribution")
    # plt.close()
    # plt.savefig("0"+'.png',dpi=90) # storing each image of graph
    for i in range(0,trial):
        if(dataset[i]=='H') :
            prior = prior * bernoulli(x,1)
            a+=1
        else :
            prior = prior * bernoulli(x,0)
            b+=1
        # plt.plot(x,normalization(a,b)*prior,'r')
        # plt.xlabel("mu")
        # plt.title("Iteration "+str(i+1))
        # plt.savefig(str(i+1)+'_.png',dpi=90)
        # plt.close()
    # plt.plot(x,normalization(a,b)*prior)
    # plt.savefig('mu_0.5'+'.png',dpi=90)
    # plt.show()
    # duration = 0.3
    # filenames = sorted(filter(os.path.isfile, [x for x in os.listdir() if x.endswith(".png")]), key=lambda p: os.path.exists(p) and os.stat(p).st_mtime or time.mktime(datetime.now().timetuple())) 
    # create_gif(filenames,duration) #creating gif 
    print("Bob's posterior estimate of getting head is " , a/(a+b))


def main():
    dataset, heads, tails = dataset_generation(160,0.65)
    a=2
    b=3
    posterior(a, b, heads, tails)
    step_posterior(a,b,dataset)
main()