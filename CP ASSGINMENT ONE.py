#!/usr/bin/env python
# coding: utf-8

# ## NDLOVU SIPHUMELELE
# ## PHYS736 CP ASSIGNMENT 1
# ## 217047276

# In[ ]:


#Question 1
num=int(input("Enter any Integer"))
if ((num%3==0) and (num%5==0)):                 #condition 1
    
    print("BlackPink")   
    
elif num%3==0:   # Executed if elif(else if) is True, then skip the rest At the end, after one or more ‘elif’ blocks
    print("Black")
        
elif num%5==0:
    print("Pink")
elif num%2==0:
    print("kniPkcalB")


# In[ ]:


#Question 2

myNumbers=[]                            #1
x = input("Enter a number:")            #2
while x !="*":                          #3
    f=float(x)
    myNumbers.append(f)
    x = input("Enter a number:")
print(myNumbers)







# In[2]:


#in this code we take "2" and "*" as and example,so what will happen is when we put 2 inside our function it is going to print 2
# and it will go back to a function and try to convert it to float and since 2 can be converted to float its going to return true
# and when we take "*" and put it in our function it is going to print "*" and try to convert it to float and since "*"
#cannot converted to float it is going to ecxept the error and then return false.

def is_float ( element ):
    try :
        float ( element )
        return True
    except ValueError :
        return False
print(is_float ( "*"))


# In[4]:


import matplotlib.pyplot as plt
import numpy as np

myNumbers=[]
x=input("Enter a number")
while is_float(x): #5
    f=float(x)
    myNumbers.append(f)
    x=input("Enter a number")
print(myNumbers)

def my_mean(myNumbers):
    mean=sum(myNumbers)/len(myNumbers)
    return mean

   
def my_median(myNumbers):
    n=len(myNumbers)
    i=n//2
    #an odd number of observations sample
    if n%2==1:
        return sorted(myNumbers)[i]
    #an even number of observations sample
    return sum(sorted(myNumbers)[i-1:i+1])/2

def variance(myNumbers,ddof=1):
    n=len(myNumbers)
    mean=sum(myNumbers)/n
    return sum(np.array([(x-mean)**2 for x in myNumbers]))/(n-ddof)


def stdev(myNumbers):
    var=variance(myNumbers)
    std_dev=np.sqrt(var)
    return std_dev
    
print("The mean is :",my_mean(myNumbers))
print("The median is:",my_median(myNumbers))
print("The stdev is: ",stdev(myNumbers))

n,bins,patches=plt.hist(myNumbers,bins=5)
plt.xlabel("Temparature")
plt.ylabel("Number of days")
plt.title("Numbers of days versus Temparature")
plt.show()


# In[6]:


#question 3
import matplotlib.pyplot as plt
import numpy as np

debtFund0 = 100000 # debt to pay off

interestRate = (6.5/12)/100 # annual interest rate divided for monthly basis

months = 5*12 # months over 5 years
monthsList = list(range(months+1))

# monthly payment amount options
onethou = 1000.0
twothou = 2000.0
threethou = 3000.0
fourthou = 4000.0

# progression of debt
debtBy1k = [debtFund0]
debtBy2k = [debtFund0]
debtBy3k = [debtFund0]
debtBy4k = [debtFund0]

# debt payment by R1,000.00  
n1k = 0
while n1k < months:
    debt1k = debtBy1k[n1k] - 1000.0
    debt1k += (debt1k * interestRate)
    debtBy1k.append(debt1k)
    n1k += 1
    
# debt payment by R2,000.00    
n2k = 0
while n2k < months:
    debt2k = debtBy2k[n2k] - 2000.0
    debt2k += (debt2k * interestRate)
    debtBy2k.append(debt2k)
    n2k += 1

# debt payment by R3,000.00  
n3k = 0
while n3k < months:
    debt3k = debtBy3k[n3k] - 3000.0
    debt3k += (debt3k * interestRate)
    debtBy3k.append(debt3k)
    n3k += 1

# debt payment by R4,000.00  
n4k = 0
while n4k < months:
    debt4k = debtBy4k[n4k] - 4000.0
    debt4k += (debt4k * interestRate)
    debtBy4k.append(debt4k)
    n4k += 1

fig, ax = plt.subplots()
   
graph1k = ax.plot(monthsList, debtBy1k, label="R1,000.00")
graph2k = ax.plot(monthsList, debtBy2k, label="R2,000.00")
graph3k = ax.plot(monthsList, debtBy3k, label="R3,000.00")
graph4k = ax.plot(monthsList, debtBy4k, label="R4,000.00")

ax.legend()
ax.set_xlim([0, 60])
ax.set_ylim([0, 100000])
ax.set_xlabel("Number of Months")
ax.set_ylabel("Debt")

plt.show()


# In[7]:


#b
i=0
m=1000
while i < 4:
    m2=m+(m*0.065/12)
    b=int(100000/m2)/12
    b2=(b)%1
    b3=b2*10
    print("Repayment in ",int(b),"years",int(b3),"months")
    m=m+1000
    i+=1
    


# In[35]:


#c
i=0
m=1000

while i < 4:
    if m >1000000: 
         print("Repayment in ",int(b),"years",int(b3),"months")
    else:
          print("Debt can never be paid")
    
    m2=m+(m*0.065/12)
    b=int(100000/m2)/12
    b2=(b)%1
    b3=b2*10
    m=m+1000
    i+=1


# In[8]:


import urllib.request
from PIL import Image

urllib.request.urlretrieve('https://raw.githubusercontent.com/AmyCPUKZN/Task_1/main/secret_message.png', "secret_message.png")
img = Image.open("secret_message.png")

imgplot = plt.imshow(img)


# In[ ]:




