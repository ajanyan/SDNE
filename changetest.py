# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 10:03:13 2019

@author: ACER
"""

f1 = open("new1.txt", "r",errors='ignore')
input= open("Author.txt","r",errors='ignore') 
lines = tuple(input)  # Slurps all lines from file
lines1=tuple(f1)
t=0
c=0
while lines1:
    listnum=[]
    date1=lines1[t]
    t=t+1
    author1=lines1[t]
    t=t+1
    list1=author1.split(';')
    #list1[0]=list1[0][3:]
    length=len(list1)
    print(length)
    j=0
    
    while j<length:
        k=-1
        print(list1[j])
        
         
        for i, linek in enumerate(lines):
                if list1[j] in linek[3:]:
                    k= lines[i-1]
                    print(k)
                    k=k[7:]
                    print(k)
                    input.close()
                    break
        if k!=-1:    
                listnum.append(k)
        j=j+1
      
    with open('c.txt', 'a') as f:
        f.write("date:%s\n" %(date1[:-1]))
        c=c+1
        for item in listnum:
            f.write("%s;" %( item[:-1]))
        f.write("\n")
    
    print("step complete")
    
f1.close()
print("completed")